package dev.pulsar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import dev.pulsar.core.Vitals
import dev.pulsar.core.VitalsStatus
import kotlin.math.max

/**
 * Draws over the camera preview: skin patches coloured by the model's attention (red = ignored,
 * green = trusted), the pulse waveform, HR / confidence or the abstain message, and the disclaimer.
 * Frame coordinates are mapped like PreviewView's FILL_CENTER, mirrored for the front camera.
 */
class VitalsOverlay(context: Context) : View(context) {
    private var frame: FrameUpdate? = null
    private var vitals: Vitals? = null
    private var modelMs = Double.NaN
    var logging = false
        set(value) {
            field = value
            invalidate()
        }

    private val dp = resources.displayMetrics.density
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * dp
    }
    private val panelPaint = Paint().apply { color = Color.argb(200, 15, 15, 15) }
    private val wavePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * dp
        color = Color.rgb(90, 220, 90)
    }
    private val bigText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f * dp
        isFakeBoldText = true
    }
    private val smallText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(210, 210, 210)
        textSize = 13f * dp
    }
    private val warnText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 200, 0)
        textSize = 12f * dp
    }
    private val path = Path()

    fun update(f: FrameUpdate) {
        frame = f
        postInvalidate()
    }

    fun update(v: Vitals, ms: Double) {
        vitals = v
        if (!ms.isNaN()) modelMs = ms
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val f = frame
        val v = vitals
        val alpha = v?.output?.alpha
        if (f?.boxes != null && f.width > 0) {
            val s = max(w / f.width, h / f.height)
            val ox = (w - f.width * s) / 2f
            val oy = (h - f.height * s) / 2f
            val amax = alpha?.maxOrNull()?.takeIf { it > 0f } ?: 1f
            val b = f.boxes
            for (p in 0 until b.size / 4) {
                val a = if (alpha != null && p < alpha.size) alpha[p] / amax else 0.5f
                boxPaint.color = Color.rgb((255 * (1 - a)).toInt(), (80 + 175 * a).toInt(), 80)
                val l = w - (b[4 * p + 2] * s + ox) // mirrored: the image's right edge is the view's left
                val r = w - (b[4 * p] * s + ox)
                canvas.drawRect(l, b[4 * p + 1] * s + oy, r, b[4 * p + 3] * s + oy, boxPaint)
            }
        }

        val panelH = 140f * dp
        val top = h - panelH
        canvas.drawRect(0f, top, w, h, panelPaint)
        val bvp = v?.output?.bvp
        if (bvp != null && bvp.isNotEmpty()) {
            val lo = bvp.min()
            val span = (bvp.max() - lo).coerceAtLeast(1e-9f)
            val x0 = 12f * dp
            val x1 = w * 0.55f
            path.reset()
            for (i in bvp.indices) {
                val x = x0 + (x1 - x0) * i / (bvp.size - 1).coerceAtLeast(1)
                val y = h - 20f * dp - (bvp[i] - lo) / span * (panelH - 50f * dp)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            canvas.drawPath(path, wavePaint)
        }
        val tx = w * 0.6f
        val status = when {
            f != null && !f.faceFound -> VitalsStatus.NO_FACE
            else -> v?.status ?: VitalsStatus.WARMING_UP
        }
        if (status == VitalsStatus.OK && v?.hrBpm != null) {
            canvas.drawText("%.0f bpm".format(v.hrBpm), tx, top + 60f * dp, bigText)
            canvas.drawText("confidence %.2f".format(v.output!!.confidence), tx, top + 88f * dp, smallText)
        } else {
            canvas.drawText(status.message, tx, top + 60f * dp, smallText)
        }
        val info = buildString {
            if (f != null) append("%.0f fps".format(f.fps))
            if (!modelMs.isNaN()) append("  model %.0f ms".format(modelMs))
            if (logging) append("  REC")
        }
        canvas.drawText(info, tx, top + 116f * dp, smallText)
        canvas.drawText("Research prototype - not a medical device", 12f * dp, 28f * dp, warnText)
    }
}
