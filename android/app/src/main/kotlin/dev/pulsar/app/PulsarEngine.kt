package dev.pulsar.app

import android.content.Context
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import dev.pulsar.core.GoldenCheck
import dev.pulsar.core.GoldenResult
import dev.pulsar.core.ModelMeta
import dev.pulsar.core.PulsarPipeline
import dev.pulsar.core.RgbImage
import dev.pulsar.core.TraceNetRunner
import dev.pulsar.core.Vitals
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean

/** Everything loaded from assets: model metadata, the ONNX session, the face landmarker, the pipeline. */
class PulsarEngine(private val context: Context) : Closeable {
    val meta: ModelMeta = ModelMeta.parse(asset("model/model_meta.json").decodeToString())
    val runner: TraceNetRunner = TraceNetRunner(asset("model/tracenet.onnx"), meta, sessionOptions())
    val faces = FaceLandmarks(context)
    val pipeline = PulsarPipeline(meta, runner)

    private fun asset(path: String): ByteArray = context.assets.open(path).use { it.readBytes() }

    /** Parity of the on-device port vs the Python reference (golden files) + model latency. */
    fun selfTest(latencyRuns: Int = 30): SelfTestReport {
        val names = context.assets.list("golden").orEmpty().filter { it.endsWith(".json") }.sorted()
        val results = names.map { GoldenCheck.run(asset("golden/$it").decodeToString(), meta, runner) }
        val w = names.firstOrNull()?.let { GoldenCheck.window(JSONObject(asset("golden/$it").decodeToString()), meta.numPatches, meta.window) }
        val ms = if (w == null) {
            emptyList()
        } else {
            runner.run(w) // warm-up
            List(latencyRuns) {
                val t0 = System.nanoTime()
                runner.run(w)
                (System.nanoTime() - t0) / 1e6
            }.sorted()
        }
        return SelfTestReport(results, ms)
    }

    override fun close() {
        runner.close()
        faces.close()
    }

    companion object {
        /** XNNPACK on 2 threads, ORT's own pool on 1 (the recommended split when XNNPACK does the work). */
        fun sessionOptions() = TraceNetRunner.defaultOptions(threads = 1).apply {
            runCatching { addXnnpack(mapOf("intra_op_num_threads" to "2")) }
        }
    }
}

class SelfTestReport(val golden: List<GoldenResult>, private val sortedMs: List<Double>) {
    val passed: Boolean get() = golden.isNotEmpty() && golden.all { it.passed }
    val medianMs: Double get() = sortedMs.getOrElse(sortedMs.size / 2) { Double.NaN }
    val p90Ms: Double get() = sortedMs.getOrElse((sortedMs.size * 9) / 10) { Double.NaN }

    override fun toString(): String = buildString {
        appendLine(if (golden.isEmpty()) "No golden files in assets/golden." else golden.joinToString("\n"))
        if (sortedMs.isNotEmpty()) {
            appendLine()
            append("Model latency: median ${"%.1f".format(medianMs)} ms, p90 ${"%.1f".format(p90Ms)} ms (budget 50 ms)")
        }
    }
}

/** What the overlay draws for every frame. */
class FrameUpdate(val width: Int, val height: Int, val boxes: IntArray?, val faceFound: Boolean, val fps: Double)

/**
 * Camera frames -> pipeline on the analysis thread; about once a second a window snapshot goes to the
 * inference thread, so inference never stalls frame capture (dropped frames would cost signal).
 */
class VitalsAnalyzer(
    private val engine: PulsarEngine,
    private val inference: ExecutorService,
    private val onFrame: (FrameUpdate) -> Unit,
    private val onVitals: (Vitals, Double) -> Unit,
) : ImageAnalysis.Analyzer {
    private val conv = FrameConverter()
    private var t0Ns = Long.MIN_VALUE
    private var lastEstimateT = Double.NEGATIVE_INFINITY
    private val busy = AtomicBoolean(false)
    private val recent = ArrayDeque<Double>()

    override fun analyze(image: ImageProxy) {
        image.use {
            val ts = image.imageInfo.timestamp // sensor timestamp, ns, monotonic
            if (t0Ns == Long.MIN_VALUE) t0Ns = ts
            val t = (ts - t0Ns) / 1e9
            conv.convert(image)
            val bmp = conv.bitmap ?: return
            val lm = engine.faces.detect(bmp, (t * 1000).toLong())
            val st = engine.pipeline.onFrame(t, RgbImage(conv.width, conv.height, conv.pixels), lm)
            recent.addLast(t)
            while (recent.size > 60) recent.removeFirst()
            val fps = if (recent.size > 1) (recent.size - 1) / (recent.last() - recent.first()).coerceAtLeast(1e-6) else 0.0
            onFrame(FrameUpdate(conv.width, conv.height, st.boxes, st.faceFound, fps))

            if (t - lastEstimateT >= 1.0 && busy.compareAndSet(false, true)) {
                lastEstimateT = t
                val window = engine.pipeline.window()
                val face = st.faceFound
                inference.execute {
                    try {
                        val t1 = System.nanoTime()
                        val out = window?.let { engine.runner.run(it) }
                        val ms = if (out == null) Double.NaN else (System.nanoTime() - t1) / 1e6
                        onVitals(engine.pipeline.policy.decide(t, face, out), ms)
                    } finally {
                        busy.set(false)
                    }
                }
            }
        }
    }
}

/** CSV of estimates for side-by-side comparison with a fingertip oximeter (a sanity check, not evaluation).
 * Thread-safe: rows come from the inference thread, start/stop from the UI. */
class SessionLog(private val dir: File) {
    private var out: PrintWriter? = null
    var file: File? = null
        private set
    val active: Boolean get() = out != null

    @Synchronized
    fun start(): File {
        stop()
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val f = File(dir, "pulsar_session_$stamp.csv")
        out = PrintWriter(f.bufferedWriter()).also { it.println("t_s,status,hr_bpm,hr_mode_bpm,confidence,model_ms") }
        file = f
        return f
    }

    @Synchronized
    fun write(v: Vitals, ms: Double) {
        val o = out ?: return
        val m = v.output
        o.println(
            listOf(
                "%.2f".format(Locale.US, v.tSec),
                v.status.name,
                m?.let { "%.1f".format(Locale.US, it.hrBpm) } ?: "",
                m?.let { "%.0f".format(Locale.US, it.hrModeBpm) } ?: "",
                m?.let { "%.3f".format(Locale.US, it.confidence) } ?: "",
                if (ms.isNaN()) "" else "%.1f".format(Locale.US, ms),
            ).joinToString(","),
        )
        o.flush()
    }

    @Synchronized
    fun stop() {
        out?.close()
        out = null
    }
}
