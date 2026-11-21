package dev.pulsar.app

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

/**
 * CameraX RGBA_8888 frame -> upright packed ARGB pixels (+ a reusable Bitmap for MediaPipe), in one pass.
 *
 * Rotating ourselves keeps landmarks and pixel crops in the same upright coordinates (as in the Python
 * pipeline, which works on upright video). Buffers are reused, so steady state allocates nothing.
 */
class FrameConverter {
    var width = 0
        private set
    var height = 0
        private set
    var pixels = IntArray(0)
        private set
    var bitmap: Bitmap? = null
        private set
    private var bytes = ByteArray(0)

    fun convert(image: ImageProxy) {
        val rot = ((image.imageInfo.rotationDegrees % 360) + 360) % 360
        val w0 = image.width
        val h0 = image.height
        val plane = image.planes[0]
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val buf = plane.buffer
        buf.rewind()
        if (bytes.size < buf.remaining()) bytes = ByteArray(buf.remaining())
        buf.get(bytes, 0, buf.remaining())

        val w = if (rot % 180 == 0) w0 else h0
        val h = if (rot % 180 == 0) h0 else w0
        if (w != width || h != height) {
            width = w
            height = h
            pixels = IntArray(w * h)
            bitmap?.recycle()
            bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
        val out = pixels
        for (y in 0 until h0) {
            var off = y * rowStride
            for (x in 0 until w0) {
                val r = bytes[off].toInt() and 0xFF
                val g = bytes[off + 1].toInt() and 0xFF
                val b = bytes[off + 2].toInt() and 0xFF
                off += pixStride
                // clockwise rotation by `rot` degrees makes the frame upright
                val idx = when (rot) {
                    90 -> x * w + (h0 - 1 - y)
                    180 -> (h0 - 1 - y) * w + (w0 - 1 - x)
                    270 -> (w0 - 1 - x) * w + y
                    else -> y * w + x
                }
                out[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap!!.setPixels(out, 0, w, 0, 0, w, h)
    }
}
