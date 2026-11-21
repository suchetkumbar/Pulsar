package dev.pulsar.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** LiveWindow / Resampler (Kotlin) vs pulsar.preprocess.live.live_window (Python) on synthetic buffers. */
class WindowParityTest {
    private class Raw(val t: DoubleArray, val rgb: DoubleArray, val valid: BooleanArray, val nPix: DoubleArray, val motion: DoubleArray)

    private fun raw(case: JSONObject): Raw {
        val r = case.getJSONObject("raw")
        return Raw(
            r.getJSONArray("t").flatDoubles(),
            r.getJSONArray("rgb").flatDoubles(),
            r.getJSONArray("valid").flatBooleans(),
            r.getJSONArray("n_pix").flatDoubles(),
            r.getJSONArray("motion").flatDoubles(),
        )
    }

    private val cases: List<JSONObject> =
        Fixtures.index().getJSONArray("windows").let { a -> List(a.length()) { Fixtures.gz(a.getString(it)) } }

    @Test
    fun windowsMatchPythonReference() {
        assertTrue(cases.size >= 3)
        for (case in cases) checkWindow(case, computeFromArrays(case), case.getString("name"))
    }

    @Test
    fun ringBufferGivesTheSameWindow() {
        // frames pushed one by one through LiveBuffer (with eviction) == the window over the whole raw buffer
        for (case in cases) {
            val r = raw(case)
            val cfg = WindowConfig.fromJson(case.getJSONObject("window"), case.getDouble("fs_hz"), 300)
            val pN = r.valid.size / r.t.size
            val buf = LiveBuffer(pN, cfg)
            for (i in r.t.indices) {
                val st = FrameStats(
                    FloatArray(pN * 3) { r.rgb[i * pN * 3 + it].toFloat() },
                    FloatArray(pN * 3),
                    IntArray(pN) { r.nPix[i * pN + it].toInt() },
                    BooleanArray(pN) { r.valid[i * pN + it] },
                    faceFound = true,
                    motion = r.motion[i],
                    boxes = null,
                    viewAnglesDeg = null,
                )
                buf.add(r.t[i], st)
            }
            assertEquals("${case.getString("name")}: nothing evicted from a 12 s buffer", r.t.size, buf.size)
            val w = buf.window()
            assertNotNull(w)
            // the buffer stores float32 traces (like the extractor), so compare at float32 resolution
            val want = computeFromArrays(case)!!
            var d = 0.0
            for (i in w!!.x.indices) d = maxOf(d, kotlin.math.abs(w.x[i] - want.x[i]).toDouble())
            assertTrue("buffer vs array window max diff $d", d < 1e-6)
        }
    }

    @Test
    fun tooShortBufferGivesNull() {
        val cfg = WindowConfig()
        val t = DoubleArray(150) { it / 30.0 }
        val w = LiveWindow.compute(t, DoubleArray(150 * 3) { 0.3 }, BooleanArray(150) { true }, DoubleArray(150) { 100.0 }, DoubleArray(150), 1, cfg)
        assertNull(w)
    }

    @Test
    fun unsortedAndDuplicateTimestamps() {
        // (t, value) with a duplicate and an out-of-order frame -> same as the cleaned, sorted series
        val t = doubleArrayOf(0.0, 0.1, 0.1, 0.3, 0.2, 0.4)
        val v = doubleArrayOf(0.0, 1.0, 9.0, 3.0, 2.0, 4.0)
        val x = DoubleArray(6 * 3) { v[it / 3] }
        val r = Resampler.resample(t, x, BooleanArray(6) { true }, 1, fs = 10.0)
        assertEquals(5, r.tGrid.size)
        for (g in 0 until 5) assertEquals(g.toDouble(), r.x[g * 3], 1e-12) // the first of the duplicates is kept
    }

    @Test
    fun longGapInvalidatesGridSamples() {
        val t = doubleArrayOf(0.0, 0.1, 0.2, 0.6, 0.7)
        val x = DoubleArray(5 * 3) { 1.0 }
        val r = Resampler.resample(t, x, BooleanArray(5) { true }, 1, fs = 10.0, maxGapS = 0.25)
        val valid = r.valid.toList()
        assertEquals(listOf(true, true, true, false, false, false, true, true), valid)
    }

    private fun computeFromArrays(case: JSONObject): ModelWindow? {
        val r = raw(case)
        val pN = r.valid.size / r.t.size
        val cfg = WindowConfig.fromJson(case.getJSONObject("window"), case.getDouble("fs_hz"), 300)
        return LiveWindow.compute(r.t, r.rgb, r.valid, r.nPix, r.motion, pN, cfg)
    }

    companion object {
        fun checkWindow(case: JSONObject, w: ModelWindow?, name: String) {
            assertNotNull("$name window", w)
            val tol = GoldenCheck.tolerance(case)
            for ((k, e) in GoldenCheck.windowErrors(case, w!!)) {
                assertTrue("$name $k max error $e > ${tol[k]}", e <= (tol[k] ?: 0.0))
            }
        }
    }
}
