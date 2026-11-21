package dev.pulsar.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Base64
import kotlin.math.abs

/** TraceExtractor (Kotlin) vs pulsar.preprocess.traces.TraceExtractor (Python) on the synthetic face fixture. */
class ExtractorParityTest {
    private val fx: JSONObject = Fixtures.gz(Fixtures.index().getString("extractor"))
    private val tol = fx.getJSONObject("tolerance")
    private val spec = PatchSpec.fromJson(fx.getJSONObject("patch_spec"))
    private val cfg = ExtractorConfig.fromJson(fx.getJSONObject("extractor"))

    @Test
    fun configMatchesContract() {
        assertEquals(32, spec.numPatches)
        assertEquals(listOf("forehead", "glabella", "nose", "cheek_r", "cheek_l", "chin"), spec.regionNames)
        assertEquals(ExtractorConfig(), cfg) // Kotlin defaults == Python constants
    }

    @Test
    fun srgbLutIsIdentical() {
        val want = fx.getJSONArray("srgb_lut").flatDoubles()
        for (i in 0 until 256) assertEquals("LUT[$i]", want[i], Srgb.LUT[i].toDouble(), 0.0)
    }

    @Test
    fun framesMatchPythonReference() {
        val w = fx.getInt("width")
        val h = fx.getInt("height")
        val imgs = fx.getJSONArray("images_rgb8_b64").let { a ->
            List(a.length()) { RgbImage.fromRgbBytes(w, h, Base64.getDecoder().decode(a.getString(it))) }
        }
        val lms = fx.getJSONObject("landmarks").let { o -> o.keySet().associateWith { o.getJSONArray(it).flatDoubles() } }
        val ex = TraceExtractor(spec, cfg)
        val frames = fx.getJSONArray("frames")
        val pN = spec.numPatches
        var worstRgb = 0.0
        var worstAngle = 0.0
        var nPixMismatch = 0
        var validPatches = 0
        for (f in 0 until frames.length()) {
            val fr = frames.getJSONObject(f)
            val key = if (fr.isNull("landmarks")) null else fr.getString("landmarks")
            val st = ex.process(imgs[fr.getInt("image")], key?.let { lms.getValue(it) })
            val e = fr.getJSONObject("expected")
            val at = "frame $f ($key)"
            assertEquals("$at face_found", e.getBoolean("face_found"), st.faceFound)
            assertEquals("$at motion", e.getDouble("motion"), st.motion, tol.getDouble("motion"))

            if (e.isNull("boxes")) {
                assertTrue("$at boxes", st.boxes == null)
            } else {
                val wantBoxes = e.getJSONArray("boxes").flatDoubles()
                for (i in wantBoxes.indices) assertEquals("$at box[$i]", wantBoxes[i].toInt(), st.boxes!![i])
                val wantAng = e.getJSONArray("view_angles").flatDoubles()
                for (p in 0 until pN) {
                    assertEquals("$at angle[$p]", wantAng[p], st.viewAnglesDeg!![p], tol.getDouble("view_angle_deg"))
                    worstAngle = maxOf(worstAngle, abs(wantAng[p] - st.viewAnglesDeg!![p]))
                }
            }

            val wantN = e.getJSONArray("n_pix").flatDoubles()
            val wantValid = e.getJSONArray("valid").flatBooleans()
            val nTol = tol.getInt("n_pix")
            for (p in 0 until pN) {
                assertTrue("$at n_pix[$p] ${st.nPix[p]} vs ${wantN[p].toInt()}", abs(st.nPix[p] - wantN[p].toInt()) <= nTol)
                if (abs(wantN[p] - spec.minPixels) > nTol) assertEquals("$at valid[$p]", wantValid[p], st.valid[p])
                if (st.nPix[p] != wantN[p].toInt()) nPixMismatch++
                if (st.valid[p]) validPatches++
            }
            val rgbTol = tol.getDouble("rgb")
            val dLin = maxAbsDiff(st.rgbLin, e.getJSONArray("rgb_lin").flatDoubles())
            val dSrgb = maxAbsDiff(st.rgbSrgb, e.getJSONArray("rgb_srgb").flatDoubles())
            assertTrue("$at rgb_lin max diff $dLin", dLin <= rgbTol)
            assertTrue("$at rgb_srgb max diff $dSrgb", dSrgb <= rgbTol)
            worstRgb = maxOf(worstRgb, dLin, dSrgb)

            if (e.isNull("skin")) {
                assertTrue("$at skin model should be empty", ex.skin.mean == null)
            } else {
                val sk = e.getJSONObject("skin")
                val mean = sk.getJSONArray("mean").flatDoubles()
                val cov = sk.getJSONArray("cov").flatDoubles()
                for (i in 0 until 2) assertEquals("$at skin mean[$i]", mean[i], ex.skin.mean!![i].toDouble(), tol.getDouble("skin_mean"))
                for (i in 0 until 4) assertEquals("$at skin cov[$i]", cov[i], ex.skin.cov!![i], tol.getDouble("skin_cov"))
                assertEquals("$at lum median", sk.getDouble("lum_median"), ex.skin.lumMedian, tol.getDouble("lum_median"))
            }
        }
        println(
            "extractor parity: ${frames.length()} frames, $validPatches valid patch-frames, worst rgb diff $worstRgb, " +
                "worst angle diff $worstAngle deg, n_pix mismatches $nPixMismatch",
        )
    }
}
