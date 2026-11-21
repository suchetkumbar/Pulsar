package dev.pulsar.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

class CoreUnitTest {
    @Test
    fun jacobiFindsTheSmallestEigenvector() {
        val rnd = Random(0)
        repeat(200) {
            val q = DoubleArray(9) { rnd.nextDouble(-1.0, 1.0) }
            val a = DoubleArray(9) // A = Q^T Q (symmetric PSD)
            for (i in 0 until 3) for (j in 0 until 3) for (k in 0 until 3) a[3 * i + j] += q[3 * k + i] * q[3 * k + j]
            val v = Geometry.smallestEigenvector(a)
            assertEquals(1.0, sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-9)
            val av = DoubleArray(3) { i -> (0 until 3).sumOf { a[3 * i + it] * v[it] } }
            val lambda = av.indices.sumOf { av[it] * v[it] }
            for (i in 0 until 3) assertEquals(lambda * v[i], av[i], 1e-9)
            // Rayleigh quotient of the smallest eigenvector <= any random direction's
            val u = DoubleArray(3) { rnd.nextDouble(-1.0, 1.0) }
            val un = sqrt(u.sumOf { it * it })
            val au = DoubleArray(3) { i -> (0 until 3).sumOf { a[3 * i + it] * u[it] } }
            assertTrue(lambda <= au.indices.sumOf { au[it] * u[it] } / (un * un) + 1e-12)
        }
    }

    @Test
    fun percentileAndMedianFollowNumpy() {
        // np.percentile(np.float32([0, .25, .5, 1]), 98) and np.median of odd/even lengths
        val s = floatArrayOf(0f, 0.25f, 0.5f, 1f)
        assertEquals(0.97f, SkinModel.percentileF32(s, 4, 98.0), 1e-6f)
        assertEquals(0.375f, SkinModel.medianF32(floatArrayOf(1f, 0.5f, 0.25f, 0f)))
        assertEquals(0.5f, SkinModel.medianF32(floatArrayOf(1f, 0.5f, 0f)))
    }

    @Test
    fun uniformSkinPatchMeansEqualTheSkinColour() {
        val spec = Fixtures.gz(Fixtures.index().getString("extractor")).getJSONObject("patch_spec").let { PatchSpec.fromJson(it) }
        // a flat image, with landmarks laid out so every patch sits inside it
        val w = 320
        val h = 240
        val px = IntArray(w * h) { (0xFF shl 24) or (200 shl 16) or (150 shl 8) or 120 }
        val lm = DoubleArray(478 * 3)
        for (i in 0 until 478) {
            lm[3 * i] = 0.3 + 0.4 * ((i * 37) % 101) / 100.0
            lm[3 * i + 1] = 0.3 + 0.4 * ((i * 53) % 97) / 96.0
            lm[3 * i + 2] = -0.01 * ((i * 11) % 7)
        }
        lm[3 * 234] = 0.25
        lm[3 * 454] = 0.75 // face width = 160 px
        val st = TraceExtractor(spec).process(RgbImage(w, h, px), lm)
        val want = floatArrayOf(Srgb.LUT[200], Srgb.LUT[150], Srgb.LUT[120])
        for (p in 0 until spec.numPatches) {
            assertTrue(st.nPix[p] > 0)
            for (c in 0 until 3) assertEquals(want[c], st.rgbLin[3 * p + c], 1e-6f)
        }
        assertTrue(st.faceFound)
    }

    @Test
    fun abstainPolicy() {
        val out = TraceNetOutput(FloatArray(300), 72f, 72f, 0.8f, FloatArray(32))
        val policy = AbstainPolicy(0.5f)
        assertEquals(VitalsStatus.OK, policy.decide(1.0, true, out).status)
        assertEquals(72f, policy.decide(1.0, true, out).hrBpm)
        assertEquals(VitalsStatus.NO_FACE, policy.decide(1.0, false, out).status)
        assertNull(policy.decide(1.0, false, out).hrBpm)
        assertEquals(VitalsStatus.WARMING_UP, policy.decide(1.0, true, null).status)
        val weak = TraceNetOutput(FloatArray(300), 72f, 72f, 0.3f, FloatArray(32))
        assertEquals(VitalsStatus.LOW_SIGNAL, policy.decide(1.0, true, weak).status)
    }

    @Test
    fun pipelineWarmsUpThenEstimates() {
        val meta = ModelMeta.parse(META_JSON)
        var calls = 0
        val fake = HrModel { w ->
            calls++
            assertEquals(300, w.tLen)
            TraceNetOutput(FloatArray(w.tLen), 70f, 70f, 0.9f, FloatArray(w.numPatches))
        }
        val pipe = PulsarPipeline(meta, fake)
        val img = RgbImage(64, 64, IntArray(64 * 64) { (0xFF shl 24) or (180 shl 16) or (130 shl 8) or 100 })
        val lm = DoubleArray(478 * 3) { if (it % 3 == 2) 0.0 else 0.3 + 0.4 * ((it * 7919) % 1000) / 1000.0 }
        assertEquals(VitalsStatus.WARMING_UP, pipe.estimate().status)
        for (i in 0 until 400) pipe.onFrame(i / 30.0, img, lm)
        val v = pipe.estimate()
        assertEquals(VitalsStatus.OK, v.status)
        assertEquals(70f, v.hrBpm)
        assertEquals(1, calls)
        assertTrue("buffer keeps ~12 s", abs(pipe.buffer.spanS - 12.0) < 0.05)
        pipe.onFrame(400 / 30.0, img, null)
        assertEquals(VitalsStatus.NO_FACE, pipe.estimate().status)
    }

    companion object {
        private val REGIONS = """{"forehead":[151,108,337,109,338,69,299,104,333],"glabella":[9,8],"nose":[6,197,195,5],
            "cheek_r":[50,101,205,117,123,187,36],"cheek_l":[280,330,425,346,352,411,266],"chin":[200,199,175]}"""
        private val IDS = "151,108,337,109,338,69,299,104,333,9,8,6,197,195,5,50,101,205,117,123,187,36," +
            "280,330,425,346,352,411,266,200,199,175"
        private val RIDS = "0,0,0,0,0,0,0,0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5"

        /** A metadata file as written before the extractor/window/region_names blocks existed. */
        val META_JSON = """{"format":"pulsar-tracenet","outputs":["bvp","hr","hr_mode","conf","alpha","hr_logits"],
            "fs_hz":30.0,"t_len":300,"num_patches":32,"bpm_grid":[40.0,200.0,1.0],"temperature":1.0,
            "patch_spec":{"regions":$REGIONS,"size_frac":0.09,"min_pixels":16,"max_view_angle_deg":70.0,"version":"v1",
            "landmark_ids":[$IDS],"region_ids":[$RIDS],"face_width_landmarks":[234,454]}}"""
    }
}
