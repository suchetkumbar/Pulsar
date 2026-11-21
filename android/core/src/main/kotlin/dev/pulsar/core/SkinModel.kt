package dev.pulsar.core

import kotlin.math.floor

/**
 * Per-face adaptive skin model (mirrors `pulsar.preprocess.skin_mask`): a Gaussian on rg chromaticity,
 * fitted to this face's own cheek pixels, plus a luminance floor *relative* to this face's median.
 * No fixed colour thresholds, so darker skin is not silently discarded.
 *
 * Pixel arrays are flat linear-RGB float32 `[n * 3]`. Per-pixel arithmetic is float32, as in numpy, so
 * threshold decisions match the Python reference bit for bit; statistics accumulate in double.
 */
class SkinModel(private val cfg: ExtractorConfig = ExtractorConfig()) {
    /** Chroma mean (float32, like numpy), covariance `[c00, c01, c10, c11]`, median luminance. */
    var mean: FloatArray? = null
        private set
    var cov: DoubleArray? = null
        private set
    var lumMedian: Double = Double.NaN
        private set

    fun reset() {
        mean = null
        cov = null
        lumMedian = Double.NaN
    }

    /** Update from seed pixels `[n * 3]` (linear RGB). */
    fun update(seed: FloatArray, n: Int = seed.size / 3) {
        if (n < cfg.skinMinSeedPixels) return
        val c0 = FloatArray(n)
        val c1 = FloatArray(n)
        val lum = FloatArray(n)
        var f0 = 0f // numpy's float32 axis-0 mean accumulates sequentially in float32
        var f1 = 0f
        var s0 = 0.0 // np.cov converts to float64 first
        var s1 = 0.0
        for (i in 0 until n) {
            val r = seed[3 * i]
            val g = seed[3 * i + 1]
            val b = seed[3 * i + 2]
            val sum = r + g + b
            val s = sum + 1e-9f
            c0[i] = r / s
            c1[i] = g / s
            lum[i] = sum / 3f
            f0 += c0[i]
            f1 += c1[i]
            s0 += c0[i]
            s1 += c1[i]
        }
        val m0 = s0 / n
        val m1 = s1 / n
        var v00 = 0.0
        var v01 = 0.0
        var v11 = 0.0
        for (i in 0 until n) {
            val d0 = c0[i] - m0
            val d1 = c1[i] - m1
            v00 += d0 * d0
            v01 += d0 * d1
            v11 += d1 * d1
        }
        val dof = (n - 1).toDouble()
        val newCov = doubleArrayOf(v00 / dof + 1e-6, v01 / dof, v01 / dof, v11 / dof + 1e-6)
        val newMean = floatArrayOf(meanF32(f0, n), meanF32(f1, n))
        val newLum = medianF32(lum).toDouble()
        val oldMean = mean
        val oldCov = cov
        if (oldMean == null || oldCov == null) {
            mean = newMean
            cov = newCov
            lumMedian = newLum
        } else {
            val m = cfg.skinMomentum
            val mf = m.toFloat()
            val omf = (1.0 - m).toFloat()
            mean = floatArrayOf(mf * oldMean[0] + omf * newMean[0], mf * oldMean[1] + omf * newMean[1])
            cov = DoubleArray(4) { m * oldCov[it] + (1.0 - m) * newCov[it] }
            lumMedian = m * lumMedian + (1.0 - m) * newLum
        }
    }

    /** Skin mask for pixels `[n * 3]` into [out]; all true until the model is initialised. */
    fun mask(px: FloatArray, n: Int, out: BooleanArray) {
        val mu = mean
        val cv = cov
        if (mu == null || cv == null) {
            out.fill(true, 0, n)
            return
        }
        val det = cv[0] * cv[3] - cv[1] * cv[2]
        val i00 = cv[3] / det
        val i01 = -cv[1] / det
        val i10 = -cv[2] / det
        val i11 = cv[0] / det
        val lumThr = (cfg.shadowFrac * lumMedian).toFloat()
        val radius2 = cfg.skinRadius2
        for (i in 0 until n) {
            val r = px[3 * i]
            val g = px[3 * i + 1]
            val b = px[3 * i + 2]
            val sum = r + g + b
            val s = sum + 1e-9f
            val d0 = (r / s - mu[0]).toDouble()
            val d1 = (g / s - mu[1]).toDouble()
            val m2 = d0 * (i00 * d0 + i01 * d1) + d1 * (i10 * d0 + i11 * d1)
            out[i] = m2 < radius2 && sum / 3f > lumThr
        }
    }

    companion object {
        /** numpy float32 mean from a float32 running sum: `true_divide(sum, intp(n))` computes in float64. */
        fun meanF32(sum: Float, n: Int): Float = (sum.toDouble() / n).toFloat()

        /** numpy `np.median` of a float32 array (float32 result). */
        fun medianF32(x: FloatArray): Float {
            val s = x.copyOf()
            s.sort()
            val n = s.size
            return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2f
        }

        /** numpy `np.percentile(x, pct)` (method "linear") of a float32 array, float32 arithmetic. */
        fun percentileF32(sorted: FloatArray, n: Int, pct: Double): Float {
            val vi = (n - 1) * (pct / 100.0)
            val lo = floor(vi).toInt()
            val hi = minOf(lo + 1, n - 1)
            val g = vi - lo
            val a = sorted[lo]
            val b = sorted[hi]
            val d = b - a
            return if (g >= 0.5) b - d * (1.0 - g).toFloat() else a + d * g.toFloat()
        }

        /** True for pixels that are NOT specular highlights (bright *and* desaturated). */
        fun specularKeep(px: FloatArray, n: Int, pct: Double, satMax: Double, out: BooleanArray) {
            if (n == 0) return
            val lum = FloatArray(n)
            for (i in 0 until n) lum[i] = (px[3 * i] + px[3 * i + 1] + px[3 * i + 2]) / 3f
            val sorted = lum.copyOf()
            sorted.sort()
            val thr = percentileF32(sorted, n, pct)
            val sm = satMax.toFloat()
            for (i in 0 until n) {
                val r = px[3 * i]
                val g = px[3 * i + 1]
                val b = px[3 * i + 2]
                val mx = maxOf(r, g, b)
                val mn = minOf(r, g, b)
                val sat = (mx - mn) / (mx + 1e-9f)
                out[i] = !(lum[i] >= thr && sat < sm)
            }
        }
    }
}
