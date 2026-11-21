package dev.pulsar.core

import kotlin.math.abs
import kotlin.math.floor

/** Patch traces on a uniform grid: x `[M][P][3]`, valid `[M][P]` (flat, row-major). */
class Resampled(val tGrid: DoubleArray, val x: DoubleArray, val valid: BooleanArray, val numPatches: Int)

/**
 * Timestamp-based resampling (port of `pulsar.preprocess.resample.resample` for x `[N, P, 3]` with a
 * per-patch validity mask `[N, P]`). Phones deliver jittery, sometimes dropped frames: never assume a
 * constant frame rate. Linear interpolation follows `np.interp` exactly; grid samples outside the
 * patch's valid range, or inside a gap longer than `maxGapS` between valid frames, are invalid.
 */
object Resampler {
    fun uniformGrid(t0: Double, t1: Double, fs: Double): DoubleArray {
        val n = floor((t1 - t0) * fs + 1e-9).toInt() + 1
        return DoubleArray(n) { t0 + it / fs }
    }

    /** Count of `xs[0 until n]` that are <= v (`np.searchsorted(..., side="right")`). */
    private fun upperBound(xs: DoubleArray, n: Int, v: Double): Int {
        var lo = 0
        var hi = n
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (xs[mid] <= v) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** `np.interp(x, xp[0 until n], fp[0 until n])` for one query point. */
    private fun interp(x: Double, xp: DoubleArray, fp: DoubleArray, n: Int): Double {
        if (x > xp[n - 1]) return fp[n - 1]
        if (x < xp[0]) return fp[0]
        val j = upperBound(xp, n, x) - 1
        if (j >= n - 1) return fp[n - 1]
        if (xp[j] == x) return fp[j]
        val slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        return slope * (x - xp[j]) + fp[j]
    }

    fun resample(
        tIn: DoubleArray,
        xIn: DoubleArray,
        validIn: BooleanArray,
        numPatches: Int,
        fs: Double,
        maxGapS: Double = 0.25,
        tGrid: DoubleArray? = null,
    ): Resampled {
        val pN = numPatches
        var t = tIn
        var x = xIn
        var valid = validIn
        require(t.isNotEmpty() && x.size == t.size * pN * 3 && valid.size == t.size * pN) { "shape mismatch" }
        if ((1 until t.size).any { t[it] <= t[it - 1] }) { // stable sort, drop repeated timestamps
            val order = t.indices.sortedBy { t[it] }
            val keep = order.filterIndexed { i, o -> i == 0 || t[o] - t[order[i - 1]] > 0 }
            t = DoubleArray(keep.size) { tIn[keep[it]] }
            x = DoubleArray(keep.size * pN * 3) { xIn[keep[it / (pN * 3)] * pN * 3 + it % (pN * 3)] }
            valid = BooleanArray(keep.size * pN) { validIn[keep[it / pN] * pN + it % pN] }
        }
        val nT = t.size
        val tg = tGrid ?: uniformGrid(t[0], t[nT - 1], fs)
        val m = tg.size
        val out = DoubleArray(m * pN * 3)
        val outValid = BooleanArray(m * pN)
        val tj = DoubleArray(nT)
        val xj = Array(3) { DoubleArray(nT) }
        for (p in 0 until pN) {
            var l = 0
            for (i in 0 until nT) {
                if (!valid[i * pN + p]) continue
                tj[l] = t[i]
                for (c in 0 until 3) xj[c][l] = x[(i * pN + p) * 3 + c]
                l++
            }
            if (l < 2) {
                for (g in 0 until m) for (c in 0 until 3) out[(g * pN + p) * 3 + c] = Double.NaN
                continue
            }
            for (g in 0 until m) {
                val tq = tg[g]
                for (c in 0 until 3) out[(g * pN + p) * 3 + c] = interp(tq, tj, xj[c], l)
                val idx = upperBound(tj, l, tq)
                val left = (idx - 1).coerceIn(0, l - 1)
                val right = idx.coerceIn(0, l - 1)
                val gap = tj[right] - tj[left]
                val onSample = abs(tq - tj[left]) <= 1e-9
                val bad = tq < tj[0] - 1e-9 || tq > tj[l - 1] + 1e-9 || (gap > maxGapS && !onSample)
                outValid[g * pN + p] = !bad
            }
        }
        return Resampled(tg, out, outValid, pN)
    }
}

/**
 * One model input window, already in the exported graph's layout:
 * x `[1, P, T, 3]`, valid `[1, P, T]` (1 = valid), nPix `[1, P]`, motion `[1]`.
 */
class ModelWindow(
    val tGrid: DoubleArray,
    val x: FloatArray,
    val valid: FloatArray,
    val nPix: FloatArray,
    val motion: Float,
    val numPatches: Int,
    val tLen: Int,
) {
    fun x(t: Int, p: Int, c: Int): Float = x[(p * tLen + t) * 3 + c]

    fun valid(t: Int, p: Int): Boolean = valid[p * tLen + t] > 0.5f

    /** Fraction of (patch, sample) cells that are valid. */
    val validFraction: Double get() = valid.count { it > 0.5f }.toDouble() / valid.size
}

/** Port of `pulsar.preprocess.live.live_window` (the single source of truth for live inference inputs). */
object LiveWindow {
    /**
     * Raw buffer t `[N]`, rgb `[N][P][3]`, valid `[N][P]`, nPix `[N][P]`, motion `[N]` -> the last
     * `cfg.tLen` grid samples, or null if the buffer is still too short.
     */
    fun compute(
        t: DoubleArray,
        rgb: DoubleArray,
        valid: BooleanArray,
        nPix: DoubleArray,
        motion: DoubleArray,
        numPatches: Int,
        cfg: WindowConfig,
    ): ModelWindow? {
        if (t.isEmpty()) return null
        val r = Resampler.resample(t, rgb, valid, numPatches, cfg.fsHz, cfg.maxGapS)
        val tl = cfg.tLen
        val m = r.tGrid.size
        if (m < tl) return null
        val pN = numPatches
        val g0 = m - tl
        val x = FloatArray(pN * tl * 3)
        val v = FloatArray(pN * tl)
        for (g in 0 until tl) {
            for (p in 0 until pN) {
                for (c in 0 until 3) {
                    val value = r.x[((g0 + g) * pN + p) * 3 + c]
                    x[(p * tl + g) * 3 + c] = if (value.isNaN()) 0f else value.toFloat()
                }
                v[p * tl + g] = if (r.valid[(g0 + g) * pN + p]) 1f else 0f
            }
        }
        // n_pix and motion: NaN-aware means over the last tLen *raw* frames
        val n = t.size
        val f0 = maxOf(0, n - tl)
        val npx = FloatArray(pN)
        for (p in 0 until pN) {
            var s = 0.0
            var cnt = 0
            for (i in f0 until n) {
                val value = nPix[i * pN + p]
                if (!value.isNaN()) {
                    s += value
                    cnt++
                }
            }
            npx[p] = if (cnt == 0) 0f else (s / cnt).toFloat()
        }
        var ms = 0.0
        var mc = 0
        for (i in f0 until n) {
            if (!motion[i].isNaN()) {
                ms += motion[i]
                mc++
            }
        }
        val mot = if (mc == 0) 0f else (ms / mc).toFloat()
        return ModelWindow(r.tGrid.copyOfRange(g0, m), x, v, npx, mot, pN, tl)
    }
}

/**
 * Ring buffer of per-frame patch stats (what the desktop demo keeps in a deque). Frames older than
 * `bufferS` before the newest are evicted; [window] assembles the model input via [LiveWindow.compute].
 */
class LiveBuffer(val numPatches: Int, val cfg: WindowConfig) {
    private class Rec(val t: Double, val rgb: FloatArray, val valid: BooleanArray, val nPix: IntArray, val motion: Double)

    private val frames = ArrayDeque<Rec>()

    val size: Int get() = frames.size

    /** Seconds covered by the buffer. */
    val spanS: Double get() = if (frames.size < 2) 0.0 else frames.last().t - frames.first().t

    fun clear() = frames.clear()

    fun add(t: Double, st: FrameStats) {
        require(st.valid.size == numPatches) { "frame has ${st.valid.size} patches, buffer $numPatches" }
        frames.addLast(Rec(t, st.rgbLin.copyOf(), st.valid.copyOf(), st.nPix.copyOf(), st.motion))
        while (frames.isNotEmpty() && frames.first().t < t - cfg.bufferS) frames.removeFirst()
    }

    fun window(): ModelWindow? {
        val n = frames.size
        if (n == 0) return null
        val pN = numPatches
        val t = DoubleArray(n)
        val rgb = DoubleArray(n * pN * 3)
        val valid = BooleanArray(n * pN)
        val nPix = DoubleArray(n * pN)
        val motion = DoubleArray(n)
        frames.forEachIndexed { i, f ->
            t[i] = f.t
            for (k in 0 until pN * 3) rgb[i * pN * 3 + k] = f.rgb[k].toDouble()
            for (p in 0 until pN) {
                valid[i * pN + p] = f.valid[p]
                nPix[i * pN + p] = f.nPix[p].toDouble()
            }
            motion[i] = f.motion
        }
        return LiveWindow.compute(t, rgb, valid, nPix, motion, pN, cfg)
    }
}
