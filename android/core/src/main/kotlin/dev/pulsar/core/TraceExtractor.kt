package dev.pulsar.core

import kotlin.math.max
import kotlin.math.sqrt

/** Packed `0xAARRGGBB` pixels, row-major (the layout of `Bitmap.getPixels` / an RGBA camera frame). */
class RgbImage(val width: Int, val height: Int, val argb: IntArray) {
    init {
        require(argb.size >= width * height) { "pixel buffer smaller than $width x $height" }
    }

    companion object {
        /** Tightly packed 8-bit RGB bytes `[h * w * 3]` (test fixtures, desktop frames). */
        fun fromRgbBytes(width: Int, height: Int, rgb: ByteArray): RgbImage {
            require(rgb.size == width * height * 3)
            val px = IntArray(width * height) {
                val r = rgb[3 * it].toInt() and 0xFF
                val g = rgb[3 * it + 1].toInt() and 0xFF
                val b = rgb[3 * it + 2].toInt() and 0xFF
                (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            return RgbImage(width, height, px)
        }
    }
}

/** One frame's patch statistics (mirrors Python `FrameStats`). Arrays are `[P * 3]` / `[P]`. */
class FrameStats(
    /** Linear-RGB skin-pixel means; NaN where no skin pixel. */
    val rgbLin: FloatArray,
    /** sRGB-encoded means in [0, 1]; NaN where no skin pixel. */
    val rgbSrgb: FloatArray,
    val nPix: IntArray,
    val valid: BooleanArray,
    val faceFound: Boolean,
    /** Mean landmark displacement since the last detected frame, in face widths. */
    val motion: Double,
    /** Patch boxes `[P * 4]` (x0, y0, x1, y1) and view angles, or null when no landmarks were used. */
    val boxes: IntArray?,
    val viewAnglesDeg: DoubleArray?,
)

/**
 * Frame + landmarks -> per-patch linear-RGB means (port of `pulsar.preprocess.traces.TraceExtractor`).
 *
 * Pixels are linearised with the inverse-sRGB LUT before averaging. The skin model is seeded from this
 * face's near-frontal cheek patches each frame. Up to `carryFrames` face-less frames reuse the last
 * landmarks. Not thread-safe: use one instance per camera stream.
 */
class TraceExtractor(val spec: PatchSpec, val cfg: ExtractorConfig = ExtractorConfig()) {
    val skin = SkinModel(cfg)
    private val lut = Srgb.LUT
    private var lastLm: DoubleArray? = null
    private var missing = 0
    private val seedPatches: IntArray =
        (0 until spec.numPatches).filter { spec.regionOf(it) in cfg.seedRegions }.toIntArray()

    fun reset() {
        skin.reset()
        lastLm = null
        missing = 0
    }

    private fun empty(): FrameStats {
        val p = spec.numPatches
        return FrameStats(
            FloatArray(p * 3) { Float.NaN },
            FloatArray(p * 3) { Float.NaN },
            IntArray(p),
            BooleanArray(p),
            faceFound = false,
            motion = 0.0,
            boxes = null,
            viewAnglesDeg = null,
        )
    }

    /** [lmNorm]: normalised landmarks `[N * 3]` (N >= 468) from the face landmarker, or null if no face. */
    fun process(img: RgbImage, lmNorm: DoubleArray?): FrameStats {
        val found = lmNorm != null
        val lm: DoubleArray
        if (lmNorm == null) {
            missing++
            val last = lastLm
            if (last == null || missing > cfg.carryFrames) return empty()
            lm = last
        } else {
            missing = 0
            lm = lmNorm
        }
        val w = img.width
        val h = img.height
        val px = Geometry.toPixels(lm, w, h)
        val fw = max(Geometry.faceWidthPx(px, spec), 1.0)
        var motion = 0.0
        val prevLm = lastLm
        if (prevLm != null && found) {
            val prev = Geometry.toPixels(prevLm, w, h)
            val n = px.size / 3
            var sum = 0.0
            for (i in 0 until n) {
                val dx = px[3 * i] - prev[3 * i]
                val dy = px[3 * i + 1] - prev[3 * i + 1]
                sum += sqrt(dx * dx + dy * dy)
            }
            motion = sum / n / fw
        }
        if (found) lastLm = lm.copyOf()

        val pN = spec.numPatches
        val boxes = Geometry.patchBoxes(px, spec, w, h)
        val angles = Geometry.viewAnglesDeg(px, spec)

        // crops in row-major order (== numpy frame[y0:y1, x0:x1].reshape(-1, 3))
        val lin = arrayOfNulls<FloatArray>(pN)
        val u8 = arrayOfNulls<IntArray>(pN)
        for (p in 0 until pN) {
            val x0 = boxes[4 * p]
            val y0 = boxes[4 * p + 1]
            val x1 = boxes[4 * p + 2]
            val y1 = boxes[4 * p + 3]
            val n = max(x1 - x0, 0) * max(y1 - y0, 0)
            val l = FloatArray(n * 3)
            val u = IntArray(n)
            var k = 0
            for (y in y0 until y1) {
                val row = y * w
                for (x in x0 until x1) {
                    val c = img.argb[row + x]
                    u[k] = c
                    l[3 * k] = lut[(c shr 16) and 0xFF]
                    l[3 * k + 1] = lut[(c shr 8) and 0xFF]
                    l[3 * k + 2] = lut[c and 0xFF]
                    k++
                }
            }
            lin[p] = l
            u8[p] = u
        }

        // skin-model seed: near-frontal cheek pixels, evenly subsampled (deterministic, as in Python)
        var seedN = 0
        for (p in seedPatches) if (lin[p]!!.isNotEmpty() && angles[p] < cfg.seedMaxAngleDeg) seedN += lin[p]!!.size / 3
        if (seedN > 0) {
            val all = FloatArray(seedN * 3)
            var off = 0
            for (p in seedPatches) {
                val l = lin[p]!!
                if (l.isNotEmpty() && angles[p] < cfg.seedMaxAngleDeg) {
                    l.copyInto(all, off)
                    off += l.size
                }
            }
            if (seedN > cfg.seedMaxPixels) {
                val m = cfg.seedMaxPixels
                val sub = FloatArray(m * 3)
                for (i in 0 until m) {
                    val j = ((i.toLong() * seedN) / m).toInt()
                    sub[3 * i] = all[3 * j]
                    sub[3 * i + 1] = all[3 * j + 1]
                    sub[3 * i + 2] = all[3 * j + 2]
                }
                skin.update(sub, m)
            } else {
                skin.update(all, seedN)
            }
        }

        val rgbLin = FloatArray(pN * 3) { Float.NaN }
        val rgbSrgb = FloatArray(pN * 3) { Float.NaN }
        val nPix = IntArray(pN)
        val valid = BooleanArray(pN)
        for (p in 0 until pN) {
            val l = lin[p]!!
            val n = l.size / 3
            if (n == 0) continue
            val keepSkin = BooleanArray(n)
            val keepSpec = BooleanArray(n)
            skin.mask(l, n, keepSkin)
            SkinModel.specularKeep(l, n, cfg.specularPct, cfg.specularSatMax, keepSpec)
            var cnt = 0
            var r = 0f // float32 running sums, like numpy's axis-0 mean
            var g = 0f
            var b = 0f
            var ur = 0.0
            var ug = 0.0
            var ub = 0.0
            val u = u8[p]!!
            for (i in 0 until n) {
                if (!(keepSkin[i] && keepSpec[i])) continue
                cnt++
                r += l[3 * i]
                g += l[3 * i + 1]
                b += l[3 * i + 2]
                ur += ((u[i] shr 16) and 0xFF).toDouble()
                ug += ((u[i] shr 8) and 0xFF).toDouble()
                ub += (u[i] and 0xFF).toDouble()
            }
            nPix[p] = cnt
            if (cnt == 0) continue
            rgbLin[3 * p] = SkinModel.meanF32(r, cnt)
            rgbLin[3 * p + 1] = SkinModel.meanF32(g, cnt)
            rgbLin[3 * p + 2] = SkinModel.meanF32(b, cnt)
            rgbSrgb[3 * p] = (ur / cnt / 255.0).toFloat()
            rgbSrgb[3 * p + 1] = (ug / cnt / 255.0).toFloat()
            rgbSrgb[3 * p + 2] = (ub / cnt / 255.0).toFloat()
            valid[p] = cnt >= spec.minPixels && angles[p] <= spec.maxViewAngleDeg
        }
        return FrameStats(rgbLin, rgbSrgb, nPix, valid, found, motion, boxes, angles)
    }
}
