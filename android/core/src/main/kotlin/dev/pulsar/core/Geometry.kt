package dev.pulsar.core

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Patch geometry on MediaPipe face landmarks (mirrors `pulsar.preprocess.patches`).
 *
 * Landmarks are flat `[N * 3]` arrays: normalised (x, y in [0, 1], z ~ x-scale, negative toward the camera)
 * or pixel-scale after [toPixels] (z scaled by the image width).
 */
object Geometry {
    fun toPixels(lmNorm: DoubleArray, width: Int, height: Int): DoubleArray {
        val out = DoubleArray(lmNorm.size)
        for (i in 0 until lmNorm.size / 3) {
            out[3 * i] = lmNorm[3 * i] * width
            out[3 * i + 1] = lmNorm[3 * i + 1] * height
            out[3 * i + 2] = lmNorm[3 * i + 2] * width
        }
        return out
    }

    fun faceWidthPx(lmPx: DoubleArray, spec: PatchSpec): Double {
        val a = spec.faceWidthLandmarks[0]
        val b = spec.faceWidthLandmarks[1]
        val dx = lmPx[3 * a] - lmPx[3 * b]
        val dy = lmPx[3 * a + 1] - lmPx[3 * b + 1]
        return sqrt(dx * dx + dy * dy)
    }

    /** Integer boxes `[P * 4]` = (x0, y0, x1, y1), clipped to the image, x1 / y1 exclusive. */
    fun patchBoxes(lmPx: DoubleArray, spec: PatchSpec, width: Int, height: Int): IntArray {
        val half = max(0.5 * spec.sizeFrac * faceWidthPx(lmPx, spec), 1.0)
        val out = IntArray(spec.numPatches * 4)
        for (p in 0 until spec.numPatches) {
            val id = spec.landmarkIds[p]
            val cx = lmPx[3 * id]
            val cy = lmPx[3 * id + 1]
            out[4 * p] = floor(cx - half).coerceIn(0.0, width.toDouble()).toInt()
            out[4 * p + 1] = floor(cy - half).coerceIn(0.0, height.toDouble()).toInt()
            out[4 * p + 2] = ceil(cx + half).coerceIn(0.0, width.toDouble()).toInt()
            out[4 * p + 3] = ceil(cy + half).coerceIn(0.0, height.toDouble()).toInt()
        }
        return out
    }

    /**
     * Angle (deg) between each patch's local surface normal and the camera axis. The normal is the
     * smallest principal axis of the `k + 1` nearest landmarks (the patch landmark included), oriented
     * toward the camera.
     */
    fun viewAnglesDeg(lmPx: DoubleArray, spec: PatchSpec): DoubleArray {
        val n = lmPx.size / 3
        val k1 = minOf(spec.viewAngleK + 1, n)
        val nnIdx = IntArray(k1)
        val nnD = DoubleArray(k1)
        val out = DoubleArray(spec.numPatches)
        val s = DoubleArray(9)
        for (p in 0 until spec.numPatches) {
            val c = spec.landmarkIds[p]
            val cx = lmPx[3 * c]
            val cy = lmPx[3 * c + 1]
            val cz = lmPx[3 * c + 2]
            var filled = 0
            for (j in 0 until n) {
                val dx = cx - lmPx[3 * j]
                val dy = cy - lmPx[3 * j + 1]
                val dz = cz - lmPx[3 * j + 2]
                val d = dx * dx + dy * dy + dz * dz
                if (filled == k1 && d >= nnD[k1 - 1]) continue
                var i = if (filled < k1) filled++ else k1 - 1 // insertion into the sorted k-nearest list
                while (i > 0 && nnD[i - 1] > d) {
                    nnD[i] = nnD[i - 1]
                    nnIdx[i] = nnIdx[i - 1]
                    i--
                }
                nnD[i] = d
                nnIdx[i] = j
            }
            var mx = 0.0
            var my = 0.0
            var mz = 0.0
            for (i in 0 until k1) {
                mx += lmPx[3 * nnIdx[i]]
                my += lmPx[3 * nnIdx[i] + 1]
                mz += lmPx[3 * nnIdx[i] + 2]
            }
            mx /= k1
            my /= k1
            mz /= k1
            s.fill(0.0)
            for (i in 0 until k1) {
                val qx = lmPx[3 * nnIdx[i]] - mx
                val qy = lmPx[3 * nnIdx[i] + 1] - my
                val qz = lmPx[3 * nnIdx[i] + 2] - mz
                s[0] += qx * qx
                s[1] += qx * qy
                s[2] += qx * qz
                s[4] += qy * qy
                s[5] += qy * qz
                s[8] += qz * qz
            }
            s[3] = s[1]
            s[6] = s[2]
            s[7] = s[5]
            val nrm = smallestEigenvector(s)
            val len = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
            val nz = -abs(nrm[2]) // oriented toward the camera (negative z)
            out[p] = Math.toDegrees(acos((-nz / (len + 1e-12)).coerceIn(-1.0, 1.0)))
        }
        return out
    }

    /** Unit eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (row-major), cyclic Jacobi. */
    fun smallestEigenvector(m: DoubleArray): DoubleArray {
        require(m.size == 9)
        val a = m.copyOf()
        val v = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val pairs = intArrayOf(0, 1, 0, 2, 1, 2)
        for (sweep in 0 until 50) {
            val off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5]
            val diag = a[0] * a[0] + a[4] * a[4] + a[8] * a[8]
            if (off <= 1e-30 * diag || off == 0.0) break
            for (k in 0 until 3) {
                val p = pairs[2 * k]
                val q = pairs[2 * k + 1]
                val apq = a[3 * p + q]
                if (apq == 0.0) continue
                val theta = (a[3 * q + q] - a[3 * p + p]) / (2.0 * apq)
                val t = (if (theta >= 0) 1.0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                val c = 1.0 / sqrt(t * t + 1.0)
                val s = t * c
                for (r in 0 until 3) { // A <- A P
                    val arp = a[3 * r + p]
                    val arq = a[3 * r + q]
                    a[3 * r + p] = c * arp - s * arq
                    a[3 * r + q] = s * arp + c * arq
                }
                for (r in 0 until 3) { // A <- P^T A
                    val apr = a[3 * p + r]
                    val aqr = a[3 * q + r]
                    a[3 * p + r] = c * apr - s * aqr
                    a[3 * q + r] = s * apr + c * aqr
                }
                for (r in 0 until 3) { // V <- V P
                    val vrp = v[3 * r + p]
                    val vrq = v[3 * r + q]
                    v[3 * r + p] = c * vrp - s * vrq
                    v[3 * r + q] = s * vrp + c * vrq
                }
            }
        }
        var best = 0
        for (i in 1 until 3) if (a[4 * i] < a[4 * best]) best = i
        return doubleArrayOf(v[best], v[3 + best], v[6 + best])
    }
}
