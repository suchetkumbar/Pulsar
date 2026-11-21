package dev.pulsar.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

/** Nested JSON number arrays -> flat arrays (JSON null -> NaN). */
object JsonArrays {
    fun doubles(a: JSONArray): DoubleArray {
        val out = ArrayList<Double>(a.length())
        fun walk(x: JSONArray) {
            for (i in 0 until x.length()) {
                val v = x.get(i)
                when {
                    v is JSONArray -> walk(v)
                    v == JSONObject.NULL -> out.add(Double.NaN)
                    else -> out.add((v as Number).toDouble())
                }
            }
        }
        walk(a)
        return out.toDoubleArray()
    }

    fun booleans(a: JSONArray): BooleanArray = doubles(a).let { d -> BooleanArray(d.size) { d[it] != 0.0 } }
}

/** Result of one golden case: max error per field vs its tolerance, and the model's run time. */
class GoldenResult(val name: String, val errors: Map<String, Double>, val tolerance: Map<String, Double>, val modelMs: Double) {
    val failures: List<String> =
        errors.filter { (k, e) -> !(e <= (tolerance[k] ?: 0.0)) }.map { (k, e) -> "$k: $e > ${tolerance[k] ?: 0.0}" }
    val passed: Boolean get() = failures.isEmpty()

    override fun toString(): String =
        "$name: " + (if (passed) "OK" else "FAIL ${failures.joinToString()}") + " (model ${"%.1f".format(modelMs)} ms)"
}

/**
 * Golden-file check (files from scripts/export_model.py -> golden/<case>.json): raw ring buffer ->
 * [LiveWindow] -> model, compared with the Python reference. Used by the JVM test, the instrumented
 * test and the app's self-test, so the phone runs the exact check the desktop does.
 */
object GoldenCheck {
    fun window(case: JSONObject, numPatches: Int, cfg: WindowConfig): ModelWindow? {
        val r = case.getJSONObject("raw")
        return LiveWindow.compute(
            JsonArrays.doubles(r.getJSONArray("t")),
            JsonArrays.doubles(r.getJSONArray("rgb")),
            JsonArrays.booleans(r.getJSONArray("valid")),
            JsonArrays.doubles(r.getJSONArray("n_pix")),
            JsonArrays.doubles(r.getJSONArray("motion")),
            numPatches,
            cfg,
        )
    }

    private fun maxAbs(a: FloatArray, b: DoubleArray): Double {
        if (a.size != b.size) return Double.POSITIVE_INFINITY
        var m = 0.0
        for (i in a.indices) m = maxOf(m, abs(a[i] - b[i]))
        return m
    }

    /** expected.x is `[T][P][3]` (Python live_window); [ModelWindow.x] is `[P][T][3]` (graph layout). */
    fun windowErrors(case: JSONObject, w: ModelWindow): Map<String, Double> {
        val e = case.getJSONObject("expected")
        val tg = JsonArrays.doubles(e.getJSONArray("t_grid"))
        val x = JsonArrays.doubles(e.getJSONArray("x"))
        val v = JsonArrays.booleans(e.getJSONArray("valid"))
        if (tg.size != w.tLen) return mapOf("t_len" to Double.POSITIVE_INFINITY)
        val pN = w.numPatches
        var dt = 0.0
        var dx = 0.0
        var badValid = 0
        for (t in 0 until w.tLen) {
            dt = maxOf(dt, abs(tg[t] - w.tGrid[t]))
            for (p in 0 until pN) {
                for (c in 0 until 3) dx = maxOf(dx, abs(w.x(t, p, c) - x[(t * pN + p) * 3 + c]))
                if (w.valid(t, p) != v[t * pN + p]) badValid++
            }
        }
        return mapOf(
            "t_grid" to dt,
            "x" to dx,
            "valid" to badValid.toDouble(),
            "n_pix" to maxAbs(w.nPix, JsonArrays.doubles(e.getJSONArray("n_pix"))),
            "motion" to abs(e.getDouble("motion") - w.motion),
        )
    }

    fun outputErrors(case: JSONObject, out: TraceNetOutput): Map<String, Double> {
        val e = case.getJSONObject("expected").getJSONObject("outputs")
        fun first(k: String) = e.getJSONArray(k).getDouble(0)
        return mapOf(
            "bvp" to maxAbs(out.bvp, JsonArrays.doubles(e.getJSONArray("bvp"))),
            "alpha" to maxAbs(out.alpha, JsonArrays.doubles(e.getJSONArray("alpha"))),
            "hr" to abs(first("hr") - out.hrBpm),
            "hr_mode" to abs(first("hr_mode") - out.hrModeBpm),
            "conf" to abs(first("conf") - out.confidence),
        )
    }

    fun tolerance(case: JSONObject): Map<String, Double> {
        val t = case.getJSONObject("tolerance")
        val out = HashMap<String, Double>()
        for (k in t.keys()) out[k] = t.getDouble(k) // keys(), not keySet(): Android's org.json hides keySet()
        out["t_grid"] = 1e-8
        out["valid"] = 0.0
        out["hr_mode"] = out["hr"] ?: 0.05
        return out
    }

    fun run(caseJson: String, meta: ModelMeta, model: HrModel): GoldenResult {
        val case = JSONObject(caseJson)
        val name = case.optString("name", "?")
        val tol = tolerance(case)
        val w = window(case, meta.numPatches, meta.window)
            ?: return GoldenResult(name, mapOf("window" to Double.POSITIVE_INFINITY), tol, 0.0)
        val errs = windowErrors(case, w).toMutableMap()
        if (case.getJSONObject("expected").has("outputs")) {
            val t0 = System.nanoTime()
            val out = model.run(w)
            val ms = (System.nanoTime() - t0) / 1e6
            errs += outputErrors(case, out)
            return GoldenResult(name, errs, tol, ms)
        }
        return GoldenResult(name, errs, tol, 0.0)
    }
}
