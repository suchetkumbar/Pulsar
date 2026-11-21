package dev.pulsar.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.zip.GZIPInputStream

/** Loads the Python-generated parity fixtures (scripts/make_android_fixtures.py) from test resources. */
object Fixtures {
    private fun stream(name: String) =
        Fixtures::class.java.getResourceAsStream("/fixtures/$name")
            ?: error("missing test resource fixtures/$name - run: uv run --no-sync python scripts/make_android_fixtures.py")

    fun index(): JSONObject = JSONObject(stream("index.json").bufferedReader().use { it.readText() })

    fun gz(name: String): JSONObject =
        JSONObject(GZIPInputStream(stream(name)).bufferedReader(Charsets.UTF_8).use { it.readText() })

    /**
     * A model export directory (tracenet.onnx + model_meta.json + golden/) for the ONNX golden test:
     * `-Dpulsar.exportDir=...`, env `PULSAR_EXPORT_DIR`, else the newest `outputs/export/<name>` in the repo.
     */
    fun exportDir(): File? {
        val explicit = System.getProperty("pulsar.exportDir") ?: System.getenv("PULSAR_EXPORT_DIR")
        if (explicit != null) return File(explicit).takeIf { File(it, "model_meta.json").isFile }
        var dir: File? = File("").absoluteFile
        while (dir != null) {
            val exports = File(dir, "outputs/export")
            if (exports.isDirectory) {
                return exports.listFiles()
                    ?.filter { File(it, "model_meta.json").isFile && File(it, "tracenet.onnx").isFile }
                    ?.maxByOrNull { File(it, "model_meta.json").lastModified() }
            }
            dir = dir.parentFile
        }
        return null
    }
}

fun JSONArray.flatDoubles(): DoubleArray = JsonArrays.doubles(this)

fun JSONArray.flatBooleans(): BooleanArray = JsonArrays.booleans(this)

fun maxAbsDiff(a: FloatArray, b: DoubleArray): Double {
    require(a.size == b.size) { "size ${a.size} != ${b.size}" }
    var m = 0.0
    for (i in a.indices) {
        val x = a[i].toDouble()
        val y = b[i]
        if (x.isNaN() || y.isNaN()) {
            if (x.isNaN() != y.isNaN()) return Double.POSITIVE_INFINITY
            continue
        }
        m = maxOf(m, kotlin.math.abs(x - y))
    }
    return m
}
