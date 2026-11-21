package dev.pulsar.core

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * End-to-end port check against a model export (scripts/export_model.py): raw buffer -> Kotlin window ->
 * ONNX Runtime -> outputs == the Python reference in golden/<case>.json. Skipped when no export exists.
 */
class GoldenModelTest {
    @Test
    fun goldenFilesReproduce() {
        val dir = Fixtures.exportDir()
        assumeTrue("no model export found (outputs/export/<name>/ or -Dpulsar.exportDir)", dir != null)
        val meta = ModelMeta.parse(File(dir, "model_meta.json").readText())
        val goldens = File(dir, "golden").listFiles { f -> f.name.endsWith(".json") }?.sortedBy { it.name }.orEmpty()
        assumeTrue("no golden files in $dir", goldens.isNotEmpty())
        TraceNetRunner(File(dir, "tracenet.onnx").readBytes(), meta).use { runner ->
            for (g in goldens) {
                val res = GoldenCheck.run(g.readText(), meta, runner)
                println("golden ${dir!!.name}/$res")
                assertTrue(res.toString(), res.passed)
                assertTrue("${res.name}: model outputs were not checked", "bvp" in res.errors)
            }
        }
    }
}
