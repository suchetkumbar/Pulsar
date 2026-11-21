package dev.pulsar.app

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device port check: the golden files bundled in assets/golden (synthetic raw buffers + the Python
 * reference outputs) must reproduce through the Kotlin window + ONNX Runtime Mobile (XNNPACK) on this phone.
 *
 *   cd android && ./gradlew :app:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class GoldenDeviceTest {
    @Test
    fun goldenFilesReproduceOnDevice() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        PulsarEngine(ctx).use { engine ->
            val report = engine.selfTest(latencyRuns = 50)
            Log.i("PulsarGolden", report.toString())
            assertTrue("no golden files bundled (run scripts/export_model.py first)", report.golden.isNotEmpty())
            assertTrue(report.toString(), report.passed)
        }
    }
}
