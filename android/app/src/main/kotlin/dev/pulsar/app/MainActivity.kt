package dev.pulsar.app

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExtendableBuilder
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Live heart rate from the front camera, fully on-device. NOT A MEDICAL DEVICE.
 * Buttons: "Log" records estimates to a CSV (for comparison with a pulse oximeter), "Self-test" runs the
 * golden-file parity check and a latency benchmark on this phone.
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: VitalsOverlay
    private lateinit var logButton: Button
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var engine: PulsarEngine? = null
    private val log by lazy { SessionLog(getExternalFilesDir(null) ?: filesDir) }

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else toast("Pulsar needs the camera to measure your pulse.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlay = VitalsOverlay(this)
        logButton = Button(this).apply {
            text = "Log"
            setOnClickListener { toggleLog() }
        }
        val selfTest = Button(this).apply {
            text = "Self-test"
            setOnClickListener { runSelfTest() }
        }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(logButton)
            addView(selfTest)
        }
        setContentView(
            FrameLayout(this).apply {
                addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
                addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
                addView(buttons, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.END))
            },
        )
        cameraExecutor.execute {
            val e = runCatching { PulsarEngine(this) }
            runOnUiThread {
                e.onSuccess {
                    engine = it
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        startCamera()
                    } else {
                        requestCamera.launch(Manifest.permission.CAMERA)
                    }
                }.onFailure { toast("Could not load the model: ${it.message}") }
            }
        }
    }

    /**
     * Camera settings matter as much as the model (docs 07 section 3): a locked 30 fps (AE must not drop the
     * frame rate in dim light, which aliases flicker into the HR band), 50 Hz anti-banding (Indian mains:
     * exposure in multiples of 10 ms cancels 100 Hz tube-light flicker) and no stabilisation (EIS warps pixels
     * between frames). Applied to both use cases so the shared capture request carries them.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private fun <T> applyCaptureSettings(builder: ExtendableBuilder<T>) {
        Camera2Interop.Extender(builder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_ANTIBANDING_MODE, CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_50HZ)
            .setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
    }

    private fun startCamera() {
        val engine = engine ?: return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            // 4:3 at ~640x480 for both use cases, so analysis and preview share one field of view
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                )
                .build()
            val previewBuilder = Preview.Builder().setResolutionSelector(selector)
            applyCaptureSettings(previewBuilder)
            val preview = previewBuilder.build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysisBuilder = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            applyCaptureSettings(analysisBuilder)
            val analysis = analysisBuilder.build()
            analysis.setAnalyzer(
                cameraExecutor,
                VitalsAnalyzer(
                    engine,
                    inferenceExecutor,
                    onFrame = { overlay.update(it) },
                    onVitals = { v, ms ->
                        overlay.update(v, ms)
                        log.write(v, ms)
                    },
                ),
            )
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleLog() {
        if (log.active) {
            log.stop()
            toast("Saved ${log.file?.absolutePath}")
        } else {
            log.start()
        }
        logButton.text = if (log.active) "Stop log" else "Log"
        overlay.logging = log.active
    }

    private fun runSelfTest() {
        val engine = engine ?: return
        toast("Running self-test...")
        inferenceExecutor.execute {
            val report = runCatching { engine.selfTest().toString() }.getOrElse { "Self-test failed: $it" }
            runOnUiThread {
                AlertDialog.Builder(this).setTitle("Pulsar self-test").setMessage(report).setPositiveButton("OK", null).show()
            }
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        super.onDestroy()
        log.stop()
        cameraExecutor.shutdown()
        inferenceExecutor.shutdown()
        // let an in-flight frame / inference finish before the native sessions are closed
        runCatching {
            cameraExecutor.awaitTermination(1, TimeUnit.SECONDS)
            inferenceExecutor.awaitTermination(1, TimeUnit.SECONDS)
        }
        engine?.close()
    }
}
