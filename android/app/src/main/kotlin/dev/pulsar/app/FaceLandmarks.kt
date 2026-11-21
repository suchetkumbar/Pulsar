package dev.pulsar.app

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.Closeable
import kotlin.math.max

/**
 * MediaPipe Face Landmarker in VIDEO mode: the same model, mode and thresholds as the Python
 * extraction (`pulsar.preprocess.face.MediaPipeLandmarker`), so the phone's landmarks match training.
 */
class FaceLandmarks(context: Context, modelAsset: String = "model/face_landmarker.task") : Closeable {
    private val landmarker: FaceLandmarker = FaceLandmarker.createFromOptions(
        context,
        FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelAsset).setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setOutputFaceBlendshapes(false)
            .setOutputFacialTransformationMatrixes(false)
            .build(),
    )
    private var lastTs = -1L

    /** Normalised landmarks `[N * 3]` (x, y, z) of the first face in an upright frame, or null. */
    fun detect(upright: Bitmap, timestampMs: Long): DoubleArray? {
        val ts = max(timestampMs, lastTs + 1) // VIDEO mode needs strictly increasing timestamps
        lastTs = ts
        val result = landmarker.detectForVideo(BitmapImageBuilder(upright).build(), ts)
        val face = result.faceLandmarks().firstOrNull() ?: return null
        val out = DoubleArray(face.size * 3)
        for ((i, p) in face.withIndex()) {
            out[3 * i] = p.x().toDouble()
            out[3 * i + 1] = p.y().toDouble()
            out[3 * i + 2] = p.z().toDouble()
        }
        return out
    }

    override fun close() = landmarker.close()
}
