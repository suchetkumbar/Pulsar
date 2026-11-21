package dev.pulsar.core

/** What the UI should show. Abstaining is a feature: it is how Pulsar avoids confident wrong numbers. */
enum class VitalsStatus(val message: String) {
    WARMING_UP("Hold still, measuring..."),
    NO_FACE("No face - look at the camera"),
    LOW_SIGNAL("Low signal - hold still / add light"),
    OK(""),
}

class Vitals(
    val tSec: Double,
    val status: VitalsStatus,
    /** Last model output, kept even when abstaining (the UI may grey it out); null before the first window. */
    val output: TraceNetOutput?,
) {
    val hrBpm: Float? get() = if (status == VitalsStatus.OK) output?.hrBpm else null
}

/**
 * Selective prediction (C4): show HR only when the calibrated confidence reaches [minConfidence]. Use the
 * same tau as the evaluation (docs 06: coverage parity is reported at tau = 0.5), so the phone's coverage
 * matches what was measured.
 */
class AbstainPolicy(val minConfidence: Float = 0.5f) {
    fun decide(tSec: Double, faceFound: Boolean, out: TraceNetOutput?): Vitals = when {
        !faceFound -> Vitals(tSec, VitalsStatus.NO_FACE, out)
        out == null -> Vitals(tSec, VitalsStatus.WARMING_UP, null)
        out.confidence < minConfidence -> Vitals(tSec, VitalsStatus.LOW_SIGNAL, out)
        else -> Vitals(tSec, VitalsStatus.OK, out)
    }
}

/**
 * Frame -> vitals, the whole on-device loop minus the camera and the landmarker:
 * [onFrame] for every camera frame (timestamps in seconds, monotonic), [estimate] about once a second.
 * Not thread-safe; call both from one thread, or hand [window] snapshots to an inference thread.
 */
class PulsarPipeline(
    val meta: ModelMeta,
    private val model: HrModel,
    val policy: AbstainPolicy = AbstainPolicy(),
) {
    val extractor = TraceExtractor(meta.patchSpec, meta.extractor)
    val buffer = LiveBuffer(meta.numPatches, meta.window)
    private var faceFound = false
    private var lastT = Double.NaN

    fun reset() {
        extractor.reset()
        buffer.clear()
        faceFound = false
        lastT = Double.NaN
    }

    fun onFrame(tSec: Double, img: RgbImage, lmNorm: DoubleArray?): FrameStats {
        if (!lastT.isNaN() && tSec < lastT - 1.0) reset() // clock jumped back (camera restarted)
        val st = extractor.process(img, lmNorm)
        buffer.add(tSec, st)
        faceFound = st.faceFound
        lastT = tSec
        return st
    }

    /** Current model window (a copy), or null while the buffer is shorter than the model window. */
    fun window(): ModelWindow? = buffer.window()

    fun estimate(): Vitals {
        if (buffer.size == 0) return Vitals(lastT, VitalsStatus.WARMING_UP, null) // no frame yet
        val w = buffer.window()
        val out = w?.let { model.run(it) }
        return policy.decide(lastT, faceFound, out)
    }
}
