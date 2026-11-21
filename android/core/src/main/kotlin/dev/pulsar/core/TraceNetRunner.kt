package dev.pulsar.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** Outputs of the exported TraceNet graph for one window (batch 1). */
class TraceNetOutput(
    /** Pulse waveform `[T]` (unit-free). */
    val bvp: FloatArray,
    /** Soft-argmax HR under the temperature-calibrated spectrum (bpm). */
    val hrBpm: Float,
    /** Arg-max HR bin (bpm). */
    val hrModeBpm: Float,
    /** Calibrated confidence: probability mass within +-5 bpm of [hrModeBpm]. */
    val confidence: Float,
    /** Patch attention weights `[P]` (sum to 1 over valid patches). */
    val alpha: FloatArray,
)

/** Anything that maps a [ModelWindow] to [TraceNetOutput] (ONNX Runtime on device, fakes in tests). */
fun interface HrModel {
    fun run(w: ModelWindow): TraceNetOutput
}

/**
 * ONNX Runtime session for `tracenet.onnx` (inputs x, valid, n_pix, motion; see model_meta.json).
 *
 * The same `ai.onnxruntime` API ships as `onnxruntime-android` (AAR) on the phone and `onnxruntime`
 * (JAR) on the desktop JVM, so this class is tested on the desktop against the Python golden files.
 */
class TraceNetRunner(
    model: ByteArray,
    val meta: ModelMeta,
    options: OrtSession.SessionOptions? = null,
) : HrModel, AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ownsOptions = options == null
    private val opts = options ?: defaultOptions()
    private val session: OrtSession = env.createSession(model, opts)

    override fun run(w: ModelWindow): TraceNetOutput {
        val p = meta.numPatches.toLong()
        val t = meta.tLen.toLong()
        require(w.numPatches.toLong() == p && w.tLen.toLong() == t) {
            "window ${w.numPatches}x${w.tLen} does not match the model ${p}x$t"
        }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(w.x), longArrayOf(1, p, t, 3)).use { x ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(w.valid), longArrayOf(1, p, t)).use { v ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(w.nPix), longArrayOf(1, p)).use { np ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(w.motion)), longArrayOf(1)).use { m ->
                        session.run(mapOf("x" to x, "valid" to v, "n_pix" to np, "motion" to m)).use { r ->
                            fun out(name: String): FloatArray {
                                val value = r.get(name).orElseThrow { IllegalStateException("model has no output $name") }
                                val buf = (value as OnnxTensor).floatBuffer
                                return FloatArray(buf.remaining()).also { buf.get(it) }
                            }
                            return TraceNetOutput(
                                bvp = out("bvp"),
                                hrBpm = out("hr")[0],
                                hrModeBpm = out("hr_mode")[0],
                                confidence = out("conf")[0],
                                alpha = out("alpha"),
                            )
                        }
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
        if (ownsOptions) opts.close()
    }

    companion object {
        /** CPU, all graph optimisations; on Android add XNNPACK via [OrtSession.SessionOptions.addXnnpack]. */
        fun defaultOptions(threads: Int = 2): OrtSession.SessionOptions =
            OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(threads)
            }
    }
}
