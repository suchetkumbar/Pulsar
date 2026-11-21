package dev.pulsar.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.pow
import kotlin.math.roundToInt

/** sRGB transfer function. All patch averaging happens in linear RGB (mirrors `pulsar.preprocess.color`). */
object Srgb {
    fun toLinear(x: Double): Double = if (x <= 0.04045) x / 12.92 else ((x + 0.055) / 1.055).pow(2.4)

    /** 256-entry uint8 sRGB -> float32 linear LUT (== `srgb8_to_linear_lut()`). */
    val LUT: FloatArray = FloatArray(256) { toLinear(it / 255.0).toFloat() }
}

internal fun JSONArray.ints(): IntArray = IntArray(length()) { getInt(it) }

internal fun JSONArray.doubles(): DoubleArray = DoubleArray(length()) { getDouble(it) }

internal fun JSONArray.strings(): List<String> = List(length()) { getString(it) }

/**
 * Landmark-anchored skin patches (mirrors `pulsar.preprocess.patches.PatchSpec.contract()`, i.e. the
 * `patch_spec` block of model_meta.json). Patch p is a square centred on landmark [landmarkIds]`[p]` with
 * side = [sizeFrac] x face width (distance between the two [faceWidthLandmarks]).
 */
class PatchSpec(
    val landmarkIds: IntArray,
    val regionIds: IntArray,
    val regionNames: List<String>,
    val sizeFrac: Double = 0.09,
    val minPixels: Int = 16,
    val maxViewAngleDeg: Double = 70.0,
    val faceWidthLandmarks: IntArray = intArrayOf(234, 454),
    val viewAngleK: Int = 10,
    val version: String = "v1",
) {
    init {
        require(landmarkIds.size == regionIds.size) { "landmark_ids and region_ids differ in length" }
        require(regionIds.all { it in regionNames.indices }) { "region id outside region_names" }
        require(faceWidthLandmarks.size == 2) { "face_width_landmarks must have 2 entries" }
    }

    val numPatches: Int get() = landmarkIds.size

    fun regionOf(p: Int): String = regionNames[regionIds[p]]

    companion object {
        fun fromJson(o: JSONObject): PatchSpec {
            val landmarkIds = o.getJSONArray("landmark_ids").ints()
            val regionIds = o.getJSONArray("region_ids").ints()
            val names = if (o.has("region_names")) {
                o.getJSONArray("region_names").strings()
            } else {
                // older metadata: recover the order from which region each patch's landmark belongs to
                val regions = o.getJSONObject("regions")
                val byId = HashMap<Int, String>()
                for (name in regions.keys()) {
                    val first = regions.getJSONArray(name).getInt(0)
                    byId[regionIds[landmarkIds.indexOf(first)]] = name
                }
                List(byId.size) { byId.getValue(it) }
            }
            return PatchSpec(
                landmarkIds = landmarkIds,
                regionIds = regionIds,
                regionNames = names,
                sizeFrac = o.getDouble("size_frac"),
                minPixels = o.getInt("min_pixels"),
                maxViewAngleDeg = o.getDouble("max_view_angle_deg"),
                faceWidthLandmarks = o.optJSONArray("face_width_landmarks")?.ints() ?: intArrayOf(234, 454),
                viewAngleK = o.optInt("view_angle_k", 10),
                version = o.optString("version", "v1"),
            )
        }
    }
}

/** Trace-extractor constants (model_meta.json `extractor`; Python `traces.extractor_contract()`). */
data class ExtractorConfig(
    val seedRegions: Set<String> = setOf("cheek_r", "cheek_l"),
    val seedMaxAngleDeg: Double = 50.0,
    val seedMaxPixels: Int = 4000,
    val carryFrames: Int = 3,
    val skinMomentum: Double = 0.95,
    val skinRadius2: Double = 9.21,
    val skinMinSeedPixels: Int = 20,
    val shadowFrac: Double = 0.35,
    val specularPct: Double = 98.0,
    val specularSatMax: Double = 0.15,
) {
    companion object {
        fun fromJson(o: JSONObject?): ExtractorConfig {
            if (o == null) return ExtractorConfig()
            val d = ExtractorConfig()
            return ExtractorConfig(
                seedRegions = o.optJSONArray("seed_regions")?.strings()?.toSet() ?: d.seedRegions,
                seedMaxAngleDeg = o.optDouble("seed_max_angle_deg", d.seedMaxAngleDeg),
                seedMaxPixels = o.optInt("seed_max_pixels", d.seedMaxPixels),
                carryFrames = o.optInt("carry_frames", d.carryFrames),
                skinMomentum = o.optDouble("skin_momentum", d.skinMomentum),
                skinRadius2 = o.optDouble("skin_radius2", d.skinRadius2),
                skinMinSeedPixels = o.optInt("skin_min_seed_pixels", d.skinMinSeedPixels),
                shadowFrac = o.optDouble("shadow_frac", d.shadowFrac),
                specularPct = o.optDouble("specular_pct", d.specularPct),
                specularSatMax = o.optDouble("specular_sat_max", d.specularSatMax),
            )
        }
    }
}

/** Live-window parameters (model_meta.json `window`; Python `live.window_contract()`). */
data class WindowConfig(
    val fsHz: Double = 30.0,
    val winS: Double = 10.0,
    val maxGapS: Double = 0.25,
    val bufferS: Double = 12.0,
) {
    val tLen: Int get() = (winS * fsHz).roundToInt()

    companion object {
        fun fromJson(o: JSONObject?, fsHz: Double, tLen: Int): WindowConfig {
            if (o == null) return WindowConfig(fsHz = fsHz, winS = tLen / fsHz, bufferS = tLen / fsHz + 2.0)
            return WindowConfig(
                fsHz = o.getDouble("fs_hz"),
                winS = o.getDouble("win_s"),
                maxGapS = o.optDouble("max_gap_s", 0.25),
                bufferS = o.optDouble("buffer_s", o.getDouble("win_s") + 2.0),
            )
        }
    }
}

/** model_meta.json: the contract between the Python training side and the phone. */
class ModelMeta(
    val fsHz: Double,
    val tLen: Int,
    val numPatches: Int,
    val bpmLo: Double,
    val bpmHi: Double,
    val bpmStep: Double,
    val temperature: Double,
    val outputs: List<String>,
    val patchSpec: PatchSpec,
    val extractor: ExtractorConfig,
    val window: WindowConfig,
    val numParams: Int,
) {
    init {
        check(patchSpec.numPatches == numPatches) { "patch_spec has ${patchSpec.numPatches} patches, model $numPatches" }
        check(window.tLen == tLen) { "window gives t_len ${window.tLen}, model expects $tLen" }
        check(window.fsHz == fsHz) { "window fs ${window.fsHz} != model fs $fsHz" }
    }

    companion object {
        const val FORMAT = "pulsar-tracenet"

        fun parse(json: String): ModelMeta {
            val o = JSONObject(json)
            require(o.optString("format") == FORMAT) { "not a $FORMAT model_meta.json" }
            val fs = o.getDouble("fs_hz")
            val tLen = o.getInt("t_len")
            val grid = o.getJSONArray("bpm_grid").doubles()
            return ModelMeta(
                fsHz = fs,
                tLen = tLen,
                numPatches = o.getInt("num_patches"),
                bpmLo = grid[0],
                bpmHi = grid[1],
                bpmStep = grid[2],
                temperature = o.optDouble("temperature", 1.0),
                outputs = o.getJSONArray("outputs").strings(),
                patchSpec = PatchSpec.fromJson(o.getJSONObject("patch_spec")),
                extractor = ExtractorConfig.fromJson(o.optJSONObject("extractor")),
                window = WindowConfig.fromJson(o.optJSONObject("window"), fs, tLen),
                numParams = o.optInt("num_params", -1),
            )
        }
    }
}
