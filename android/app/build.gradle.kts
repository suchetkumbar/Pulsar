plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "dev.pulsar.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.pulsar.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") } // phones + emulator; trims native libs
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        noCompress += listOf("onnx", "task") // memory-map-friendly model assets
    }
}

dependencies {
    implementation(project(":core"))
    implementation(libs.onnxruntime.android)
    implementation(libs.mediapipe.tasks.vision)
    implementation(libs.camerax.core)
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)
    implementation(libs.activity.ktx)

    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
}

// Model assets come from the Python side and are never committed (gitignored):
//   outputs/export/<pulsar.export>/{tracenet.onnx, model_meta.json, golden/*.json}  (scripts/export_model.py)
//   data/assets/face_landmarker.task                                               (MediaPipe model)
val repoRoot = rootProject.layout.projectDirectory.dir("..")
val exportName = providers.gradleProperty("pulsar.export").orElse("smoke_A5")
val copyPulsarAssets by tasks.registering(Copy::class) {
    description = "Copies the exported TraceNet, its model_meta.json, golden files and the face landmarker into assets/"
    val export = repoRoot.dir("outputs/export").dir(exportName)
    from(export) {
        include("tracenet.onnx", "model_meta.json")
        into("model")
    }
    from(export.map { it.dir("golden") }) {
        include("*.json")
        into("golden")
    }
    from(repoRoot.dir("data/assets")) {
        include("face_landmarker.task")
        into("model")
    }
    into(layout.projectDirectory.dir("src/main/assets"))
    val out = layout.projectDirectory.dir("src/main/assets/model")
    doLast {
        for (f in listOf("tracenet.onnx", "model_meta.json", "face_landmarker.task")) {
            check(out.file(f).asFile.isFile) {
                "missing assets/model/$f: run scripts/export_model.py (or pass -Ppulsar.export=<name>) and make sure " +
                    "data/assets/face_landmarker.task exists (python -c 'from pulsar.preprocess.face import ensure_model; ensure_model()')"
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(copyPulsarAssets) }
