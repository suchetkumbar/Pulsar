import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    `java-library`
    alias(libs.plugins.kotlin.jvm)
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    // Provided at runtime: org.json by the Android platform, ai.onnxruntime by onnxruntime-android in :app.
    compileOnly(libs.orgjson)
    compileOnly(libs.onnxruntime.jvm)

    testImplementation(libs.junit)
    testImplementation(libs.orgjson)
    testImplementation(libs.onnxruntime.jvm)
}

tasks.test {
    useJUnit()
    // GoldenModelTest finds ../outputs/export/<newest>/ by walking up from here; override with -Dpulsar.exportDir
    System.getProperty("pulsar.exportDir")?.let { systemProperty("pulsar.exportDir", it) }
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}
