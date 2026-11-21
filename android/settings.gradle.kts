pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "pulsar-android"

// :core is pure Kotlin/JVM (preprocessing + ONNX Runtime wrapper), tested on the desktop against the Python
// reference. :app needs the Android SDK; `-Ppulsar.coreOnly=true` skips it (e.g. on a machine without the SDK).
include(":core")
if (providers.gradleProperty("pulsar.coreOnly").orNull != "true") {
    include(":app")
}
