// Versions live in gradle/libs.versions.toml. AGP 9 compiles Kotlin itself ("built-in Kotlin"); declaring the
// Kotlin Gradle plugin here pins the Kotlin version both :core and :app compile with.
plugins {
    alias(libs.plugins.kotlin.jvm) apply false
}
