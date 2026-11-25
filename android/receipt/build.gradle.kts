// Hishab Receipt: a temporary test app. Photo of a shop receipt -> kiln OCR -> rules (shop, kind,
// cross-checked total) -> one short message -> the Hishab model through kiln -> one ledger line.
// Built from Lekha; plain Android views and no library except kiln.
plugins {
    id("com.android.application")
}

android {
    namespace = "dev.kiln.receipt"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.kiln.receipt"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // signed with the debug key: this app is for testing, not for a store
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":kiln"))
}
