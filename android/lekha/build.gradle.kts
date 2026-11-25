// Lekha: a small app for trying kiln's OCR by hand. Plain Android views and no other library than
// kiln, so it builds offline from what the kiln library already needs.
plugins {
    id("com.android.application")
}

android {
    namespace = "dev.kiln.lekha"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.kiln.lekha"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.6.0"
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

// The C test suite's pages go into the app's assets, so it has something to read before the
// first photo. Copied at build time rather than kept twice in the repository.
abstract class CopySamplePages : DefaultTask() {
    @get:InputFiles abstract val pages: ConfigurableFileCollection
    @get:OutputDirectory abstract val out: DirectoryProperty

    @TaskAction fun copy() {
        val dir = out.get().asFile
        dir.deleteRecursively()
        dir.mkdirs()
        pages.forEach { it.copyTo(File(dir, it.name)) }
    }
}

val copySamplePages = tasks.register<CopySamplePages>("copySamplePages") {
    val media = rootDir.parentFile.resolve("tests/media")
    pages.from(listOf("ocr_english.png", "ocr_bengali.png", "ocr_english_upside_down.png").map { media.resolve(it) })
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(copySamplePages, CopySamplePages::out)
    }
}

dependencies {
    implementation(project(":kiln"))
}
