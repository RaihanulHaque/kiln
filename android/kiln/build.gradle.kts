plugins {
    id("com.android.library")
}

android {
    namespace = "dev.kiln"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        minSdk = 29
        ndk { abiFilters += listOf("arm64-v8a") }
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                // c++_static: libkiln_jni.so carries its own C++ runtime, so it cannot clash with
                // another library's in the same app
                arguments += listOf("-DANDROID_STL=c++_static", "-DANDROID_PLATFORM=android-29", "-DCMAKE_BUILD_TYPE=Release")
                targets += listOf("kiln_jni")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")   // chatFlow returns a Flow
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

// Host unit tests (src/test): the same JNI file built for this computer, so the Kotlin API and
// the JNI glue can be tested on the desktop JVM against the models in ../../models.
val hostJniDir = layout.buildDirectory.dir("host-jni").get().asFile
val buildHostJni by tasks.registering(Exec::class) {
    val src = file("src/main/cpp")
    commandLine(
        "sh", "-c",
        "cmake -S '$src' -B '$hostJniDir' -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build '$hostJniDir' --target kiln_jni -j 8",
    )
}
tasks.withType<Test>().configureEach {
    dependsOn(buildHostJni)
    systemProperty("java.library.path", hostJniDir.absolutePath)
    systemProperty("kiln.repo", rootDir.parentFile.absolutePath)
}
