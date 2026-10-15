package dev.kiln

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

// On a phone: the library loads, finds the fast cores, and (when a model is given) chats.
//   ./gradlew :kiln:connectedAndroidTest \
//       -Pandroid.testInstrumentationRunnerArguments.model=/sdcard/Android/data/dev.kiln.test/files/gemma-3-270m-it-Q8_0.gguf
@RunWith(AndroidJUnit4::class)
class DeviceTest {
    @Test
    fun loadsAndProbes() {
        assertEquals("0.5.0", Kiln.version)
        val d = Kiln.probeDevice()
        assertEquals("arm64-v8a", d.abi)
        assertTrue("$d", d.cpuCores > 0 && d.cpuBigCores in 1..d.cpuCores)
        println("kiln ${Kiln.version} on ${d.soc}: ${d.cpuBigCores} of ${d.cpuCores} cores fast, mask 0x${d.cpuFastMask.toString(16)}")
    }

    @Test
    fun chatsWithAModel() {
        val path = InstrumentationRegistry.getArguments().getString("model")
        assumeTrue("no model given", path != null && File(path).canRead())
        KilnModel.load(path!!).use { m ->
            m.newContext().use { ctx ->
                val r = ctx.chatCollect(listOf(Message.user("Name three colours.")), sampling = Sampling.greedy(32))
                assertTrue("$r", r.answer.isNotBlank() && r.stats.generatedTokens > 0)
                println("${m.info.arch}: \"${r.answer}\" at ${r.stats.genTps} tok/s")
            }
        }
    }
}
