package dev.kiln

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

// Runs the Kotlin API and the JNI glue on the desktop JVM against the test models of the C
// suite (scripts/fetch_models.sh --media --speech). A test whose model is missing is skipped.
// Needs libkiln_jni built for this computer on java.library.path (Gradle's test task does that).
class ModelsTest {
    private val repo: File = System.getProperty("kiln.repo")?.let(::File)
        ?: generateSequence(File("").absoluteFile) { it.parentFile }.first { File(it, "include/kiln/kiln.h").exists() }
    private val models = System.getenv("KILN_MODELS")?.let(::File) ?: File(repo, "models")

    private fun model(name: String): String {
        val f = File(models, name)
        assumeTrue("missing $f", f.exists())
        return f.path
    }

    private fun media(name: String) = File(repo, "tests/media/$name").path

    private fun wordsOnly(s: String) = s.lowercase().split(Regex("[^a-z0-9]+")).filter { it.isNotEmpty() }.joinToString(" ")

    private fun dot(a: FloatArray, b: FloatArray): Float = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()

    @Test
    fun probeVersionAndDefaults() {
        val d = Kiln.probeDevice()
        assertTrue("$d", d.cpuCores > 0 && d.cpuBigCores > 0 && d.ramTotalBytes > 0)
        assertEquals("0.5.0", Kiln.version)
        // the Kotlin defaults are kiln's defaults
        val n = Native.defaults()
        val s = Sampling()
        assertEquals(n[0].toFloat(), s.temperature)
        assertEquals(n[1].toFloat(), s.topP)
        assertEquals(n[2].toInt(), s.topK)
        assertEquals(n[3].toFloat(), s.minP)
        assertEquals(n[4].toFloat(), s.repeatPenalty)
        assertEquals(n[5].toInt(), s.repeatLastN)
        assertEquals(n[6].toInt(), s.seed)
        assertEquals(n[7].toInt(), s.maxTokens)
        val c = ContextParams()
        assertEquals(n[8].toInt(), c.nCtx)
        assertEquals(n[9].toInt(), c.nBatch)
        assertEquals(n[10] != 0.0, c.flashAttn)
        val m = ModelParams()
        assertEquals(n[11].toInt(), m.nGpuLayers)
        assertEquals(n[12].toInt(), m.nThreads)
        assertEquals(n[13] != 0.0, m.useMmap)
        assertEquals(n[14] != 0.0, m.useMlock)
        val o = ChatOptions()
        assertEquals(n[15] != 0.0, o.addAssistantPrefix)
        assertEquals(n[16] != 0.0, o.enableThinking)
        assertEquals(n[17].toInt(), o.maxReasoningTokens)
        println("${d.soc} (${d.cpuCores} cores, ${d.cpuBigCores} fast), kiln ${Kiln.version}")
    }

    @Test
    fun errorsCarryKilnsMessage() {
        try {
            KilnModel.load("/nonexistent/model.gguf")
            fail("loaded a missing file")
        } catch (e: KilnException) {
            assertEquals(Status.MODEL_LOAD, e.status)
            assertTrue(e.message!!.contains("model"))
        }
    }

    @Test
    fun utf8CarryHoldsSplitCharacters() {
        val c = Utf8Carry()
        val e = "é".encodeToByteArray()
        assertEquals("caf", c.push("caf".encodeToByteArray()))
        assertEquals("", c.push(e.copyOfRange(0, 1)))
        assertEquals("é", c.push(e.copyOfRange(1, 2)))
        val smile = "🙂".encodeToByteArray()
        assertEquals("", c.push(smile.copyOfRange(0, 3)))
        assertEquals("🙂", c.push(smile.copyOfRange(3, 4)))
        assertEquals("a�b", c.push(byteArrayOf('a'.code.toByte(), 0xFF.toByte(), 'b'.code.toByte())))
    }

    @Test
    fun chatStreamsReusesCacheAndStops() {
        KilnModel.load(model("gemma-3-270m-it-Q8_0.gguf")).use { m ->
            assertEquals("gemma3", m.info.arch)
            assertTrue(!m.info.isDecisionModel && !m.info.isEmbeddingModel && !m.info.isSpeechModel)
            val toks = m.tokenize("Hello, world 🙂")
            assertEquals("Hello, world 🙂", m.detokenize(toks))
            assertTrue(m.applyChatTemplate(listOf(Message.user("Name a fruit."))).contains("Name a fruit."))

            m.newContext().use { ctx ->
                val greedy = Sampling.greedy(maxTokens = 24)
                val turn = mutableListOf(Message.user("Name three colours, separated by commas."))
                val streamed = StringBuilder()
                var pieces = 0
                val stats = ctx.chat(turn, sampling = greedy) {
                    assertEquals(Channel.ANSWER, it.channel)
                    streamed.append(it.text)
                    pieces++
                    true
                }
                assertTrue(streamed.isNotBlank() && pieces > 1)
                assertTrue("$stats", stats.generatedTokens > 0 && stats.promptTokens > 5 && stats.genTps > 0)
                println("reply: \"$streamed\" (${stats.generatedTokens} tok at ${"%.0f".format(stats.genTps)} tok/s)")

                ctx.reset()
                val again = ctx.chatCollect(turn, sampling = greedy)
                assertEquals(streamed.toString(), again.answer)

                turn += Message.assistant(again.answer)
                turn += Message.user("And one more?")
                val next = ctx.chatCollect(turn, sampling = greedy)
                assertTrue("${next.stats}", next.stats.cachedTokens > 0)

                var n = 0
                val stopped = ctx.chat(listOf(Message.user("Count from 1 to 50.")), sampling = Sampling.greedy()) { ++n < 3 }
                assertTrue("$stopped n=$n", stopped.stoppedByUser && n == 3)
            }
        }
    }

    @Test
    fun cancelFlowAndExceptions() {
        KilnModel.load(model("gemma-3-270m-it-Q8_0.gguf")).use { m ->
            m.newContext().use { ctx ->
                val long = listOf(Message.user("Write a long story about a lighthouse keeper."))
                val sampling = Sampling.greedy(maxTokens = 2000)

                // cancel from another thread
                var first = true
                try {
                    ctx.chat(long, sampling = sampling) {
                        if (first) {
                            first = false
                            Thread { ctx.cancel() }.apply { start() }.join()
                        }
                        true
                    }
                    fail("not cancelled")
                } catch (e: KilnException) {
                    assertEquals(Status.CANCELLED, e.status)
                }

                // an exception in the callback reaches the caller, and the context still works
                try {
                    ctx.chat(long, sampling = sampling) { throw IllegalStateException("from the callback") }
                    fail("exception lost")
                } catch (e: IllegalStateException) {
                    assertEquals("from the callback", e.message)
                }
                assertTrue(ctx.chatCollect(listOf(Message.user("Say hi.")), sampling = Sampling.greedy(8)).stats.generatedTokens > 0)

                // a Flow: whole reply, then the stats
                val events = runBlocking { ctx.chatFlow(listOf(Message.user("Name a fruit.")), sampling = Sampling.greedy(16)).toList() }
                assertTrue(events.dropLast(1).all { it is Piece } && events.last() is GenStats)

                // a collector that stops early cancels the generation, and the context is free again
                val few = runBlocking { ctx.chatFlow(long, sampling = sampling).take(3).toList() }
                assertEquals(3, few.size)
                assertTrue(ctx.chatCollect(listOf(Message.user("Say hi.")), sampling = Sampling.greedy(8)).stats.generatedTokens > 0)
            }
            // closing the model while a context is open is refused
            val ctx = m.newContext()
            try {
                m.close()
                fail("closed with a context open")
            } catch (_: IllegalStateException) {
            }
            ctx.close()
            ctx.cancel() // after close: does nothing
        }
    }

    @Test
    fun thinkingGoesToItsOwnChannel() {
        KilnModel.load(model("Qwen3.5-0.8B-Q4_K_M.gguf")).use { m ->
            assertTrue(m.info.supportsThinking)
            m.newContext().use { ctx ->
                val r = ctx.chatCollect(
                    listOf(Message.user("What is 12 times 7?")),
                    ChatOptions(enableThinking = true, maxReasoningTokens = 32),
                    Sampling.greedy(128),
                )
                assertTrue("$r", r.reasoning.isNotBlank() && r.stats.reasoningTokens in 1..33)
                assertTrue(!r.answer.contains("<think>") && !r.reasoning.contains("<think>"))
                println("reasoning ${r.stats.reasoningTokens} tok, answer \"${r.answer}\"")
            }
        }
    }

    @Test
    fun decisionModel() {
        KilnModel.load(model("d1-omni-600M-Q8_0.gguf")).use { m ->
            assertTrue(m.info.isDecisionModel)
            m.newContext(ContextParams(nCtx = 4096)).use { ctx ->
                val state = "Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and nobody " +
                    "answered. I want my money back today or I will dispute the charge with my bank."
                val refund = ctx.decide(state, Question("wants_refund", QuestionType.YES_NO, "Is the customer asking for a refund?"))
                assertEquals(2, refund.probs.size)
                assertTrue(abs(refund.probs.sum() - 1f) < 1e-3f)
                assertTrue("$refund", refund.value > 0.5f && refund.best == 1)
                val weather = ctx.decide(state, Question("about_weather", QuestionType.YES_NO, "Is the customer asking about the weather?"))
                assertTrue(weather.value < 0.5f && weather.best == 0)

                val team = Question(
                    "team", QuestionType.CHOICE, "Which team should handle this?",
                    listOf(
                        DecisionOption("billing", "Payments, refunds and invoices"),
                        DecisionOption("shipping", "Delivery status and lost parcels"),
                        DecisionOption("technical", "The product does not work as expected"),
                    ),
                )
                val t = ctx.decide(state, team)
                assertTrue(t.probs.size == 3 && t.best in 0..2 && t.confidence in 0f..1f)
                assertArrayEquals(t.probs, ctx.decide(state, team).probs, 0f)
                val urgency = Question(
                    "urgency", QuestionType.SCORE, "How urgent is this?",
                    listOf(DecisionOption(description = "Can wait"), DecisionOption(description = "Today"), DecisionOption(description = "Blocking the customer now")),
                )
                val u = ctx.decide(state, urgency)
                assertTrue("$u", u.probs.size == 3 && u.value in 0f..2f)
                println("refund? yes=${"%.4f".format(refund.value)}; team: ${team.options[t.best].key}; urgency ${"%.2f".format(u.value)}")
            }
        }
    }

    @Test
    fun embeddingsRankAndTruncate() {
        KilnModel.load(model("embeddinggemma-2-UD-Q4_K_XL.gguf")).use { m ->
            val n = m.info.nEmbd
            assertTrue(m.info.isEmbeddingModel && n > 0)
            m.newContext().use { ctx ->
                val q = ctx.embed("when does the train to London leave", EmbedOptions.QUERY)
                assertEquals(n, q.vector.size)
                assertTrue(q.stats.prefixed && abs(dot(q.vector, q.vector) - 1f) < 1e-4f)
                val train = ctx.embed("The train to London leaves at nine in the morning.", EmbedOptions.DOCUMENT)
                val flour = ctx.embed("Add two cups of flour and one egg to the bowl.", EmbedOptions.DOCUMENT)
                assertTrue(dot(q.vector, train.vector) > dot(q.vector, flour.vector))
                assertEquals(256, ctx.embed("hello", EmbedOptions(dimensions = 256)).vector.size)
                println("train ${"%.3f".format(dot(q.vector, train.vector))}, flour ${"%.3f".format(dot(q.vector, flour.vector))}")
            }
        }
    }

    @Test
    fun imageEmbeddingsFromFileAndBytes() {
        val path = model("embeddinggemma-2-UD-Q4_K_XL.gguf")
        val proj = model("mmproj-embeddinggemma-2-Q8_0.gguf")
        KilnModel.load(path).use { m ->
            val pi = m.loadProjector(proj)
            assertTrue(pi.supportsImages && !pi.generatesAudio)
            m.newContext().use { ctx ->
                val red = ctx.embedMedia(null, listOf(Media.File(media("red_circle.png"))))
                assertTrue("${red.stats}", red.stats.mediaTokens > 0 && red.stats.mediaMs > 0)
                val red2 = ctx.embedMedia(null, listOf(Media.Bytes(File(media("red_circle.png")).readBytes())))
                assertArrayEquals(red.vector, red2.vector, 0f)
                val blue = ctx.embedMedia(null, listOf(Media.File(media("blue_square.png"))))
                val q = ctx.embed("a red circle", EmbedOptions.QUERY)
                assertTrue(dot(q.vector, red.vector) > dot(q.vector, blue.vector))
                val mixed = ctx.embedMedia("A shape $MEDIA_MARKER on white", listOf(Media.File(media("red_circle.png"))))
                assertTrue(mixed.stats.tokens > red.stats.tokens)
            }
        }
    }

    private fun transcribeTrainClip(file: String, expectLang: String) {
        KilnModel.load(model(file)).use { m ->
            assertTrue("${m.info}", m.info.isSpeechModel && m.info.backendName.startsWith("whisper.cpp/"))
            m.newContext().use { ctx ->
                val clip = media("speech_train.wav")
                val t = ctx.transcribeCollect(Media.File(clip))
                val w = wordsOnly(t.text)
                assertTrue(
                    "got \"${t.text}\"",
                    w == "the train to london leaves at nine in the morning" || w == "the train to london leaves at 9 in the morning",
                )
                assertEquals(expectLang, t.stats.language)
                assertTrue("${t.stats}", t.stats.audioSeconds in 3.0..3.3 && t.stats.stoppedByEos)
                assertEquals(t.text, ctx.transcribeCollect(Media.Bytes(File(clip).readBytes())).text)
                try {
                    ctx.transcribe(Media.File(clip)) { throw IllegalStateException("from onText") }
                    fail("exception lost")
                } catch (e: IllegalStateException) {
                    assertEquals("from onText", e.message)
                }
                println("$file: \"${t.text}\", ${"%.1f".format(t.stats.audioSeconds)} s of audio in ${"%.0f".format(t.stats.ms)} ms")
            }
        }
    }

    @Test
    fun transcribeParakeet() = transcribeTrainClip("ggml-parakeet-tdt-0.6b-v3-q8_0.bin", "")

    @Test
    fun transcribeWhisper() = transcribeTrainClip("ggml-base-q5_1.bin", "English")

    @Test
    fun speakAndReadBack() {
        val path = model("pocket-tts-en.gguf")
        val proj = model("mmproj-pocket-tts-en.gguf")
        val text = "The train to London leaves at nine in the morning."
        val speech = KilnModel.load(path).use { m ->
            val pi = m.loadProjector(proj)
            assertTrue("$pi", pi.generatesAudio && pi.sampleRate == 24000)
            m.newContext().use { ctx ->
                val opts = SpeakOptions(voice = Media.File(media("speech_train.wav")), seed = 7)
                val sp = ctx.speakCollect(text, opts)
                assertEquals(24000, sp.sampleRate)
                val secs = sp.samples.size / 24000.0
                assertTrue("$secs s ${sp.stats}", secs in 1.5..8.0 && sp.stats.stoppedByEos)
                val peak = sp.samples.maxOf { abs(it) }
                assertTrue("peak $peak", peak > 0.05f && peak <= 1f)
                var pieces = 0
                val st = ctx.speak(text, opts) { _, _ -> pieces++; false }
                assertTrue("$st", st.stoppedByUser && pieces == 1)
                try {
                    ctx.speak(text, opts) { _, _ -> throw IllegalStateException("from onAudio") }
                    fail("exception lost")
                } catch (e: IllegalStateException) {
                    assertEquals("from onAudio", e.message)
                }
                println("${"%.2f".format(secs)} s of audio in ${"%.0f".format(sp.stats.ms)} ms")
                sp
            }
        }
        // the honest check: Parakeet writes the words down again
        val asr = File(models, "ggml-parakeet-tdt-0.6b-v3-q8_0.bin")
        if (!asr.exists()) return
        val heard = KilnModel.load(asr.path).use { m ->
            m.newContext().use { it.transcribeCollect(Media.Bytes(wav(speech.samples, speech.sampleRate))) }
        }
        val w = wordsOnly(heard.text)
        assertTrue("Parakeet heard \"${heard.text}\"", w.contains("train") && w.contains("london"))
        println("read back: \"${heard.text}\"")
    }

    /** 16-bit mono PCM wav in memory. */
    private fun wav(samples: FloatArray, rate: Int): ByteArray {
        val data = samples.size * 2
        val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray()).putInt(16)
            .putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
            .put("data".toByteArray()).putInt(data)
        for (s in samples) b.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        return ByteArrayOutputStream().apply { write(b.array()) }.toByteArray()
    }
}
