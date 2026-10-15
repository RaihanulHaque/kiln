package dev.kiln.reactnative

import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import dev.kiln.Backend
import dev.kiln.ChatOptions
import dev.kiln.ContextParams
import dev.kiln.DecisionOption
import dev.kiln.EmbedKind
import dev.kiln.EmbedOptions
import dev.kiln.Embedding
import dev.kiln.GenStats
import dev.kiln.Kiln
import dev.kiln.KilnContext
import dev.kiln.KilnException
import dev.kiln.KilnModel
import dev.kiln.LogLevel
import dev.kiln.Media
import dev.kiln.Message
import dev.kiln.ModelParams
import dev.kiln.Piece
import dev.kiln.ProjectorParams
import dev.kiln.Question
import dev.kiln.QuestionType
import dev.kiln.Role
import dev.kiln.Sampling
import dev.kiln.SpeakOptions
import dev.kiln.TranscribeOptions
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * The TurboModule behind src/NativeKiln.ts. It holds the models and contexts by number (JS
 * never sees a native pointer) and runs every blocking call on a background thread, so the JS
 * thread is never blocked. The work itself is the dev.kiln Kotlin library in ../../android.
 */
class KilnModule(context: ReactApplicationContext) : NativeKilnSpec(context) {
    private val models = ConcurrentHashMap<Int, KilnModel>()
    private val contexts = ConcurrentHashMap<Int, KilnContext>()
    private val stopped = ConcurrentHashMap.newKeySet<Int>()
    private val ids = AtomicInteger(0)
    private val worker: ExecutorService = Executors.newCachedThreadPool { r -> Thread(r, "kiln-worker").apply { isDaemon = true } }

    /** Runs [block] off the JS thread and settles [promise] with its result or error. */
    private fun run(promise: Promise, block: () -> Any?) {
        worker.execute {
            try {
                promise.resolve(block())
            } catch (e: KilnException) {
                promise.reject(e.status.name, e.message, e)
            } catch (e: IllegalStateException) {
                promise.reject("INVALID_STATE", e.message, e)
            } catch (e: Throwable) {
                promise.reject("INTERNAL", e.message ?: e.toString(), e)
            }
        }
    }

    private fun model(id: Double): KilnModel = models[id.toInt()] ?: throw IllegalStateException("no model ${id.toInt()} (closed?)")
    private fun ctx(id: Double): KilnContext = contexts[id.toInt()] ?: throw IllegalStateException("no context ${id.toInt()} (closed?)")

    // ------------------------------------------------------------ library

    override fun version(): String = Kiln.version

    override fun probeDevice(promise: Promise) = run(promise) {
        val d = Kiln.probeDevice()
        Arguments.createMap().apply {
            putString("soc", d.soc); putString("abi", d.abi)
            putInt("cpuCores", d.cpuCores); putInt("cpuBigCores", d.cpuBigCores)
            putString("cpuFastMask", java.lang.Long.toUnsignedString(d.cpuFastMask))
            putDouble("ramTotalBytes", d.ramTotalBytes.toDouble()); putDouble("ramAvailableBytes", d.ramAvailableBytes.toDouble())
            putBoolean("hasFp16", d.hasFp16); putBoolean("hasDotprod", d.hasDotprod); putBoolean("hasI8mm", d.hasI8mm)
            putBoolean("hasSve", d.hasSve); putBoolean("hasGpu", d.hasGpu); putString("gpuName", d.gpuName)
            putBoolean("hasNpu", d.hasNpu); putString("npuName", d.npuName)
        }
    }

    override fun setLogLevel(level: Double) = Kiln.setLogLevel(LogLevel.entries[level.toInt().coerceIn(0, 3)])

    // ------------------------------------------------------------ models

    override fun loadModel(path: String, backend: String, nGpuLayers: Double, nThreads: Double, useMmap: Boolean, promise: Promise) =
        run(promise) {
            val b = Backend.entries.firstOrNull { it.name.equals(backend, ignoreCase = true) } ?: Backend.AUTO
            val m = KilnModel.load(path, ModelParams(backend = b, nGpuLayers = nGpuLayers.toInt(), nThreads = nThreads.toInt(), useMmap = useMmap))
            val id = ids.incrementAndGet()
            models[id] = m
            val i = m.info
            Arguments.createMap().apply {
                putInt("handle", id)
                putString("backendName", i.backendName); putString("arch", i.arch); putString("quant", i.quant)
                putDouble("nParams", i.nParams.toDouble()); putInt("nCtxTrain", i.nCtxTrain); putInt("nVocab", i.nVocab)
                putInt("nGpuLayersUsed", i.nGpuLayersUsed); putDouble("modelBytes", i.modelBytes.toDouble())
                putBoolean("supportsThinking", i.supportsThinking); putBoolean("isDecisionModel", i.isDecisionModel)
                putBoolean("isEmbeddingModel", i.isEmbeddingModel); putInt("nEmbd", i.nEmbd); putBoolean("isSpeechModel", i.isSpeechModel)
            }
        }

    override fun freeModel(model: Double, promise: Promise) = run(promise) {
        val m = models[model.toInt()] ?: return@run null
        m.close() // throws while the model still has open contexts; then it stays registered
        models.remove(model.toInt())
        null
    }

    override fun loadProjector(model: Double, path: String, imageMaxTokens: Double, promise: Promise) = run(promise) {
        val p = model(model).loadProjector(path, ProjectorParams(imageMaxTokens.toInt()))
        Arguments.createMap().apply {
            putBoolean("supportsImages", p.supportsImages); putBoolean("supportsAudio", p.supportsAudio)
            putBoolean("generatesAudio", p.generatesAudio); putInt("sampleRate", p.sampleRate)
        }
    }

    override fun tokenize(model: Double, text: String, addSpecial: Boolean, promise: Promise) = run(promise) {
        Arguments.createArray().apply { model(model).tokenize(text, addSpecial).forEach { pushInt(it) } }
    }

    override fun detokenize(model: Double, tokens: ReadableArray, promise: Promise) = run(promise) {
        model(model).detokenize(IntArray(tokens.size()) { tokens.getInt(it) })
    }

    override fun applyChatTemplate(model: Double, messages: ReadableArray, options: ReadableMap, promise: Promise) = run(promise) {
        model(model).applyChatTemplate(messages.toMessages(), options.toChatOptions())
    }

    // ------------------------------------------------------------ contexts

    override fun newContext(model: Double, nCtx: Double, nBatch: Double, flashAttn: Boolean, promise: Promise) = run(promise) {
        val c = model(model).newContext(ContextParams(nCtx.toInt(), nBatch.toInt(), flashAttn))
        val id = ids.incrementAndGet()
        contexts[id] = c
        id
    }

    override fun freeContext(context: Double, promise: Promise) = run(promise) {
        contexts.remove(context.toInt())?.close()
        null
    }

    override fun resetContext(context: Double, promise: Promise) = run(promise) {
        ctx(context).reset()
        null
    }

    override fun cancel(context: Double) {
        contexts[context.toInt()]?.cancel()
    }

    override fun stop(requestId: Double) {
        stopped.add(requestId.toInt())
    }

    /** The streaming callback for one call: sends each piece to JS, and stops when JS asked. */
    private fun pieces(requestId: Double): (Piece) -> Boolean {
        val id = requestId.toInt()
        return { p ->
            emitOnPiece(
                Arguments.createMap().apply {
                    putInt("requestId", id)
                    putString("channel", if (p.channel == dev.kiln.Channel.REASONING) "reasoning" else "answer")
                    putString("text", p.text)
                    putInt("tokenId", p.tokenId)
                },
            )
            !stopped.contains(id)
        }
    }

    override fun chat(context: Double, requestId: Double, messages: ReadableArray, options: ReadableMap, sampling: ReadableMap, promise: Promise) =
        run(promise) {
            try {
                ctx(context).chat(messages.toMessages(), options.toChatOptions(), sampling.toSampling(), pieces(requestId)).toMap()
            } finally {
                stopped.remove(requestId.toInt())
            }
        }

    override fun generate(context: Double, requestId: Double, prompt: String, sampling: ReadableMap, promise: Promise) = run(promise) {
        try {
            ctx(context).generate(prompt, sampling.toSampling(), pieces(requestId)).toMap()
        } finally {
            stopped.remove(requestId.toInt())
        }
    }

    override fun decide(context: Double, state: String?, question: ReadableMap, promise: Promise) = run(promise) {
        val opts = question.getArray("options")
        val q = Question(
            id = question.getString("id") ?: "",
            type = when (question.getString("type")) {
                "choice" -> QuestionType.CHOICE
                "score" -> QuestionType.SCORE
                else -> QuestionType.YES_NO
            },
            instructions = question.getString("instructions") ?: "",
            options = List(opts?.size() ?: 0) { i ->
                val o = opts!!.getMap(i)
                DecisionOption(o?.str("key"), o?.str("description"))
            },
        )
        val d = ctx(context).decide(state, q)
        Arguments.createMap().apply {
            putArray("probs", Arguments.createArray().apply { d.probs.forEach { pushDouble(it.toDouble()) } })
            putInt("best", d.best); putDouble("value", d.value.toDouble()); putDouble("confidence", d.confidence.toDouble())
            putInt("promptTokens", d.promptTokens); putDouble("ms", d.ms)
        }
    }

    override fun embed(context: Double, text: String, options: ReadableMap, promise: Promise) = run(promise) {
        ctx(context).embed(text, options.toEmbedOptions()).toMap()
    }

    override fun embedMedia(context: Double, text: String?, media: ReadableArray, options: ReadableMap, promise: Promise) = run(promise) {
        val items = List(media.size()) { media.getMap(it)!!.toMedia() }
        ctx(context).embedMedia(text, items, options.toEmbedOptions()).toMap()
    }

    override fun transcribe(context: Double, requestId: Double, audio: ReadableMap, options: ReadableMap, promise: Promise) = run(promise) {
        val each = pieces(requestId)
        val text = StringBuilder()
        try {
            val st = ctx(context).transcribe(
                audio.toMedia(),
                TranscribeOptions(options.str("language"), options.str("prompt"), options.int("maxTokens")),
            ) { t ->
                text.append(t)
                each(Piece(dev.kiln.Channel.ANSWER, t, 0))
            }
            Arguments.createMap().apply {
                putString("text", text.toString())
                putDouble("audioSeconds", st.audioSeconds); putInt("audioTokens", st.audioTokens); putInt("promptTokens", st.promptTokens)
                putInt("generatedTokens", st.generatedTokens); putDouble("encodeMs", st.encodeMs); putDouble("promptMs", st.promptMs)
                putDouble("genMs", st.genMs); putDouble("ms", st.ms); putString("language", st.language)
                putBoolean("stoppedByEos", st.stoppedByEos); putBoolean("stoppedByUser", st.stoppedByUser)
            }
        } finally {
            stopped.remove(requestId.toInt())
        }
    }

    override fun speak(context: Double, text: String, options: ReadableMap, outPath: String, promise: Promise) = run(promise) {
        val voice = if (options.hasKey("voice") && !options.isNull("voice")) options.getMap("voice")!!.toMedia() else null
        val seed = options.int("seed")
        val sp = ctx(context).speakCollect(
            text,
            SpeakOptions(
                voice = voice, language = options.str("language"), maxFrames = options.int("maxFrames"), topK = options.int("topK"),
                topP = if (options.hasKey("topP")) options.getDouble("topP").toFloat() else 0f, seed = if (seed < 0) null else seed,
            ),
        )
        writeWav(File(outPath), sp.samples, sp.sampleRate)
        val st = sp.stats
        Arguments.createMap().apply {
            putString("path", outPath); putInt("sampleRate", sp.sampleRate); putDouble("audioSeconds", st.audioSeconds)
            putInt("frames", st.frames); putInt("promptTokens", st.promptTokens); putDouble("promptMs", st.promptMs)
            putDouble("genMs", st.genMs); putDouble("ms", st.ms); putBoolean("stoppedByEos", st.stoppedByEos)
            putBoolean("stoppedByUser", st.stoppedByUser)
        }
    }

    /** The app is going away (or reloading): stop running calls and free everything. */
    override fun invalidate() {
        contexts.values.forEach { it.cancel() }
        contexts.values.forEach { runCatching { it.close() } }
        contexts.clear()
        models.values.forEach { runCatching { it.close() } }
        models.clear()
        worker.shutdown()
        super.invalidate()
    }
}

// ---------------------------------------------------------------- conversions

private fun ReadableMap.str(key: String): String? = if (hasKey(key) && !isNull(key)) getString(key) else null
private fun ReadableMap.int(key: String, default: Int = 0): Int = if (hasKey(key) && !isNull(key)) getDouble(key).toInt() else default
private fun ReadableMap.bool(key: String, default: Boolean): Boolean = if (hasKey(key) && !isNull(key)) getBoolean(key) else default
private fun ReadableMap.float(key: String, default: Float): Float = if (hasKey(key) && !isNull(key)) getDouble(key).toFloat() else default

private fun ReadableArray.toMessages(): List<Message> = List(size()) { i ->
    val m = getMap(i)!!
    val role = when (m.getString("role")) {
        "system" -> Role.SYSTEM
        "assistant" -> Role.ASSISTANT
        else -> Role.USER
    }
    Message(role, m.getString("content") ?: "")
}

private fun ReadableMap.toChatOptions() = ChatOptions(
    addAssistantPrefix = bool("addAssistantPrefix", true),
    enableThinking = bool("enableThinking", false),
    maxReasoningTokens = int("maxReasoningTokens"),
)

private fun ReadableMap.toSampling(): Sampling {
    val d = Sampling()
    return Sampling(
        temperature = float("temperature", d.temperature), topP = float("topP", d.topP), topK = int("topK", d.topK),
        minP = float("minP", d.minP), repeatPenalty = float("repeatPenalty", d.repeatPenalty), repeatLastN = int("repeatLastN", d.repeatLastN),
        seed = if (hasKey("seed") && !isNull("seed")) getDouble("seed").toLong().toInt() else d.seed, maxTokens = int("maxTokens", d.maxTokens),
    )
}

private fun ReadableMap.toEmbedOptions() = EmbedOptions(
    kind = when (str("kind")) {
        "query" -> EmbedKind.QUERY
        "document" -> EmbedKind.DOCUMENT
        else -> EmbedKind.RAW
    },
    title = str("title"),
    dimensions = int("dimensions"),
    normalize = bool("normalize", true),
)

private fun ReadableMap.toMedia(): Media {
    str("path")?.let { return Media.File(it) }
    val b64 = str("base64") ?: throw IllegalArgumentException("media needs a path or base64 bytes")
    return Media.Bytes(Base64.decode(b64, Base64.DEFAULT))
}

private fun GenStats.toMap(): WritableMap = Arguments.createMap().apply {
    putInt("promptTokens", promptTokens); putInt("cachedTokens", cachedTokens); putInt("generatedTokens", generatedTokens)
    putInt("reasoningTokens", reasoningTokens); putDouble("promptMs", promptMs); putDouble("genMs", genMs)
    putDouble("promptTps", promptTps); putDouble("genTps", genTps); putBoolean("stoppedByEos", stoppedByEos)
    putBoolean("stoppedByUser", stoppedByUser); putBoolean("reasoningTruncated", reasoningTruncated)
}

private fun Embedding.toMap(): WritableMap = Arguments.createMap().apply {
    putArray("vector", Arguments.createArray().apply { vector.forEach { pushDouble(it.toDouble()) } })
    putInt("tokens", stats.tokens); putInt("mediaTokens", stats.mediaTokens); putInt("dimensions", stats.dimensions)
    putBoolean("prefixed", stats.prefixed); putDouble("mediaMs", stats.mediaMs); putDouble("ms", stats.ms)
}

/** 16-bit mono PCM wav. */
private fun writeWav(file: File, samples: FloatArray, rate: Int) {
    file.parentFile?.mkdirs()
    val data = samples.size * 2
    val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray()).putInt(16)
        .putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
        .put("data".toByteArray()).putInt(data)
    for (s in samples) b.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
    FileOutputStream(file).use { it.write(b.array()) }
}
