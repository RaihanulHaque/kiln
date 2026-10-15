package dev.kiln

import java.util.concurrent.atomic.AtomicInteger

/**
 * A loaded model file (GGUF, or a whisper.cpp `.bin` for Whisper and Parakeet).
 *
 * One model can serve several [KilnContext]s, also from different threads. Close it when done
 * (`use { }`), after closing its contexts: native memory is not freed by the garbage collector.
 */
class KilnModel private constructor(handle: Long) : AutoCloseable {
    @Volatile private var handle: Long = handle
    private val openContexts = AtomicInteger(0)

    /** What actually happened at load: backend, architecture, quantisation, and what kind of model this is. */
    val info: ModelInfo = run {
        val n = LongArray(11)
        val s = Native.modelInfo(handle, n)
        ModelInfo(
            backendUsed = Backend.entries.getOrElse(n[0].toInt()) { Backend.AUTO }, backendName = s[0].str(), arch = s[1].str(),
            quant = s[2].str(), nParams = n[1], nCtxTrain = n[2].toInt(), nVocab = n[3].toInt(), nGpuLayersUsed = n[4].toInt(),
            modelBytes = n[5], supportsThinking = n[6] != 0L, isDecisionModel = n[7] != 0L, isEmbeddingModel = n[8] != 0L,
            nEmbd = n[9].toInt(), isSpeechModel = n[10] != 0L,
        )
    }

    companion object {
        /** Loads a model. Blocking: run it off the main thread. */
        @JvmStatic
        @JvmOverloads
        fun load(path: String, params: ModelParams = ModelParams()): KilnModel {
            val h = Native.modelLoad(path.utf8(), params.backend.ordinal, params.nGpuLayers, params.nThreads, params.useMmap, params.useMlock)
            return try {
                KilnModel(h)
            } catch (e: Throwable) {
                Native.modelFree(h)
                throw e
            }
        }
    }

    /**
     * Loads the image / audio encoders that belong to this model (its "mmproj" GGUF): an
     * embedding model that takes media, Qwen3-ASR, or a text-to-speech model. Call once, before
     * creating contexts. The encoders are freed with the model.
     */
    @JvmOverloads
    fun loadProjector(path: String, params: ProjectorParams = ProjectorParams()): ProjectorInfo {
        check(openContexts.get() == 0) { "load the projector before creating contexts" }
        val r = Native.modelLoadProjector(live(), path.utf8(), params.imageMaxTokens)
        return ProjectorInfo(r[0] != 0, r[1] != 0, r[2] != 0, r[3])
    }

    /** A new context: one conversation with its own KV cache. */
    @JvmOverloads
    fun newContext(params: ContextParams = ContextParams()): KilnContext {
        openContexts.incrementAndGet()
        try {
            return KilnContext(this, Native.contextCreate(live(), params.nCtx, params.nBatch, params.flashAttn))
        } catch (e: Throwable) {
            openContexts.decrementAndGet()
            throw e
        }
    }

    @JvmOverloads
    fun tokenize(text: String, addSpecial: Boolean = false): IntArray = Native.tokenize(live(), text.utf8(), addSpecial)

    /** Turns tokens back into text. Bytes that are not valid UTF-8 become U+FFFD. */
    fun detokenize(tokens: IntArray): String = Native.detokenize(live(), tokens).str()

    /** The prompt the model's own chat template makes of [messages]. For generation use [KilnContext.chat]. */
    @JvmOverloads
    fun applyChatTemplate(messages: List<Message>, options: ChatOptions = ChatOptions()): String {
        val (roles, contents) = messages.toNative()
        return Native.applyChatTemplate(live(), roles, contents, options.addAssistantPrefix, options.enableThinking).str()
    }

    /** Frees the model. Close its contexts first. */
    override fun close() {
        synchronized(this) {
            val h = handle
            if (h == 0L) return
            check(openContexts.get() == 0) { "close the model's contexts first (${openContexts.get()} open)" }
            handle = 0L
            Native.modelFree(h)
        }
    }

    internal fun live(): Long = handle.also { check(it != 0L) { "the model is closed" } }

    internal fun contextClosed() {
        openContexts.decrementAndGet()
    }
}

internal fun List<Message>.toNative(): Pair<IntArray, Array<ByteArray>> =
    IntArray(size) { this[it].role.ordinal } to Array(size) { this[it].content.utf8() }
