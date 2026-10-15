package dev.kiln

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * One conversation and its KV cache, made with [KilnModel.newContext].
 *
 * Every call blocks until it is done: run it on a background thread, or use [chatFlow], which
 * does that for you. One call runs at a time; [cancel] works from any thread. Callbacks run on
 * the calling thread. Close the context when done (`use { }`).
 */
class KilnContext internal constructor(private val model: KilnModel, handle: Long) : AutoCloseable {
    private var handle: Long = handle
    private val handleLock = Any()          // guards handle against close() while cancel() reads it
    private val callLock = ReentrantLock()  // one call at a time; close() waits for it

    val modelInfo: ModelInfo get() = model.info

    private fun noReentry() =
        check(!callLock.isHeldByCurrentThread) { "a callback cannot call its own context; return false to stop" }

    private inline fun <T> call(block: (Long) -> T): T = noReentry().let { callLock.withLock {
        val h = synchronized(handleLock) { handle }
        check(h != 0L) { "the context is closed" }
        block(h)
    } }

    /** Stops the running call, from any thread: it throws [KilnException] with [Status.CANCELLED]. */
    fun cancel() {
        synchronized(handleLock) { if (handle != 0L) Native.cancel(handle) }
    }

    /** Forgets the conversation (clears the KV cache). */
    fun reset() = call { Native.contextReset(it) }

    /**
     * Answers the last message of [messages], streaming the reply to [onPiece]. Pass the whole
     * conversation every turn; kiln reuses what is already in the cache. Return false from
     * [onPiece] to stop (the call then returns with `stoppedByUser`).
     */
    @JvmOverloads
    fun chat(
        messages: List<Message>,
        options: ChatOptions = ChatOptions(),
        sampling: Sampling = Sampling(),
        onPiece: (Piece) -> Boolean = { true },
    ): GenStats = call { h ->
        val (roles, contents) = messages.toNative()
        val stats = DoubleArray(GenStats.SIZE)
        val sink = TextSink(onPiece)
        Native.chat(
            h, roles, contents, options.addAssistantPrefix, options.enableThinking, options.maxReasoningTokens,
            sampling.floats(), sampling.ints(), sink, stats,
        )
        sink.finish()
        GenStats.of(stats)
    }

    /** [chat], collecting the whole reply. */
    @JvmOverloads
    fun chatCollect(messages: List<Message>, options: ChatOptions = ChatOptions(), sampling: Sampling = Sampling()): Reply {
        val answer = StringBuilder()
        val reasoning = StringBuilder()
        val stats = chat(messages, options, sampling) {
            (if (it.channel == Channel.ANSWER) answer else reasoning).append(it.text)
            true
        }
        return Reply(answer.toString(), reasoning.toString(), stats)
    }

    /**
     * [chat] as a Flow: [Piece]s as they are written, then the [GenStats]. The work runs on
     * [Dispatchers.IO]; cancelling the collector cancels the generation.
     */
    @JvmOverloads
    fun chatFlow(messages: List<Message>, options: ChatOptions = ChatOptions(), sampling: Sampling = Sampling()): Flow<ChatEvent> =
        callbackFlow {
            val worker = launch(Dispatchers.IO) {
                try {
                    val stats = chat(messages, options, sampling) { trySendBlocking(it).isSuccess }
                    send(stats)
                    channel.close()
                } catch (e: Throwable) {
                    channel.close(e)
                }
            }
            awaitClose { if (worker.isActive) cancel() }
        }

    /** Continues a raw prompt that is already in the model's template. */
    @JvmOverloads
    fun generate(prompt: String, sampling: Sampling = Sampling(), onPiece: (Piece) -> Boolean = { true }): GenStats = call { h ->
        val stats = DoubleArray(GenStats.SIZE)
        val sink = TextSink(onPiece)
        Native.generate(h, prompt.utf8(), sampling.floats(), sampling.ints(), sink, stats)
        sink.finish()
        GenStats.of(stats)
    }

    /** Decision models: the probability of each answer to one typed question about [state]. */
    fun decide(state: String?, question: Question): Decision = call { h ->
        val probs = FloatArray(maxOf(question.options.size, 2))
        val a = DoubleArray(6)
        Native.decide(
            h, state.utf8OrNull(), question.id.utf8(), question.type.ordinal, question.instructions.utf8(),
            Array(question.options.size) { question.options[it].key.utf8OrNull() },
            Array(question.options.size) { question.options[it].description.utf8OrNull() },
            probs, a,
        )
        Decision(probs.copyOf(a[0].toInt()), a[1].toInt(), a[2].toFloat(), a[3].toFloat(), a[4].toInt(), a[5])
    }

    /** Embedding models: the vector for [text]. */
    @JvmOverloads
    fun embed(text: String, options: EmbedOptions = EmbedOptions()): Embedding = call { h ->
        val out = FloatArray(outLen(options))
        val stats = DoubleArray(EmbedStats.SIZE)
        val n = Native.embed(h, text.utf8(), options.kind.ordinal, options.title.utf8OrNull(), options.dimensions, options.normalize, out, stats)
        Embedding(out.copyOf(n), EmbedStats.of(stats))
    }

    /**
     * Embedding models with a projector: one vector for a text with images and / or audio, or
     * for media alone (text null). Each [MEDIA_MARKER] in the text is replaced by the next item.
     */
    @JvmOverloads
    fun embedMedia(text: String?, media: List<Media>, options: EmbedOptions = EmbedOptions()): Embedding = call { h ->
        val out = FloatArray(outLen(options))
        val stats = DoubleArray(EmbedStats.SIZE)
        val n = Native.embedMedia(
            h, text.utf8OrNull(), Array(media.size) { media[it].pathBytes() }, Array(media.size) { media[it].dataBytes() },
            options.kind.ordinal, options.title.utf8OrNull(), options.dimensions, options.normalize, out, stats,
        )
        Embedding(out.copyOf(n), EmbedStats.of(stats))
    }

    private fun outLen(options: EmbedOptions): Int {
        val full = maxOf(model.info.nEmbd, 1)
        return if (options.dimensions > 0) minOf(options.dimensions, full) else full
    }

    /** Speech models: writes down one audio clip, streaming the words to [onText]. Each call clears the context. */
    @JvmOverloads
    fun transcribe(audio: Media, options: TranscribeOptions = TranscribeOptions(), onText: (String) -> Boolean = { true }): TranscribeStats =
        call { h ->
            val stats = DoubleArray(TranscribeStats.SIZE)
            val sink = TextSink { onText(it.text) }
            val lang = Native.transcribe(
                h, audio.pathBytes(), audio.dataBytes(), options.language.utf8OrNull(), options.prompt.utf8OrNull(),
                options.maxTokens, sink, stats,
            )
            sink.finish()
            TranscribeStats.of(stats, lang.str())
        }

    /** [transcribe], collecting the whole text. */
    @JvmOverloads
    fun transcribeCollect(audio: Media, options: TranscribeOptions = TranscribeOptions()): Transcript {
        val text = StringBuilder()
        val stats = transcribe(audio, options) {
            text.append(it)
            true
        }
        return Transcript(text.toString(), stats)
    }

    /**
     * Text-to-speech models (projector loaded): speaks [text], handing over audio about once a
     * second as mono floats at the given sample rate. Return false to stop. The conversation is kept.
     */
    @JvmOverloads
    fun speak(text: String, options: SpeakOptions = SpeakOptions(), onAudio: (FloatArray, Int) -> Boolean = { _, _ -> true }): SpeakStats =
        call { h ->
            val stats = DoubleArray(SpeakStats.SIZE)
            val sink = PcmSink(onAudio)
            Native.speak(
                h, text.utf8(), options.voice?.pathBytes(), options.voice?.dataBytes(), options.language.utf8OrNull(), options.maxFrames,
                options.topK, options.topP, options.seed?.toLong()?.and(0xFFFFFFFFL) ?: -1L, sink, stats,
            )
            SpeakStats.of(stats)
        }

    /** [speak], collecting all the audio. */
    @JvmOverloads
    fun speakCollect(text: String, options: SpeakOptions = SpeakOptions()): Speech {
        val chunks = ArrayList<FloatArray>()
        var rate = 0
        val stats = speak(text, options) { pcm, r ->
            chunks.add(pcm)
            rate = r
            true
        }
        val all = FloatArray(chunks.sumOf { it.size })
        var at = 0
        for (c in chunks) {
            c.copyInto(all, at)
            at += c.size
        }
        return Speech(all, if (rate > 0) rate else stats.sampleRate, stats)
    }

    /** Stops a running call, waits for it, and frees the context. */
    override fun close() {
        noReentry()
        cancel()
        callLock.withLock {
            synchronized(handleLock) {
                val h = handle
                if (h == 0L) return
                handle = 0L
                Native.contextFree(h)
            }
            model.contextClosed()
        }
    }
}

/** Turns the raw pieces from native code into [Piece]s of whole characters. */
private class TextSink(private val onPiece: (Piece) -> Boolean) : PieceSink {
    private val carry = arrayOf(Utf8Carry(), Utf8Carry())
    private val lastToken = IntArray(2)
    private var stopped = false

    override fun onPiece(channel: Int, bytes: ByteArray, tokenId: Int): Boolean {
        val i = if (channel == 1) 1 else 0
        lastToken[i] = tokenId
        val text = carry[i].push(bytes)
        if (text.isEmpty()) return true
        if (!onPiece(Piece(Channel.entries[i], text, tokenId))) stopped = true
        return !stopped
    }

    /** Hands over bytes still held when generation ends. */
    fun finish() {
        if (stopped) return
        for (i in intArrayOf(1, 0)) {
            val text = carry[i].finish()
            if (text.isNotEmpty()) onPiece(Piece(Channel.entries[i], text, lastToken[i]))
        }
    }
}

private class PcmSink(private val onAudio: (FloatArray, Int) -> Boolean) : AudioSink {
    override fun onAudio(samples: FloatArray, sampleRate: Int): Boolean = onAudio.invoke(samples, sampleRate)
}
