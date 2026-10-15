package dev.kiln

// The JNI functions in src/main/cpp/kiln_jni.cpp. Strings go both ways as UTF-8 byte arrays;
// numbers come back in arrays the caller allocates. See that file for the order of each array.
internal object Native {
    init {
        System.loadLibrary("kiln_jni")
    }

    @JvmStatic external fun version(): ByteArray
    @JvmStatic external fun defaults(): DoubleArray
    @JvmStatic external fun setLogLevel(level: Int)
    @JvmStatic external fun probe(nums: LongArray): Array<ByteArray>

    @JvmStatic external fun modelLoad(path: ByteArray, backend: Int, nGpuLayers: Int, nThreads: Int, mmap: Boolean, mlock: Boolean): Long
    @JvmStatic external fun modelFree(model: Long)
    @JvmStatic external fun modelInfo(model: Long, nums: LongArray): Array<ByteArray>
    @JvmStatic external fun modelLoadProjector(model: Long, path: ByteArray, imageMaxTokens: Int): IntArray
    @JvmStatic external fun tokenize(model: Long, text: ByteArray, addSpecial: Boolean): IntArray
    @JvmStatic external fun detokenize(model: Long, tokens: IntArray): ByteArray
    @JvmStatic external fun applyChatTemplate(model: Long, roles: IntArray, contents: Array<ByteArray>, prefix: Boolean, thinking: Boolean): ByteArray

    @JvmStatic external fun contextCreate(model: Long, nCtx: Int, nBatch: Int, flashAttn: Boolean): Long
    @JvmStatic external fun contextFree(ctx: Long)
    @JvmStatic external fun contextReset(ctx: Long)
    @JvmStatic external fun cancel(ctx: Long)

    @JvmStatic external fun chat(
        ctx: Long, roles: IntArray, contents: Array<ByteArray>, prefix: Boolean, thinking: Boolean, maxReasoning: Int,
        samplingF: FloatArray, samplingI: LongArray, sink: PieceSink?, stats: DoubleArray,
    )

    @JvmStatic external fun generate(ctx: Long, prompt: ByteArray, samplingF: FloatArray, samplingI: LongArray, sink: PieceSink?, stats: DoubleArray)

    @JvmStatic external fun decide(
        ctx: Long, state: ByteArray?, id: ByteArray, type: Int, instructions: ByteArray,
        keys: Array<ByteArray?>, descriptions: Array<ByteArray?>, probs: FloatArray, answer: DoubleArray,
    )

    @JvmStatic external fun embed(
        ctx: Long, text: ByteArray, kind: Int, title: ByteArray?, dims: Int, normalize: Boolean, out: FloatArray, stats: DoubleArray,
    ): Int

    @JvmStatic external fun embedMedia(
        ctx: Long, text: ByteArray?, paths: Array<ByteArray?>, datas: Array<ByteArray?>, kind: Int, title: ByteArray?, dims: Int,
        normalize: Boolean, out: FloatArray, stats: DoubleArray,
    ): Int

    @JvmStatic external fun transcribe(
        ctx: Long, path: ByteArray?, data: ByteArray?, language: ByteArray?, prompt: ByteArray?, maxTokens: Int,
        sink: PieceSink?, stats: DoubleArray,
    ): ByteArray

    @JvmStatic external fun speak(
        ctx: Long, text: ByteArray, voicePath: ByteArray?, voiceData: ByteArray?, language: ByteArray?, maxFrames: Int, topK: Int,
        topP: Float, seed: Long, sink: AudioSink?, stats: DoubleArray,
    )
}

/** Called from native code for each token piece: the raw bytes, which may end mid-character. */
internal interface PieceSink {
    fun onPiece(channel: Int, bytes: ByteArray, tokenId: Int): Boolean
}

/** Called from native code with each piece of generated audio. */
internal interface AudioSink {
    fun onAudio(samples: FloatArray, sampleRate: Int): Boolean
}

internal fun String.utf8(): ByteArray {
    require(indexOf('\u0000') < 0) { "text must not contain NUL characters: kiln takes C strings" }
    return encodeToByteArray()
}
internal fun String?.utf8OrNull(): ByteArray? = this?.utf8()
internal fun ByteArray.str(): String = decodeToString()

/**
 * Collects bytes and hands out whole UTF-8 characters. A token can end in the middle of a
 * character ("é" is two bytes; some models split it over two tokens). Invalid bytes become U+FFFD.
 */
internal class Utf8Carry {
    private var held = ByteArray(0)

    fun push(bytes: ByteArray): String {
        val all = if (held.isEmpty()) bytes else held + bytes
        val cut = completeLength(all)
        held = all.copyOfRange(cut, all.size)
        return if (cut == 0) "" else all.decodeToString(0, cut)
    }

    fun finish(): String {
        val s = held.decodeToString()
        held = ByteArray(0)
        return s
    }

    // Length of the prefix that does not end in an unfinished character.
    private fun completeLength(a: ByteArray): Int {
        var i = a.size - 1
        var back = 0
        while (i >= 0 && back < 4) {
            val c = a[i].toInt() and 0xFF
            if (c and 0xC0 != 0x80) { // not a continuation byte: ASCII or the start of a character
                val need = when {
                    c < 0x80 -> 1
                    c in 0xC0..0xDF -> 2
                    c in 0xE0..0xEF -> 3
                    c in 0xF0..0xF7 -> 4
                    else -> 1
                }
                return if (a.size - i < need) i else a.size
            }
            i--
            back++
        }
        return a.size // only continuation bytes: invalid, let the decoder replace them
    }
}
