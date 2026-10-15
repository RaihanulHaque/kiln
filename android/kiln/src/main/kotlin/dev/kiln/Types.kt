package dev.kiln

// Kotlin versions of the structs in kiln.h. Default values are kiln's own (kiln.cpp, the
// *_default() functions); the host test checks a few of them against the library.

/** What went wrong, as kiln reports it (`kiln_status`). */
enum class Status(val code: Int) {
    INVALID_ARG(1), NO_BACKEND(2), MODEL_LOAD(3), OUT_OF_MEMORY(4), CONTEXT_FULL(5),
    /** Stopped by [KilnContext.cancel]. */
    CANCELLED(6),
    BACKEND(7), NOT_SUPPORTED(8), INTERNAL(99), UNKNOWN(-1);

    companion object {
        fun of(code: Int): Status = entries.firstOrNull { it.code == code } ?: UNKNOWN
    }
}

/** A failed kiln call: the status and kiln's message for it. */
class KilnException(val status: Status, message: String) : RuntimeException("$message ($status)") {
    // called from native code
    internal constructor(code: Int, message: ByteArray) : this(Status.of(code), message.str())
}

enum class Backend { AUTO, CPU, GPU, NPU }

data class DeviceInfo(
    val soc: String,
    val abi: String,
    val cpuCores: Int,
    /** Cores kiln treats as fast: all but the little cores. */
    val cpuBigCores: Int,
    /** Bit i set = cpu i is one of those cores; 0 = no affinity information. */
    val cpuFastMask: Long,
    val ramTotalBytes: Long,
    val ramAvailableBytes: Long,
    val hasFp16: Boolean,
    val hasDotprod: Boolean,
    val hasI8mm: Boolean,
    val hasSve: Boolean,
    val hasGpu: Boolean,
    val gpuName: String,
    val hasNpu: Boolean,
    val npuName: String,
)

data class ModelParams(
    val backend: Backend = Backend.AUTO,
    /** -1 = all layers if a GPU is chosen, 0 = none. */
    val nGpuLayers: Int = -1,
    /** 0 = from the device probe. */
    val nThreads: Int = 0,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
)

/** What actually happened when the model was loaded. */
data class ModelInfo(
    val backendUsed: Backend,
    /** e.g. "llama.cpp/cpu", "whisper.cpp/cpu" */
    val backendName: String,
    val arch: String,
    val quant: String,
    val nParams: Long,
    val nCtxTrain: Int,
    val nVocab: Int,
    val nGpuLayersUsed: Int,
    val modelBytes: Long,
    val supportsThinking: Boolean,
    /** Answers typed questions with [KilnContext.decide]; cannot write text. */
    val isDecisionModel: Boolean,
    /** Turns text into a vector with [KilnContext.embed]; cannot write text. */
    val isEmbeddingModel: Boolean,
    /** Embedding models: length of the vector; else 0. */
    val nEmbd: Int,
    /** Whisper or Parakeet: transcribes without a projector. False for Qwen3-ASR. */
    val isSpeechModel: Boolean,
)

data class ProjectorParams(
    /** Most tokens one image may take; a large photo is scaled down. 0 = 280, negative = the model's own limit. */
    val imageMaxTokens: Int = 0,
)

data class ProjectorInfo(
    val supportsImages: Boolean,
    val supportsAudio: Boolean,
    /** A text-to-speech projector: the model can [KilnContext.speak]. */
    val generatesAudio: Boolean,
    val sampleRate: Int,
)

data class ContextParams(
    /** 0 = the model's default, capped by RAM. */
    val nCtx: Int = 0,
    val nBatch: Int = 512,
    val flashAttn: Boolean = true,
)

enum class Role { SYSTEM, USER, ASSISTANT }

data class Message(val role: Role, val content: String) {
    companion object {
        fun system(content: String) = Message(Role.SYSTEM, content)
        fun user(content: String) = Message(Role.USER, content)
        fun assistant(content: String) = Message(Role.ASSISTANT, content)
    }
}

data class ChatOptions(
    /** Append the assistant turn opener (true for generation). */
    val addAssistantPrefix: Boolean = true,
    /** Reasoning models only. false = answer directly. */
    val enableThinking: Boolean = false,
    /** After this many reasoning tokens kiln closes the think block and the model answers. <= 0 = unlimited. */
    val maxReasoningTokens: Int = 0,
)

/** How the next token is picked. */
data class Sampling(
    /** <= 0 = greedy: always the most likely token. */
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    /** 0 = random. */
    val seed: Int = 0,
    /** <= 0 = until the model stops or the context is full. */
    val maxTokens: Int = 0,
) {
    companion object {
        /** Always the most likely token, no repeat penalty: the same prompt gives the same text. */
        fun greedy(maxTokens: Int = 0) = Sampling(temperature = 0f, repeatPenalty = 1f, maxTokens = maxTokens)
    }

    internal fun floats() = floatArrayOf(temperature, topP, minP, repeatPenalty)
    internal fun ints() = longArrayOf(topK.toLong(), repeatLastN.toLong(), seed.toLong() and 0xFFFFFFFFL, maxTokens.toLong())
}

enum class Channel {
    ANSWER,
    /** Inside the model's think block; the tags themselves are removed. */
    REASONING,
}

/** What [KilnContext.chatFlow] emits: pieces of text, then the stats. */
sealed interface ChatEvent

/** A piece of streamed text, always whole characters. */
data class Piece(val channel: Channel, val text: String, val tokenId: Int) : ChatEvent

data class GenStats(
    val promptTokens: Int,
    /** Of the prompt tokens, how many were already in the KV cache (earlier turns). */
    val cachedTokens: Int,
    val generatedTokens: Int,
    val reasoningTokens: Int,
    val promptMs: Double,
    val genMs: Double,
    val promptTps: Double,
    val genTps: Double,
    val stoppedByEos: Boolean,
    val stoppedByUser: Boolean,
    val reasoningTruncated: Boolean,
) : ChatEvent {
    internal companion object {
        const val SIZE = 11
        fun of(s: DoubleArray) = GenStats(
            s[0].toInt(), s[1].toInt(), s[2].toInt(), s[3].toInt(), s[4], s[5], s[6], s[7], s[8] != 0.0, s[9] != 0.0, s[10] != 0.0,
        )
    }
}

data class Reply(val answer: String, val reasoning: String, val stats: GenStats)

enum class QuestionType {
    /** Yes or no. */
    YES_NO,
    /** Pick one of the options. */
    CHOICE,
    /** Rate on a scale; the options are the levels, lowest first (2 to 10). */
    SCORE,
}

data class DecisionOption(
    /** Choice: the label returned. Yes / no: "true" or "false". Score: ignored. */
    val key: String? = null,
    val description: String? = null,
)

data class Question(
    /** Short name, e.g. "wants_refund". */
    val id: String,
    val type: QuestionType,
    /** The question itself. */
    val instructions: String,
    val options: List<DecisionOption> = emptyList(),
)

data class Decision(
    /** One probability per option. Yes / no: [no, yes]. */
    val probs: FloatArray,
    /** Index of the most probable entry. */
    val best: Int,
    /** Yes / no: P(yes). Score: the expected level, 0 .. n-1. Choice: P(best). */
    val value: Float,
    val confidence: Float,
    val promptTokens: Int,
    val ms: Double,
)

enum class EmbedKind {
    /** The text exactly as given. */
    RAW,
    /** A search query: kiln adds the model's query prefix. */
    QUERY,
    /** A text to be searched: kiln adds the model's document prefix. */
    DOCUMENT,
}

data class EmbedOptions(
    val kind: EmbedKind = EmbedKind.RAW,
    /** Documents only, optional. */
    val title: String? = null,
    /** Keep only the first N numbers. <= 0 = the full vector. */
    val dimensions: Int = 0,
    /** Scale the vector to length 1, so similarity is the dot product. */
    val normalize: Boolean = true,
) {
    companion object {
        val QUERY = EmbedOptions(kind = EmbedKind.QUERY)
        val DOCUMENT = EmbedOptions(kind = EmbedKind.DOCUMENT)
    }
}

data class EmbedStats(
    val tokens: Int,
    val mediaTokens: Int,
    val dimensions: Int,
    val prefixed: Boolean,
    val mediaMs: Double,
    val ms: Double,
) {
    internal companion object {
        const val SIZE = 6
        fun of(s: DoubleArray) = EmbedStats(s[0].toInt(), s[1].toInt(), s[2].toInt(), s[3] != 0.0, s[4], s[5])
    }
}

class Embedding(val vector: FloatArray, val stats: EmbedStats)

/** Where a media item goes inside the text: "A red shoe <__media__> seen from the side". */
const val MEDIA_MARKER = "<__media__>"

/** One image or audio clip: a file, or the bytes of such a file. The kind is read from the content. */
sealed class Media {
    class File(val path: String) : Media()
    class Bytes(val data: ByteArray) : Media()

    internal fun pathBytes(): ByteArray? = (this as? File)?.path?.utf8()
    internal fun dataBytes(): ByteArray? = (this as? Bytes)?.data
}

data class TranscribeOptions(
    /** The language spoken, e.g. "English", when known. null = the model works it out. */
    val language: String? = null,
    /** Qwen3-ASR: replaces the instruction. Whisper: text the words may follow on from. Parakeet: ignored. */
    val prompt: String? = null,
    /** <= 0 = until the model stops or the context is full. */
    val maxTokens: Int = 0,
)

data class TranscribeStats(
    val audioSeconds: Double,
    val audioTokens: Int,
    val promptTokens: Int,
    val generatedTokens: Int,
    val encodeMs: Double,
    val promptMs: Double,
    val genMs: Double,
    /** The whole call. Faster than real time when ms < audioSeconds * 1000. */
    val ms: Double,
    /** The language as the model named it, e.g. "English"; empty if it did not say. */
    val language: String,
    val stoppedByEos: Boolean,
    val stoppedByUser: Boolean,
) {
    internal companion object {
        const val SIZE = 10
        fun of(s: DoubleArray, language: String) =
            TranscribeStats(s[0], s[1].toInt(), s[2].toInt(), s[3].toInt(), s[4], s[5], s[6], s[7], language, s[8] != 0.0, s[9] != 0.0)
    }
}

data class Transcript(val text: String, val stats: TranscribeStats)

data class SpeakOptions(
    /** A few seconds of someone speaking; the output imitates that voice. Pocket TTS needs one. */
    val voice: Media? = null,
    /** Qwen3-TTS: "en", "zh", "de", ... (null = "en"). Pocket TTS: ignored. */
    val language: String? = null,
    /** Longest output in frames (80 ms each for Pocket TTS); <= 0 = 512. */
    val maxFrames: Int = 0,
    /** <= 0 = 40. */
    val topK: Int = 0,
    /** <= 0 = 0.95. */
    val topP: Float = 0f,
    /** null = random; a value makes the same text give the same audio. */
    val seed: Int? = null,
)

data class SpeakStats(
    val audioSeconds: Double,
    val sampleRate: Int,
    val frames: Int,
    val promptTokens: Int,
    val promptMs: Double,
    val genMs: Double,
    val ms: Double,
    val stoppedByEos: Boolean,
    val stoppedByUser: Boolean,
) {
    internal companion object {
        const val SIZE = 9
        fun of(s: DoubleArray) = SpeakStats(s[0], s[1].toInt(), s[2].toInt(), s[3].toInt(), s[4], s[5], s[6], s[7] != 0.0, s[8] != 0.0)
    }
}

/** Generated audio: mono floats in -1..1. */
class Speech(val samples: FloatArray, val sampleRate: Int, val stats: SpeakStats)

enum class LogLevel { ERROR, WARN, INFO, DEBUG }
