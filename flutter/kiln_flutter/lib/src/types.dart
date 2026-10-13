import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'kiln_bindings_generated.dart' as b;

String _str(Array<Char> a, int cap) {
  final bytes = <int>[];
  for (var i = 0; i < cap; i++) {
    final c = a[i];
    if (c == 0) break;
    bytes.add(c & 0xff);
  }
  return String.fromCharCodes(bytes);
}

class KilnException implements Exception {
  KilnException(this.status, this.message);
  final b.kiln_status status;
  final String message;
  @override
  String toString() => 'KilnException(${status.name}): $message';
}

enum KilnBackend { auto, cpu, gpu, npu }

enum KilnChannel { answer, reasoning }

class KilnDeviceInfo {
  KilnDeviceInfo({
    required this.soc,
    required this.abi,
    required this.cpuCores,
    required this.cpuFastCores,
    required this.cpuFastMask,
    required this.ramTotalBytes,
    required this.ramAvailableBytes,
    required this.hasFp16,
    required this.hasDotprod,
    required this.hasI8mm,
    required this.hasSve,
    required this.hasGpu,
    required this.gpuName,
    required this.hasNpu,
    required this.npuName,
  });

  factory KilnDeviceInfo.fromNative(b.kiln_device_info d) => KilnDeviceInfo(
        soc: _str(d.soc, 64),
        abi: _str(d.abi, 16),
        cpuCores: d.cpu_cores,
        cpuFastCores: d.cpu_big_cores,
        cpuFastMask: d.cpu_fast_mask,
        ramTotalBytes: d.ram_total_bytes,
        ramAvailableBytes: d.ram_available_bytes,
        hasFp16: d.has_fp16,
        hasDotprod: d.has_dotprod,
        hasI8mm: d.has_i8mm,
        hasSve: d.has_sve,
        hasGpu: d.has_gpu,
        gpuName: _str(d.gpu_name, 64),
        hasNpu: d.has_npu,
        npuName: _str(d.npu_name, 64),
      );

  final String soc, abi, gpuName, npuName;
  final int cpuCores, cpuFastCores, cpuFastMask, ramTotalBytes, ramAvailableBytes;
  final bool hasFp16, hasDotprod, hasI8mm, hasSve, hasGpu, hasNpu;

  Map<String, Object> toMap() => {
        'soc': soc, 'abi': abi, 'cpuCores': cpuCores, 'cpuFastCores': cpuFastCores,
        'cpuFastMask': cpuFastMask, 'ramTotalBytes': ramTotalBytes,
        'ramAvailableBytes': ramAvailableBytes, 'hasFp16': hasFp16, 'hasDotprod': hasDotprod,
        'hasI8mm': hasI8mm, 'hasSve': hasSve, 'hasGpu': hasGpu, 'gpuName': gpuName,
        'hasNpu': hasNpu, 'npuName': npuName,
      };
  factory KilnDeviceInfo.fromMap(Map m) => KilnDeviceInfo(
        soc: m['soc'], abi: m['abi'], cpuCores: m['cpuCores'], cpuFastCores: m['cpuFastCores'],
        cpuFastMask: m['cpuFastMask'], ramTotalBytes: m['ramTotalBytes'],
        ramAvailableBytes: m['ramAvailableBytes'], hasFp16: m['hasFp16'], hasDotprod: m['hasDotprod'],
        hasI8mm: m['hasI8mm'], hasSve: m['hasSve'], hasGpu: m['hasGpu'], gpuName: m['gpuName'],
        hasNpu: m['hasNpu'], npuName: m['npuName'],
      );

  @override
  String toString() =>
      '$soc ($abi), $cpuCores cores ($cpuFastCores fast), ${(ramTotalBytes / 1e9).toStringAsFixed(1)} GB RAM'
      '${hasGpu ? ', GPU: $gpuName' : ''}${hasNpu ? ', NPU: $npuName' : ''}';
}

/// What actually happened when the model loaded. The truth-telling struct.
class KilnModelInfo {
  KilnModelInfo({
    required this.backend,
    required this.backendName,
    required this.arch,
    required this.quant,
    required this.nParams,
    required this.nCtxTrain,
    required this.nVocab,
    required this.gpuLayersUsed,
    required this.modelBytes,
    required this.supportsThinking,
    this.isDecisionModel = false,
    this.isEmbeddingModel = false,
    this.nEmbd = 0,
    this.isSpeechModel = false,
  });

  factory KilnModelInfo.fromNative(b.kiln_model_info i) => KilnModelInfo(
        backend: KilnBackend.values[i.backend_usedAsInt],
        backendName: _str(i.backend_name, 64),
        arch: _str(i.arch, 32),
        quant: _str(i.quant, 16),
        nParams: i.n_params,
        nCtxTrain: i.n_ctx_train,
        nVocab: i.n_vocab,
        gpuLayersUsed: i.n_gpu_layers_used,
        modelBytes: i.model_bytes,
        supportsThinking: i.supports_thinking,
        isDecisionModel: i.is_decision_model,
        isEmbeddingModel: i.is_embedding_model,
        nEmbd: i.n_embd,
        isSpeechModel: i.is_speech_model,
      );

  final KilnBackend backend;
  final String backendName, arch, quant;
  final int nParams, nCtxTrain, nVocab, gpuLayersUsed, modelBytes;
  final bool supportsThinking;

  /// The model answers typed questions with [KilnContext.decide] and cannot generate text.
  final bool isDecisionModel;

  /// The model turns text (and, with a projector, images and audio) into vectors with
  /// [KilnContext.embed] and cannot generate text.
  final bool isEmbeddingModel;

  /// Embedding models: how many numbers a full vector has. 0 for other models.
  final int nEmbd;

  /// A Whisper or Parakeet file: writes down audio with [KilnContext.transcribe], needs no
  /// projector, and cannot generate text. False for Qwen3-ASR, which is a chat-style model
  /// that transcribes once its projector is loaded.
  final bool isSpeechModel;

  Map<String, Object> toMap() => {
        'backend': backend.index, 'backendName': backendName, 'arch': arch, 'quant': quant,
        'nParams': nParams, 'nCtxTrain': nCtxTrain, 'nVocab': nVocab,
        'gpuLayersUsed': gpuLayersUsed, 'modelBytes': modelBytes, 'supportsThinking': supportsThinking,
        'isDecisionModel': isDecisionModel, 'isEmbeddingModel': isEmbeddingModel, 'nEmbd': nEmbd,
        'isSpeechModel': isSpeechModel,
      };
  factory KilnModelInfo.fromMap(Map m) => KilnModelInfo(
        backend: KilnBackend.values[m['backend']], backendName: m['backendName'], arch: m['arch'],
        quant: m['quant'], nParams: m['nParams'], nCtxTrain: m['nCtxTrain'], nVocab: m['nVocab'],
        gpuLayersUsed: m['gpuLayersUsed'], modelBytes: m['modelBytes'], supportsThinking: m['supportsThinking'],
        isDecisionModel: m['isDecisionModel'] ?? false,
        isEmbeddingModel: m['isEmbeddingModel'] ?? false, nEmbd: m['nEmbd'] ?? 0,
        isSpeechModel: m['isSpeechModel'] ?? false,
      );

  @override
  String toString() =>
      '$arch $quant, ${(nParams / 1e9).toStringAsFixed(2)}B params, '
      '${(modelBytes / 1e9).toStringAsFixed(2)} GB, via $backendName'
      '${supportsThinking ? ', thinking' : ''}${isDecisionModel ? ', decision model' : ''}'
      '${isEmbeddingModel ? ', embedding model ($nEmbd numbers)' : ''}'
      '${isSpeechModel ? ', speech model' : ''}';
}

class KilnModelParams {
  const KilnModelParams({
    this.backend = KilnBackend.auto,
    this.gpuLayers = -1,
    this.threads = 0,
    this.useMmap = true,
    this.useMlock = false,
  });
  final KilnBackend backend;
  final int gpuLayers, threads;
  final bool useMmap, useMlock;
  Map<String, Object> toMap() =>
      {'backend': backend.index, 'gpuLayers': gpuLayers, 'threads': threads, 'useMmap': useMmap, 'useMlock': useMlock};
  factory KilnModelParams.fromMap(Map m) => KilnModelParams(
      backend: KilnBackend.values[m['backend']], gpuLayers: m['gpuLayers'], threads: m['threads'],
      useMmap: m['useMmap'], useMlock: m['useMlock']);
}

class KilnContextParams {
  const KilnContextParams({this.nCtx = 0, this.nBatch = 512, this.flashAttn = true});
  final int nCtx, nBatch;
  final bool flashAttn;
  Map<String, Object> toMap() => {'nCtx': nCtx, 'nBatch': nBatch, 'flashAttn': flashAttn};
  factory KilnContextParams.fromMap(Map m) =>
      KilnContextParams(nCtx: m['nCtx'], nBatch: m['nBatch'], flashAttn: m['flashAttn']);
}

class KilnChatMessage {
  const KilnChatMessage(this.role, this.content);
  const KilnChatMessage.system(String c) : this('system', c);
  const KilnChatMessage.user(String c) : this('user', c);
  const KilnChatMessage.assistant(String c) : this('assistant', c);
  final String role, content;
  Map<String, Object> toMap() => {'role': role, 'content': content};
  factory KilnChatMessage.fromMap(Map m) => KilnChatMessage(m['role'], m['content']);
}

class KilnSamplingParams {
  const KilnSamplingParams({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.minP = 0.05,
    this.repeatPenalty = 1.1,
    this.repeatLastN = 64,
    this.seed = 0,
    this.maxTokens = 0,
  });
  const KilnSamplingParams.greedy({int maxTokens = 0})
      : this(temperature: 0, maxTokens: maxTokens);
  final double temperature, topP, minP, repeatPenalty;
  final int topK, repeatLastN, seed, maxTokens;
  Map<String, Object> toMap() => {
        'temperature': temperature, 'topP': topP, 'topK': topK, 'minP': minP,
        'repeatPenalty': repeatPenalty, 'repeatLastN': repeatLastN, 'seed': seed, 'maxTokens': maxTokens,
      };
  factory KilnSamplingParams.fromMap(Map m) => KilnSamplingParams(
        temperature: m['temperature'], topP: m['topP'], topK: m['topK'], minP: m['minP'],
        repeatPenalty: m['repeatPenalty'], repeatLastN: m['repeatLastN'], seed: m['seed'], maxTokens: m['maxTokens'],
      );
}

class KilnGenStats {
  KilnGenStats({
    required this.promptTokens,
    required this.cachedTokens,
    required this.generatedTokens,
    required this.reasoningTokens,
    required this.promptMs,
    required this.genMs,
    required this.promptTps,
    required this.genTps,
    required this.stoppedByEos,
    required this.stoppedByUser,
    this.reasoningTruncated = false,
  });
  factory KilnGenStats.fromNative(b.kiln_gen_stats s) => KilnGenStats(
        promptTokens: s.prompt_tokens,
        cachedTokens: s.cached_tokens,
        generatedTokens: s.generated_tokens,
        reasoningTokens: s.reasoning_tokens,
        promptMs: s.prompt_ms,
        genMs: s.gen_ms,
        promptTps: s.prompt_tps,
        genTps: s.gen_tps,
        stoppedByEos: s.stopped_by_eos,
        stoppedByUser: s.stopped_by_user,
        reasoningTruncated: s.reasoning_truncated,
      );
  final int promptTokens, cachedTokens, generatedTokens, reasoningTokens;
  final double promptMs, genMs, promptTps, genTps;
  final bool stoppedByEos, stoppedByUser;

  /// The thinking budget (`maxReasoningTokens`) was reached and kiln closed the think block.
  final bool reasoningTruncated;

  Map<String, Object> toMap() => {
        'promptTokens': promptTokens, 'cachedTokens': cachedTokens, 'generatedTokens': generatedTokens,
        'reasoningTokens': reasoningTokens, 'promptMs': promptMs, 'genMs': genMs,
        'promptTps': promptTps, 'genTps': genTps, 'stoppedByEos': stoppedByEos, 'stoppedByUser': stoppedByUser,
        'reasoningTruncated': reasoningTruncated,
      };
  factory KilnGenStats.fromMap(Map m) => KilnGenStats(
        promptTokens: m['promptTokens'], cachedTokens: m['cachedTokens'], generatedTokens: m['generatedTokens'],
        reasoningTokens: m['reasoningTokens'], promptMs: m['promptMs'], genMs: m['genMs'],
        promptTps: m['promptTps'], genTps: m['genTps'], stoppedByEos: m['stoppedByEos'], stoppedByUser: m['stoppedByUser'],
        reasoningTruncated: m['reasoningTruncated'] ?? false,
      );

  @override
  String toString() =>
      'prompt $promptTokens tok ($cachedTokens cached) ${promptMs.round()} ms, '
      'gen $generatedTokens tok ($reasoningTokens reasoning${reasoningTruncated ? ', cut at budget' : ''}) ${genMs.round()} ms, '
      '${genTps.toStringAsFixed(1)} tok/s'
      '${stoppedByEos ? ' [eos]' : stoppedByUser ? ' [stopped]' : ''}';
}

enum KilnQuestionType { yesNo, choice, score }

/// One option of a [KilnQuestion].
class KilnOption {
  const KilnOption(this.key, [this.description]);

  /// The label of the option, e.g. "billing".
  final String key;

  /// What the option means. Helps the model; may be left out.
  final String? description;

  Map<String, Object?> toMap() => {'key': key, 'description': description};
  factory KilnOption.fromMap(Map m) => KilnOption(m['key'], m['description']);
}

/// A typed question for a decision model. See [KilnContext.decide].
class KilnQuestion {
  const KilnQuestion._(this.type, this.instructions, this.id, this.options);

  /// A yes / no question. [yes] and [no] optionally describe what each answer means.
  factory KilnQuestion.yesNo(String instructions, {String id = 'question', String? yes, String? no}) =>
      KilnQuestion._(KilnQuestionType.yesNo, instructions, id, [
        if (yes != null) KilnOption('true', yes),
        if (no != null) KilnOption('false', no),
      ]);

  /// Pick one of [options] (at least two).
  const KilnQuestion.choice(String instructions, List<KilnOption> options, {String id = 'question'})
      : this._(KilnQuestionType.choice, instructions, id, options);

  /// Rate on a scale. [levels] describe each level, lowest first (2 to 10 of them).
  factory KilnQuestion.score(String instructions, List<String> levels, {String id = 'question'}) =>
      KilnQuestion._(KilnQuestionType.score, instructions, id,
          [for (var i = 0; i < levels.length; i++) KilnOption('$i', levels[i])]);

  final KilnQuestionType type;

  /// The question itself.
  final String instructions;

  /// A short name for the question, e.g. "wants_refund".
  final String id;
  final List<KilnOption> options;

  Map<String, Object?> toMap() => {
        'type': type.index, 'instructions': instructions, 'id': id,
        'options': [for (final o in options) o.toMap()],
      };
  factory KilnQuestion.fromMap(Map m) => KilnQuestion._(KilnQuestionType.values[m['type']],
      m['instructions'], m['id'], [for (final o in m['options'] as List) KilnOption.fromMap(o as Map)]);
}

/// The answer of a decision model to one [KilnQuestion].
class KilnDecision {
  KilnDecision({
    required this.type,
    required this.probabilities,
    required this.keys,
    required this.best,
    required this.value,
    required this.confidence,
    required this.promptTokens,
    required this.ms,
  });

  final KilnQuestionType type;

  /// One probability per option, in the order the options were given; they sum to 1.
  /// For a yes / no question there are two: `[0]` is no, `[1]` is yes.
  final List<double> probabilities;

  /// The label of each entry of [probabilities]. Yes / no: `['no', 'yes']`.
  /// Score: the level descriptions.
  final List<String> keys;

  /// Index of the most probable entry.
  final int best;

  /// Label of the most probable entry.
  String get bestKey => keys[best];

  /// Yes / no: the probability of yes. Score: the expected level, from 0 to levels - 1.
  /// Choice: the probability of the best option.
  final double value;

  /// 0 to 1 for choice and score. For yes / no, how far the probability of yes is from 0.5.
  final double confidence;

  final int promptTokens;
  final double ms;

  /// Yes / no questions only: the model says yes.
  bool get yes => type == KilnQuestionType.yesNo && best == 1;

  Map<String, Object> toMap() => {
        'type': type.index, 'probabilities': probabilities, 'keys': keys, 'best': best,
        'value': value, 'confidence': confidence, 'promptTokens': promptTokens, 'ms': ms,
      };
  factory KilnDecision.fromMap(Map m) => KilnDecision(
        type: KilnQuestionType.values[m['type']],
        probabilities: (m['probabilities'] as List).cast<double>(),
        keys: (m['keys'] as List).cast<String>(),
        best: m['best'], value: m['value'], confidence: m['confidence'],
        promptTokens: m['promptTokens'], ms: m['ms'],
      );

  @override
  String toString() =>
      '$bestKey (${(probabilities[best] * 100).toStringAsFixed(1)}%), '
      '$promptTokens tok, ${ms.round()} ms';
}

/// What a text is for, when it is embedded for search. Models are trained with a short prefix
/// for each role, and kiln adds the right one.
enum KilnEmbedKind {
  /// Embed the text exactly as given.
  raw,

  /// A search query ("how do I reset my password").
  query,

  /// A text that will be searched (a note, a paragraph of a document).
  document,
}

/// One image or one audio clip for [KilnContext.embedMedia] or [KilnContext.transcribe]: a file
/// on the device, or the bytes of such a file. Images: jpg, png, bmp, gif. Audio: wav, mp3, flac.
class KilnMedia {
  /// A file on the device.
  const KilnMedia.file(String this.path) : bytes = null;

  /// The contents of an image or audio file, for example from a picker or the network.
  const KilnMedia.bytes(Uint8List this.bytes) : path = null;

  final String? path;
  final Uint8List? bytes;

  /// Put this in the text where a media item belongs:
  /// `'A red shoe ${KilnMedia.marker} seen from the side'`.
  static const marker = '<__media__>';

  Map<String, Object?> toMap() => {'path': path, 'bytes': bytes};
  factory KilnMedia.fromMap(Map m) =>
      m['path'] != null ? KilnMedia.file(m['path']) : KilnMedia.bytes(m['bytes']);
}

/// What a projector file can encode. See [KilnModel.loadProjector].
class KilnProjectorInfo {
  const KilnProjectorInfo({required this.supportsImages, required this.supportsAudio});
  final bool supportsImages, supportsAudio;
  Map<String, Object> toMap() => {'supportsImages': supportsImages, 'supportsAudio': supportsAudio};
  factory KilnProjectorInfo.fromMap(Map m) =>
      KilnProjectorInfo(supportsImages: m['supportsImages'], supportsAudio: m['supportsAudio']);
  @override
  String toString() => 'projector: images ${supportsImages ? 'yes' : 'no'}, audio ${supportsAudio ? 'yes' : 'no'}';
}

/// The vector an embedding model made for a text, an image or an audio clip.
/// Inputs with a similar meaning get similar vectors; compare two with [similarity].
class KilnEmbedding {
  KilnEmbedding({
    required this.vector,
    required this.tokens,
    required this.mediaTokens,
    required this.prefixed,
    required this.mediaMs,
    required this.ms,
  });

  /// The numbers. Length 1 unless `normalize: false` was passed. Store this to search later.
  final Float32List vector;

  /// Tokens the model read: text, prefix and media together.
  final int tokens;

  /// Of those, how many came from images and audio.
  final int mediaTokens;

  /// A task prefix was added (see [KilnEmbedKind]).
  final bool prefixed;

  /// Time inside the image / audio encoders, and total time, in milliseconds.
  final double mediaMs, ms;

  /// How many numbers [vector] has.
  int get dimensions => vector.length;

  /// How close this is to [other]: about 1 for the same meaning, about 0 for unrelated.
  /// Both must come from the same model with the same `dimensions`.
  double similarity(KilnEmbedding other) => cosine(vector, other.vector);

  /// Cosine similarity of two vectors, for vectors you stored yourself.
  static double cosine(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('vectors have different lengths: ${a.length} and ${b.length}');
    }
    var dot = 0.0, na = 0.0, nb = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return na == 0 || nb == 0 ? 0 : dot / math.sqrt(na * nb);
  }

  Map<String, Object> toMap() => {
        'vector': vector, 'tokens': tokens, 'mediaTokens': mediaTokens, 'prefixed': prefixed,
        'mediaMs': mediaMs, 'ms': ms,
      };
  factory KilnEmbedding.fromMap(Map m) => KilnEmbedding(
        vector: m['vector'], tokens: m['tokens'], mediaTokens: m['mediaTokens'],
        prefixed: m['prefixed'], mediaMs: m['mediaMs'], ms: m['ms'],
      );

  @override
  String toString() =>
      '$dimensions numbers, $tokens tok${mediaTokens > 0 ? ' ($mediaTokens media, encoder ${mediaMs.round()} ms)' : ''}, '
      '${ms.round()} ms${prefixed ? ', prefix added' : ''}';
}

/// What happened in one [KilnContext.transcribe] call.
class KilnTranscribeStats {
  KilnTranscribeStats({
    required this.audioSeconds,
    required this.audioTokens,
    required this.promptTokens,
    required this.generatedTokens,
    required this.encodeMs,
    required this.promptMs,
    required this.genMs,
    required this.ms,
    required this.language,
    required this.stoppedByEos,
    required this.stoppedByUser,
  });
  factory KilnTranscribeStats.fromNative(b.kiln_transcribe_stats s) => KilnTranscribeStats(
        audioSeconds: s.audio_seconds,
        audioTokens: s.audio_tokens,
        promptTokens: s.prompt_tokens,
        generatedTokens: s.generated_tokens,
        encodeMs: s.encode_ms,
        promptMs: s.prompt_ms,
        genMs: s.gen_ms,
        ms: s.ms,
        language: _str(s.language, 32),
        stoppedByEos: s.stopped_by_eos,
        stoppedByUser: s.stopped_by_user,
      );

  /// Length of the clip.
  final double audioSeconds;

  /// Tokens the clip took, tokens of the whole prompt (instruction and clip; Whisper and
  /// Parakeet take no instruction), and tokens written.
  final int audioTokens, promptTokens, generatedTokens;

  /// Time in the audio encoder, reading the prompt, writing the words, and the whole call
  /// (reading the file included), in milliseconds.
  final double encodeMs, promptMs, genMs, ms;

  /// The language the model heard, as it named it ("English"); empty when it did not say
  /// (Parakeet never does; Qwen3-ASR writes "None" for a clip with no speech).
  final String language;

  /// The model finished by itself; or the callback / [KilnTranscription.cancel] stopped it.
  final bool stoppedByEos, stoppedByUser;

  /// How many times faster than real time: 2.0 means a 10-second clip took 5 seconds.
  double get speed => ms > 0 ? audioSeconds * 1000 / ms : 0;

  Map<String, Object> toMap() => {
        'audioSeconds': audioSeconds, 'audioTokens': audioTokens, 'promptTokens': promptTokens,
        'generatedTokens': generatedTokens, 'encodeMs': encodeMs, 'promptMs': promptMs, 'genMs': genMs, 'ms': ms,
        'language': language, 'stoppedByEos': stoppedByEos, 'stoppedByUser': stoppedByUser,
      };
  factory KilnTranscribeStats.fromMap(Map m) => KilnTranscribeStats(
        audioSeconds: m['audioSeconds'], audioTokens: m['audioTokens'], promptTokens: m['promptTokens'],
        generatedTokens: m['generatedTokens'], encodeMs: m['encodeMs'], promptMs: m['promptMs'], genMs: m['genMs'],
        ms: m['ms'], language: m['language'], stoppedByEos: m['stoppedByEos'], stoppedByUser: m['stoppedByUser'],
      );

  @override
  String toString() =>
      '${audioSeconds.toStringAsFixed(1)} s of audio in ${ms.round()} ms (${speed.toStringAsFixed(1)}x real time), '
      '$generatedTokens tok${language.isNotEmpty ? ', $language' : ''}${stoppedByEos ? '' : ' [cut]'}';
}

/// One streamed fragment of model output.
class KilnPiece {
  const KilnPiece(this.channel, this.text);
  final KilnChannel channel;
  final String text;
  bool get isReasoning => channel == KilnChannel.reasoning;
}

/// Helper: read a C string field into Dart.
String cStr(Pointer<Char> p) => p == nullptr ? '' : p.cast<Utf8>().toDartString();
