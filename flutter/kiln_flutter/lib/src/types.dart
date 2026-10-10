import 'dart:ffi';

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
      );

  final KilnBackend backend;
  final String backendName, arch, quant;
  final int nParams, nCtxTrain, nVocab, gpuLayersUsed, modelBytes;
  final bool supportsThinking;

  Map<String, Object> toMap() => {
        'backend': backend.index, 'backendName': backendName, 'arch': arch, 'quant': quant,
        'nParams': nParams, 'nCtxTrain': nCtxTrain, 'nVocab': nVocab,
        'gpuLayersUsed': gpuLayersUsed, 'modelBytes': modelBytes, 'supportsThinking': supportsThinking,
      };
  factory KilnModelInfo.fromMap(Map m) => KilnModelInfo(
        backend: KilnBackend.values[m['backend']], backendName: m['backendName'], arch: m['arch'],
        quant: m['quant'], nParams: m['nParams'], nCtxTrain: m['nCtxTrain'], nVocab: m['nVocab'],
        gpuLayersUsed: m['gpuLayersUsed'], modelBytes: m['modelBytes'], supportsThinking: m['supportsThinking'],
      );

  @override
  String toString() =>
      '$arch $quant, ${(nParams / 1e9).toStringAsFixed(2)}B params, '
      '${(modelBytes / 1e9).toStringAsFixed(2)} GB, via $backendName'
      '${supportsThinking ? ', thinking' : ''}';
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
      );
  final int promptTokens, cachedTokens, generatedTokens, reasoningTokens;
  final double promptMs, genMs, promptTps, genTps;
  final bool stoppedByEos, stoppedByUser;

  Map<String, Object> toMap() => {
        'promptTokens': promptTokens, 'cachedTokens': cachedTokens, 'generatedTokens': generatedTokens,
        'reasoningTokens': reasoningTokens, 'promptMs': promptMs, 'genMs': genMs,
        'promptTps': promptTps, 'genTps': genTps, 'stoppedByEos': stoppedByEos, 'stoppedByUser': stoppedByUser,
      };
  factory KilnGenStats.fromMap(Map m) => KilnGenStats(
        promptTokens: m['promptTokens'], cachedTokens: m['cachedTokens'], generatedTokens: m['generatedTokens'],
        reasoningTokens: m['reasoningTokens'], promptMs: m['promptMs'], genMs: m['genMs'],
        promptTps: m['promptTps'], genTps: m['genTps'], stoppedByEos: m['stoppedByEos'], stoppedByUser: m['stoppedByUser'],
      );

  @override
  String toString() =>
      'prompt $promptTokens tok ($cachedTokens cached) ${promptMs.round()} ms, '
      'gen $generatedTokens tok ($reasoningTokens reasoning) ${genMs.round()} ms, '
      '${genTps.toStringAsFixed(1)} tok/s'
      '${stoppedByEos ? ' [eos]' : stoppedByUser ? ' [stopped]' : ''}';
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
