import 'dart:async';
import 'dart:ffi';
import 'dart:typed_data';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'kiln_bindings_generated.dart' as b;
import 'native.dart';
import 'types.dart';
import 'worker.dart';

const bool _trace = bool.fromEnvironment('KILN_TRACE');
void _t(String m) {
  if (_trace) print('[kiln] $m'); // ignore: avoid_print
}

/// Entry point. One [Kiln] per app is plenty; it owns a background isolate
/// where every blocking native call runs.
class Kiln {
  Kiln._(this._worker);
  final KilnWorker _worker;

  /// [libraryPath] is only needed on desktop, pointing at a host build of libkiln.
  static Future<Kiln> create({String? libraryPath}) async {
    KilnNative.libraryPathOverride = libraryPath;
    return Kiln._(await KilnWorker.spawn(libraryPath));
  }

  static String get version => cStr(KilnNative.bindings.kiln_version());

  Future<KilnDeviceInfo> probeDevice() async =>
      KilnDeviceInfo.fromMap(await _worker.call<Map>('probe', {}));

  Future<KilnModel> loadModel(
    String path, {
    KilnModelParams params = const KilnModelParams(),
  }) async {
    final r = await _worker.call<Map>('load', {
      'path': path,
      'params': params.toMap(),
    });
    return KilnModel._(
      this,
      r['handle'] as int,
      KilnModelInfo.fromMap(r['info'] as Map),
    );
  }

  void dispose() => _worker.dispose();
}

class KilnModel {
  KilnModel._(this._kiln, this._handle, this.info);
  final Kiln _kiln;
  final int _handle;

  /// What actually loaded: backend, quant, size, thinking support.
  final KilnModelInfo info;

  Future<KilnContext> createContext({
    KilnContextParams params = const KilnContextParams(),
  }) async {
    final h = await _kiln._worker.call<int>('createContext', {
      'model': _handle,
      'params': params.toMap(),
    });
    return KilnContext._(this, h);
  }

  /// Renders the model's chat template. Mostly for debugging; [KilnContext.chat] does this for you.
  Future<String> applyChatTemplate(
    List<KilnChatMessage> messages, {
    bool addAssistantPrefix = true,
    bool enableThinking = false,
  }) => _kiln._worker.call<String>('template', {
    'model': _handle,
    'messages': messages.map((m) => m.toMap()).toList(),
    'addAssistantPrefix': addAssistantPrefix,
    'enableThinking': enableThinking,
  });

  /// Loads the image / audio encoders that belong to an embedding model: its projector file,
  /// an "mmproj" GGUF published next to the model. Call it once after loading the model and
  /// before [KilnContext.embedMedia]. The encoders stay in memory until the model is disposed,
  /// and are often larger than the text model itself.
  ///
  /// [imageMaxTokens] is the most tokens one image may take; a large photo is scaled down to
  /// fit. Time grows much faster than this number. 0 means 280, a negative value means the
  /// model's own limit.
  Future<KilnProjectorInfo> loadProjector(
    String path, {
    int imageMaxTokens = 0,
  }) async => KilnProjectorInfo.fromMap(
    await _kiln._worker.call<Map>('loadProjector', {
      'model': _handle,
      'path': path,
      'imageMaxTokens': imageMaxTokens,
    }),
  );

  Future<void> dispose() =>
      _kiln._worker.call<void>('freeModel', {'handle': _handle});
}

/// A streaming generation in progress.
class KilnGeneration {
  KilnGeneration._(this.pieces, this.stats, this._cancel);

  /// Token pieces as they arrive, tagged answer or reasoning.
  final Stream<KilnPiece> pieces;

  /// Resolves when generation ends, with the truth about what happened.
  final Future<KilnGenStats> stats;

  final void Function() _cancel;

  /// Stops generation. [stats] still resolves, with stoppedByUser = true.
  void cancel() => _cancel();

  /// Convenience: collect the answer channel only.
  Future<String> get answer =>
      pieces.where((p) => !p.isReasoning).map((p) => p.text).join();
}

/// A transcription in progress. See [KilnContext.transcribe].
class KilnTranscription {
  KilnTranscription._(this.pieces, this.stats, this._cancel);

  /// The words as they are produced. Whisper and Parakeet give a sentence at a time,
  /// Qwen3-ASR a token at a time.
  final Stream<String> pieces;

  /// Resolves when the call ends, with the clip length, the time taken and the language.
  final Future<KilnTranscribeStats> stats;

  final void Function() _cancel;

  /// Stops the call. [stats] still resolves, with stoppedByUser = true.
  void cancel() => _cancel();

  /// Convenience: the whole text.
  Future<String> get text => pieces.join();
}

/// Speech being generated. See [KilnContext.speak].
class KilnSpeech {
  KilnSpeech._(this.pieces, this.stats, this._cancel);

  /// Pieces of audio as they are made, about a second each: mono float samples in -1..1.
  /// Play them as they arrive, or collect them with [wav].
  final Stream<KilnAudioPiece> pieces;

  /// Resolves when the call ends: length, timing, and whether the model finished by itself.
  final Future<KilnSpeakStats> stats;

  final void Function() _cancel;

  /// Stops generation. [stats] still resolves, with stoppedByUser = true.
  void cancel() => _cancel();

  /// Convenience: the whole utterance as the bytes of a 16-bit wav file.
  Future<Uint8List> get wav async {
    final all = <double>[];
    var rate = 0;
    await for (final p in pieces) {
      all.addAll(p.samples);
      rate = p.sampleRate;
    }
    return pcmToWav(Float32List.fromList(all), rate);
  }
}

/// One piece of generated audio.
class KilnAudioPiece {
  const KilnAudioPiece(this.samples, this.sampleRate);
  final Float32List samples;
  final int sampleRate;
  double get seconds => samples.length / sampleRate;
}

class KilnContext {
  KilnContext._(this.model, this._handle);
  final KilnModel model;
  final int _handle;
  bool _busy = false;

  /// Pass the full conversation every turn; kiln reuses whatever prefix is already
  /// in the KV cache. Only one chat may run on a context at a time.
  ///
  /// [maxReasoningTokens] is a thinking budget: after that many reasoning tokens kiln
  /// closes the think block itself and the model goes on to answer (0 = unlimited).
  /// Small reasoning models often never stop thinking on their own, so set it when
  /// [enableThinking] is on. [KilnGenStats.reasoningTruncated] reports when it was used.
  KilnGeneration chat(
    List<KilnChatMessage> messages, {
    bool enableThinking = false,
    int maxReasoningTokens = 0,
    KilnSamplingParams sampling = const KilnSamplingParams(),
  }) {
    if (_busy) {
      throw StateError(
        'kiln: context is busy; await the previous generation or cancel it',
      );
    }
    _busy = true;
    _t('chat start');
    final port = ReceivePort();
    final ctrl = StreamController<KilnPiece>();
    port.listen((msg) {
      final m = msg as List;
      ctrl.add(KilnPiece(KilnChannel.values[m[0] as int], m[1] as String));
    });
    final flag = _CancelFlag();
    final stats = model._kiln._worker
        .call<Map>('chat', {
          'context': _handle,
          'cancelFlag': flag.address,
          'messages': messages.map((m) => m.toMap()).toList(),
          'enableThinking': enableThinking,
          'maxReasoningTokens': maxReasoningTokens,
          'sampling': sampling.toMap(),
        }, stream: port.sendPort)
        .then((r) => KilnGenStats.fromMap(r['stats'] as Map))
        .catchError((Object e) {
          // Mirror the failure onto the pieces stream (before it is closed below) so a
          // listener that only consumes pieces also learns about it.
          if (!ctrl.isClosed) ctrl.addError(e);
          throw e;
        })
        .whenComplete(() async {
          _t('chat complete');
          _busy = false;
          flag.free();
          // let queued piece messages drain before closing
          await Future<void>.delayed(Duration.zero);
          port.close();
          // Do not await close(): its future only completes once a listener has drained
          // the stream, and a caller is allowed to ignore `pieces` and await `stats` alone.
          unawaited(ctrl.close());
        });
    return KilnGeneration._(ctrl.stream, stats, () {
      _t('cancel');
      if (flag.set()) {
        KilnNative.bindings.kiln_cancel(
          Pointer<b.kiln_context>.fromAddress(_handle),
        );
      }
      _t('cancel sent');
    });
  }

  /// Ask a decision model one typed question about [state] (the text the question is about).
  /// The model does not write text: it returns a probability per option in one forward pass.
  ///
  /// ```dart
  /// final d = await ctx.decide(
  ///   KilnQuestion.choice('Which team should handle this?',
  ///       [KilnOption('billing', 'Payments and refunds'), KilnOption('shipping', 'Delivery problems')]),
  ///   state: 'My parcel never arrived.');
  /// print(d.bestKey);   // shipping
  /// ```
  ///
  /// Throws a [KilnException] with `KILN_ERR_NOT_SUPPORTED` when the model is not a decision
  /// model; check [KilnModelInfo.isDecisionModel].
  Future<KilnDecision> decide(KilnQuestion question, {String? state}) async {
    if (_busy) {
      throw StateError(
        'kiln: context is busy; await the previous generation or cancel it',
      );
    }
    return KilnDecision.fromMap(
      await model._kiln._worker.call<Map>('decide', {
        'context': _handle,
        'question': question.toMap(),
        'state': state,
      }),
    );
  }

  /// Turn a text into a vector with an embedding model. Texts with a similar meaning get
  /// similar vectors, which is what search over the user's own notes needs.
  ///
  /// ```dart
  /// final note  = await ctx.embed('Knead the dough and let it rise.', kind: KilnEmbedKind.document);
  /// final query = await ctx.embed('how do I bake bread', kind: KilnEmbedKind.query);
  /// print(query.similarity(note));   // about 0.7; an unrelated note scores lower
  /// ```
  ///
  /// Use [KilnEmbedKind.query] for what the user types and [KilnEmbedKind.document] for what
  /// is searched. [title] is an optional document title, for models that take one.
  /// [dimensions] keeps only the first N numbers, which is smaller to store; only models
  /// trained for it stay accurate (EmbeddingGemma: 512, 256, 128). 0 keeps the full vector.
  ///
  /// Each call stands alone. A text longer than the context (`KilnContextParams.nCtx`, 2048
  /// tokens by default for embedding models) throws `KILN_ERR_CONTEXT_FULL`: nothing is cut
  /// silently, so split long documents into pieces. Throws `KILN_ERR_NOT_SUPPORTED` when the
  /// model is not an embedding model; check [KilnModelInfo.isEmbeddingModel].
  Future<KilnEmbedding> embed(
    String text, {
    KilnEmbedKind kind = KilnEmbedKind.raw,
    String? title,
    int dimensions = 0,
    bool normalize = true,
  }) => _embed(text, const [], kind, title, dimensions, normalize);

  /// Like [embed], for images and audio, alone or together with [text]. The result is in the
  /// same space as [embed]'s, so a typed query can find a photo or a voice note.
  ///
  /// ```dart
  /// await model.loadProjector('/path/to/mmproj.gguf');
  /// final photo = await ctx.embedMedia([KilnMedia.file('/path/to/holiday.jpg')]);
  /// final query = await ctx.embed('a lake between mountains', kind: KilnEmbedKind.query);
  /// print(query.similarity(photo));
  /// ```
  ///
  /// Each [KilnMedia.marker] in [text] is replaced by the next item; without a marker the
  /// items follow the text. No task prefix is added when media are present. An image costs a
  /// few hundred tokens of the context and takes seconds on a phone CPU; audio costs a few
  /// dozen tokens per second. Needs [KilnModel.loadProjector] first.
  Future<KilnEmbedding> embedMedia(
    List<KilnMedia> media, {
    String? text,
    int dimensions = 0,
    bool normalize = true,
  }) {
    if (media.isEmpty) {
      throw ArgumentError(
        'embedMedia needs at least one media item; use embed() for text alone',
      );
    }
    return _embed(text, media, KilnEmbedKind.raw, null, dimensions, normalize);
  }

  Future<KilnEmbedding> _embed(
    String? text,
    List<KilnMedia> media,
    KilnEmbedKind kind,
    String? title,
    int dimensions,
    bool normalize,
  ) async {
    if (_busy) {
      throw StateError(
        'kiln: context is busy; await the previous generation or cancel it',
      );
    }
    final n = model.info.nEmbd;
    return KilnEmbedding.fromMap(
      await model._kiln._worker.call<Map>('embed', {
        'context': _handle,
        'text': text,
        'media': [for (final m in media) m.toMap()],
        'kind': kind.index,
        'title': title,
        'dimensions': dimensions,
        'normalize': normalize,
        // room for the answer; the C side rejects a `dimensions` above the model's size
        'cap': dimensions > 0 && dimensions < n ? dimensions : n,
      }),
    );
  }

  /// Write down what is said in an audio clip (wav, mp3 or flac; a file or its bytes).
  ///
  /// ```dart
  /// final model = await kiln.loadModel('/path/to/ggml-parakeet-tdt-0.6b-v3-q8_0.bin');
  /// final ctx = await model.createContext();
  /// final t = ctx.transcribe(KilnMedia.file('/path/to/voice_note.wav'));
  /// print(await t.text);          // "The train to London leaves at nine in the morning."
  /// print((await t.stats).speed); // times faster than real time
  /// ```
  ///
  /// Works with a Whisper or Parakeet file ([KilnModelInfo.isSpeechModel]), or with Qwen3-ASR
  /// after [KilnModel.loadProjector]. Always picks the most likely word, so the same clip
  /// gives the same text. Each call stands alone; with Qwen3-ASR it also clears the
  /// context's conversation. [language] is a hint such as `'English'` (Parakeet ignores it; an
  /// English-only Whisper file refuses another language). [prompt] replaces Qwen3-ASR's
  /// instruction, or gives Whisper words the text may follow on from. [maxTokens] caps the
  /// text (Whisper: per 30-second window; Parakeet: ignored), 0 = no cap.
  ///
  /// Throws `KILN_ERR_NOT_SUPPORTED` on a model that cannot transcribe, `KILN_ERR_INVALID_ARG`
  /// for a file that is not audio, and with Qwen3-ASR `KILN_ERR_CONTEXT_FULL` for a clip
  /// longer than the context (cut long recordings into pieces).
  KilnTranscription transcribe(
    KilnMedia audio, {
    String? language,
    String? prompt,
    int maxTokens = 0,
  }) {
    if (_busy) {
      throw StateError(
        'kiln: context is busy; await the previous generation or cancel it',
      );
    }
    _busy = true;
    final port = ReceivePort();
    final ctrl = StreamController<String>();
    port.listen((msg) => ctrl.add(msg as String));
    final flag = _CancelFlag();
    final stats = model._kiln._worker
        .call<Map>('transcribe', {
          'context': _handle,
          'cancelFlag': flag.address,
          'audio': audio.toMap(),
          'language': language,
          'prompt': prompt,
          'maxTokens': maxTokens,
        }, stream: port.sendPort)
        .then((r) => KilnTranscribeStats.fromMap(r['stats'] as Map))
        .catchError((Object e) {
          if (!ctrl.isClosed) ctrl.addError(e);
          throw e;
        })
        .whenComplete(() async {
          _busy = false;
          flag.free();
          await Future<void>.delayed(Duration.zero);
          port.close();
          unawaited(ctrl.close());
        });
    return KilnTranscription._(ctrl.stream, stats, () {
      if (flag.set()) {
        KilnNative.bindings.kiln_cancel(
          Pointer<b.kiln_context>.fromAddress(_handle),
        );
      }
    });
  }

  /// Say [text] with a text-to-speech model, in the voice of [voice] (a few seconds of
  /// someone speaking, a file or its bytes). Pieces of audio arrive about once a second.
  ///
  /// ```dart
  /// final model = await kiln.loadModel('/path/to/pocket-tts-en.gguf');
  /// await model.loadProjector('/path/to/mmproj-pocket-tts-en.gguf');   // generatesAudio: true
  /// final ctx = await model.createContext();
  /// final s = ctx.speak('The train leaves at nine.', voice: KilnMedia.file('/path/to/me.wav'));
  /// final wavBytes = await s.wav;          // play it, or play s.pieces as they come
  /// print((await s.stats).speed);          // times faster than real time
  /// ```
  ///
  /// Pocket TTS produces almost nothing without a [voice]. [language] is for Qwen3-TTS
  /// ("en", "zh", ...); Pocket TTS ignores it. [maxFrames] caps the length (80 ms a frame),
  /// 0 = 512. The same [seed] gives the same audio; null means random. Speaking leaves the
  /// chat in this context untouched. Throws `KILN_ERR_NOT_SUPPORTED` when the loaded
  /// projector does not generate audio; check [KilnProjectorInfo.generatesAudio].
  KilnSpeech speak(
    String text, {
    KilnMedia? voice,
    String? language,
    int maxFrames = 0,
    int? seed,
  }) {
    if (_busy) {
      throw StateError(
        'kiln: context is busy; await the previous generation or cancel it',
      );
    }
    _busy = true;
    final port = ReceivePort();
    final ctrl = StreamController<KilnAudioPiece>();
    port.listen((msg) {
      final m = msg as List;
      ctrl.add(KilnAudioPiece(m[0] as Float32List, m[1] as int));
    });
    final flag = _CancelFlag();
    final stats = model._kiln._worker
        .call<Map>('speak', {
          'context': _handle,
          'cancelFlag': flag.address,
          'text': text,
          'voice': voice?.toMap(),
          'language': language,
          'maxFrames': maxFrames,
          'seed': seed ?? 0xFFFFFFFF,
        }, stream: port.sendPort)
        .then((r) => KilnSpeakStats.fromMap(r['stats'] as Map))
        .catchError((Object e) {
          if (!ctrl.isClosed) ctrl.addError(e);
          throw e;
        })
        .whenComplete(() async {
          _busy = false;
          flag.free();
          await Future<void>.delayed(Duration.zero);
          port.close();
          unawaited(ctrl.close());
        });
    return KilnSpeech._(ctrl.stream, stats, () {
      if (flag.set()) {
        KilnNative.bindings.kiln_cancel(
          Pointer<b.kiln_context>.fromAddress(_handle),
        );
      }
    });
  }

  /// Clears the KV cache. Next chat starts from scratch.
  Future<void> reset() {
    _t('reset');
    return model._kiln._worker.call<void>('reset', {'handle': _handle});
  }

  Future<void> dispose() =>
      model._kiln._worker.call<void>('freeContext', {'handle': _handle});
}

/// One request's cancel flag, in native memory so the worker isolate can read it. Set by
/// cancel() even before the worker has started the call; freed when the call is over, after
/// which cancel() does nothing (it runs on this isolate, like the completion, so they cannot race).
class _CancelFlag {
  final Pointer<Int32> _p = calloc<Int32>();
  bool _freed = false;

  int get address => _p.address;

  /// Sets the flag; false when the call is already over.
  bool set() {
    if (_freed) return false;
    _p.value = 1;
    return true;
  }

  void free() {
    if (_freed) return;
    _freed = true;
    calloc.free(_p);
  }
}
