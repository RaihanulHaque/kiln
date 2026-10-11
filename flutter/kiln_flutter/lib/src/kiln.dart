import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';

import 'kiln_bindings_generated.dart' as b;
import 'native.dart';
import 'types.dart';
import 'worker.dart';

const bool _trace = bool.fromEnvironment('KILN_TRACE');
void _t(String m) { if (_trace) print('[kiln] $m'); } // ignore: avoid_print

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

  Future<KilnModel> loadModel(String path, {KilnModelParams params = const KilnModelParams()}) async {
    final r = await _worker.call<Map>('load', {'path': path, 'params': params.toMap()});
    return KilnModel._(this, r['handle'] as int, KilnModelInfo.fromMap(r['info'] as Map));
  }

  void dispose() => _worker.dispose();
}

class KilnModel {
  KilnModel._(this._kiln, this._handle, this.info);
  final Kiln _kiln;
  final int _handle;

  /// What actually loaded: backend, quant, size, thinking support.
  final KilnModelInfo info;

  Future<KilnContext> createContext({KilnContextParams params = const KilnContextParams()}) async {
    final h = await _kiln._worker.call<int>('createContext', {'model': _handle, 'params': params.toMap()});
    return KilnContext._(this, h);
  }

  /// Renders the model's chat template. Mostly for debugging; [KilnContext.chat] does this for you.
  Future<String> applyChatTemplate(List<KilnChatMessage> messages,
          {bool addAssistantPrefix = true, bool enableThinking = false}) =>
      _kiln._worker.call<String>('template', {
        'model': _handle,
        'messages': messages.map((m) => m.toMap()).toList(),
        'addAssistantPrefix': addAssistantPrefix,
        'enableThinking': enableThinking,
      });

  Future<void> dispose() => _kiln._worker.call<void>('freeModel', {'handle': _handle});
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
    if (_busy) throw StateError('kiln: context is busy; await the previous generation or cancel it');
    _busy = true;
    _t('chat start');
    final port = ReceivePort();
    final ctrl = StreamController<KilnPiece>();
    port.listen((msg) {
      final m = msg as List;
      ctrl.add(KilnPiece(KilnChannel.values[m[0] as int], m[1] as String));
    });
    final stats = model._kiln._worker
        .call<Map>('chat', {
          'context': _handle,
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
          // let queued piece messages drain before closing
          await Future<void>.delayed(Duration.zero);
          port.close();
          // Do not await close(): its future only completes once a listener has drained
          // the stream, and a caller is allowed to ignore `pieces` and await `stats` alone.
          unawaited(ctrl.close());
        });
    return KilnGeneration._(ctrl.stream, stats, () {
      _t('cancel');
      KilnNative.bindings.kiln_cancel(Pointer<b.kiln_context>.fromAddress(_handle));
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
    if (_busy) throw StateError('kiln: context is busy; await the previous generation or cancel it');
    return KilnDecision.fromMap(await model._kiln._worker
        .call<Map>('decide', {'context': _handle, 'question': question.toMap(), 'state': state}));
  }

  /// Clears the KV cache. Next chat starts from scratch.
  Future<void> reset() { _t('reset'); return model._kiln._worker.call<void>('reset', {'handle': _handle}); }

  Future<void> dispose() => model._kiln._worker.call<void>('freeContext', {'handle': _handle});
}
