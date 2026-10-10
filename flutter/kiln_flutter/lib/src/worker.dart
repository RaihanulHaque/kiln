// Background isolate that owns the native handles. kiln_chat blocks, so every
// call that can take time runs here, and pieces are streamed back over a port.
import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'kiln_bindings_generated.dart' as b;
import 'native.dart';
import 'types.dart';

const bool _trace = bool.fromEnvironment('KILN_TRACE');
void _t(String m) { if (_trace) print('[kiln-worker] $m'); } // ignore: avoid_print

class _Req {
  _Req(this.id, this.op, this.args, [this.stream]);
  final int id;
  final String op;
  final Map<String, Object?> args;
  final SendPort? stream;
}

class KilnWorker {
  KilnWorker._(this._isolate, this._send);

  final Isolate _isolate;
  final SendPort _send;
  final _pending = <int, Completer<Object?>>{};
  int _nextId = 0;
  late final ReceivePort _rx;

  static Future<KilnWorker> spawn(String? libraryPath) async {
    final rx = ReceivePort();
    final isolate = await Isolate.spawn(_main, [rx.sendPort, libraryPath], debugName: 'kiln-worker');
    final send = await rx.first as SendPort;
    final w = KilnWorker._(isolate, send);
    w._rx = ReceivePort();
    w._rx.listen(w._onReply);
    send.send(w._rx.sendPort);
    return w;
  }

  void _onReply(dynamic msg) {
    final m = msg as List;
    final id = m[0] as int;
    final c = _pending.remove(id);
    if (c == null) return;
    if (m[1] == 'err') {
      c.completeError(KilnException(b.kiln_status.fromValue(m[2] as int), m[3] as String));
    } else {
      c.complete(m[2]);
    }
  }

  Future<T> call<T>(String op, Map<String, Object?> args, {SendPort? stream}) {
    final id = _nextId++;
    final c = Completer<Object?>();
    _pending[id] = c;
    _send.send(_Req(id, op, args, stream));
    return c.future.then((v) => v as T);
  }

  void dispose() {
    _rx.close();
    _isolate.kill(priority: Isolate.immediate);
  }

  // ------------------------------------------------------------ isolate side

  static void _main(List init) {
    final boot = init[0] as SendPort;
    KilnNative.libraryPathOverride = init[1] as String?;
    final rx = ReceivePort();
    boot.send(rx.sendPort);
    SendPort? reply;
    final k = KilnNative.bindings;

    rx.listen((msg) {
      if (msg is SendPort) { reply = msg; return; }
      final req = msg as _Req;
      _t('recv #${req.id} ${req.op}');
      try {
        final result = _dispatch(k, req);
        _t('done #${req.id} ${req.op}');
        reply!.send([req.id, 'ok', result]);
      } on KilnException catch (e) {
        reply!.send([req.id, 'err', e.status.value, e.message]);
      } catch (e) {
        reply!.send([req.id, 'err', b.kiln_status.KILN_ERR_INTERNAL.value, e.toString()]);
      }
    });
  }

  static Never _fail(b.KilnBindings k, b.kiln_status s) =>
      throw KilnException(s, cStr(k.kiln_last_error()));

  static Object? _dispatch(b.KilnBindings k, _Req r) {
    switch (r.op) {
      case 'probe':
        final d = calloc<b.kiln_device_info>();
        try {
          final s = k.kiln_probe_device(d);
          if (s != b.kiln_status.KILN_OK) _fail(k, s);
          return KilnDeviceInfo.fromNative(d.ref).toMap();
        } finally { calloc.free(d); }

      case 'load':
        final p = KilnModelParams.fromMap(r.args['params'] as Map);
        final path = (r.args['path'] as String).toNativeUtf8();
        final mp = calloc<b.kiln_model_params>();
        final out = calloc<Pointer<b.kiln_model>>();
        try {
          mp.ref = k.kiln_model_params_default();
          mp.ref.backendAsInt = p.backend.index;
          mp.ref.n_gpu_layers = p.gpuLayers;
          mp.ref.n_threads = p.threads;
          mp.ref.use_mmap = p.useMmap;
          mp.ref.use_mlock = p.useMlock;
          final s = k.kiln_model_load(path.cast(), mp, out);
          if (s != b.kiln_status.KILN_OK) _fail(k, s);
          final info = calloc<b.kiln_model_info>();
          try {
            k.kiln_model_get_info(out.value, info);
            return {'handle': out.value.address, 'info': KilnModelInfo.fromNative(info.ref).toMap()};
          } finally { calloc.free(info); }
        } finally { calloc.free(path); calloc.free(mp); calloc.free(out); }

      case 'freeModel':
        k.kiln_model_free(Pointer.fromAddress(r.args['handle'] as int));
        return null;

      case 'createContext':
        final p = KilnContextParams.fromMap(r.args['params'] as Map);
        final cp = calloc<b.kiln_context_params>();
        final out = calloc<Pointer<b.kiln_context>>();
        try {
          cp.ref = k.kiln_context_params_default();
          cp.ref.n_ctx = p.nCtx;
          cp.ref.n_batch = p.nBatch;
          cp.ref.flash_attn = p.flashAttn;
          final s = k.kiln_context_create(Pointer.fromAddress(r.args['model'] as int), cp, out);
          if (s != b.kiln_status.KILN_OK) _fail(k, s);
          return out.value.address;
        } finally { calloc.free(cp); calloc.free(out); }

      case 'freeContext':
        k.kiln_context_free(Pointer.fromAddress(r.args['handle'] as int));
        return null;

      case 'reset':
        final s = k.kiln_context_reset(Pointer.fromAddress(r.args['handle'] as int));
        if (s != b.kiln_status.KILN_OK) _fail(k, s);
        return null;

      case 'template':
        return _template(k, r);

      case 'chat':
        return _chat(k, r);
    }
    throw KilnException(b.kiln_status.KILN_ERR_INVALID_ARG, 'unknown op ${r.op}');
  }

  static (Pointer<b.kiln_chat_message>, List<Pointer<Utf8>>) _marshalMessages(List msgs) {
    final arr = calloc<b.kiln_chat_message>(msgs.length);
    final strings = <Pointer<Utf8>>[];
    for (var i = 0; i < msgs.length; i++) {
      final m = KilnChatMessage.fromMap(msgs[i] as Map);
      final role = m.role.toNativeUtf8();
      final content = m.content.toNativeUtf8();
      strings..add(role)..add(content);
      arr[i].role = role.cast();
      arr[i].content = content.cast();
    }
    return (arr, strings);
  }

  static String _template(b.KilnBindings k, _Req r) {
    final msgs = r.args['messages'] as List;
    final (arr, strings) = _marshalMessages(msgs);
    final opts = calloc<b.kiln_chat_options>();
    try {
      opts.ref = k.kiln_chat_options_default();
      opts.ref.add_assistant_prefix = r.args['addAssistantPrefix'] as bool;
      opts.ref.enable_thinking = r.args['enableThinking'] as bool;
      final model = Pointer<b.kiln_model>.fromAddress(r.args['model'] as int);
      var need = k.kiln_apply_chat_template(model, arr, msgs.length, opts, nullptr, 0);
      if (need < 0) need = -need;
      final buf = calloc<Char>(need + 1);
      try {
        final n = k.kiln_apply_chat_template(model, arr, msgs.length, opts, buf, need + 1);
        if (n < 0) throw KilnException(b.kiln_status.KILN_ERR_BACKEND, cStr(k.kiln_last_error()));
        return buf.cast<Utf8>().toDartString(length: n);
      } finally { calloc.free(buf); }
    } finally {
      calloc.free(arr); calloc.free(opts);
      for (final s in strings) {
        calloc.free(s);
      }
    }
  }

  static Map<String, Object?> _chat(b.KilnBindings k, _Req r) {
    final msgs = r.args['messages'] as List;
    final (arr, strings) = _marshalMessages(msgs);
    final opts = calloc<b.kiln_chat_options>();
    final sp = calloc<b.kiln_sampling_params>();
    final st = calloc<b.kiln_gen_stats>();
    final stream = r.stream!;

    // Called synchronously from kiln_chat on this isolate's thread. Pieces go to the
    // main isolate; cancellation comes back via kiln_cancel, which is thread-safe.
    final cb = NativeCallable<b.kiln_token_cbFunction>.isolateLocal(
      (int ch, Pointer<Char> piece, int tok, Pointer<Void> user) {
        stream.send([ch, piece.cast<Utf8>().toDartString()]);
        return true;
      },
      exceptionalReturn: false,
    );
    try {
      opts.ref = k.kiln_chat_options_default();
      opts.ref.enable_thinking = r.args['enableThinking'] as bool;
      final p = KilnSamplingParams.fromMap(r.args['sampling'] as Map);
      sp.ref = k.kiln_sampling_params_default();
      sp.ref.temperature = p.temperature;
      sp.ref.top_p = p.topP;
      sp.ref.top_k = p.topK;
      sp.ref.min_p = p.minP;
      sp.ref.repeat_penalty = p.repeatPenalty;
      sp.ref.repeat_last_n = p.repeatLastN;
      sp.ref.seed = p.seed;
      sp.ref.max_tokens = p.maxTokens;
      final ctx = Pointer<b.kiln_context>.fromAddress(r.args['context'] as int);
      _t('kiln_chat enter');
      final s = k.kiln_chat(ctx, arr, msgs.length, opts, sp, cb.nativeFunction, nullptr, st);
      _t('kiln_chat exit status=${s.name}');
      final stats = KilnGenStats.fromNative(st.ref).toMap();
      if (s != b.kiln_status.KILN_OK && s != b.kiln_status.KILN_ERR_CANCELLED) _fail(k, s);
      return {'status': s.value, 'stats': stats};
    } finally {
      cb.close();
      calloc.free(arr); calloc.free(opts); calloc.free(sp); calloc.free(st);
      for (final x in strings) {
        calloc.free(x);
      }
    }
  }
}
