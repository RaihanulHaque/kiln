// On-device integration test for kiln_flutter.
//
// Needs a model in the app's external files dir. Push one first:
//   adb push models/LiquidAI_LFM2.5-350M_Q4_K_M.gguf \
//       /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
// then
//   cd flutter/kiln_flutter/example && flutter test integration_test -d <device>
//
// Missing models skip their tests rather than fail, so the suite is usable on any device.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:integration_test/integration_test.dart';
import 'package:kiln_flutter/kiln_flutter.dart';
import 'package:path_provider/path_provider.dart';

const lfm = 'LiquidAI_LFM2.5-350M_Q4_K_M.gguf';
const qwen = 'Qwen3.5-0.8B-Q4_K_M.gguf';
// The small model is fetched automatically when missing so the suite is self-sufficient on
// any device (229 MB, once). Qwen is optional: push it if you want the thinking test.
const lfmUrl = 'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf';
const gemma = 'gemma-3-270m-it-Q8_0.gguf';
// Speed group: these are fetched on demand (533 MB + 292 MB) only when KILN_SPEED=true.
const speedModels = {
  lfm: lfmUrl,
  qwen: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
  gemma: 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf',
};
const bool speed = bool.fromEnvironment('KILN_SPEED');

/// Download to a .part file and rename when complete, so a dropped connection never leaves a
/// truncated file that a later run would take for a model. Three attempts.
Future<File> download(Directory dir, String name, String url) async {
  final f = File('${dir.path}/$name');
  if (f.existsSync()) return f;
  final part = File('${f.path}.part');
  for (var attempt = 1; ; attempt++) {
    // ignore: avoid_print
    print('downloading $name (attempt $attempt) ...');
    final client = http.Client();
    try {
      final resp = await client.send(http.Request('GET', Uri.parse(url)));
      if (resp.statusCode != 200) throw HttpException('HTTP ${resp.statusCode}');
      await resp.stream.pipe(part.openWrite());
      final want = resp.contentLength;
      if (want != null && part.lengthSync() != want) throw const HttpException('short download');
      part.renameSync(f.path);
      // ignore: avoid_print
      print('downloaded ${f.lengthSync() ~/ 1000000} MB');
      return f;
    } catch (e) {
      if (part.existsSync()) part.deleteSync();
      if (attempt == 3) rethrow;
    } finally {
      client.close();
    }
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Kiln kiln;
  late Directory dir;

  setUpAll(() async {
    kiln = await Kiln.create();
    dir = Platform.isAndroid
        ? ((await getExternalStorageDirectory()) ?? await getApplicationDocumentsDirectory())
        : await getApplicationDocumentsDirectory();
    await download(dir, lfm, lfmUrl);
  });
  tearDownAll(() => kiln.dispose());

  Future<File> fetch(String name, String url) => download(dir, name, url);

  File? model(String name) {
    final f = File('${dir.path}/$name');
    return f.existsSync() ? f : null;
  }

  test('version and probe', () async {
    expect(Kiln.version, isNotEmpty);
    final d = await kiln.probeDevice();
    expect(d.cpuCores, greaterThan(0));
    expect(d.cpuFastCores, inInclusiveRange(1, d.cpuCores));
    expect(d.ramTotalBytes, greaterThan(1 << 30));
    expect(d.abi, 'arm64-v8a');
    // ignore: avoid_print
    print('device: $d');
  });

  test('bad paths fail cleanly', () async {
    await expectLater(kiln.loadModel('${dir.path}/does-not-exist.gguf'), throwsA(isA<KilnException>()));
    final junk = File('${dir.path}/junk.bin')..writeAsBytesSync(List.generate(4096, (i) => i * 31 & 0xff));
    try {
      await expectLater(kiln.loadModel(junk.path), throwsA(isA<KilnException>()));
    } finally {
      junk.deleteSync();
    }
  });

  group('LFM2.5 350M', () {
    late KilnModel m;
    late KilnContext ctx;

    setUpAll(() async {
      final f = model(lfm);
      if (f == null) return;
      m = await kiln.loadModel(f.path);
      ctx = await m.createContext(params: const KilnContextParams(nCtx: 2048));
    });
    tearDownAll(() async {
      if (model(lfm) == null) return;
      await ctx.dispose();
      await m.dispose();
    });

    test('info is truthful', () {
      if (model(lfm) == null) markTestSkipped('no $lfm on device');
      if (model(lfm) == null) return;
      expect(m.info.arch, 'lfm2');
      expect(m.info.backendName, startsWith('llama.cpp/'));
      expect(m.info.modelBytes, greaterThan(100 << 20));
      expect(m.info.supportsThinking, isFalse);
      // ignore: avoid_print
      print('model: ${m.info}');
    });

    test('template renders roles', () async {
      if (model(lfm) == null) return;
      final p = await m.applyChatTemplate([const KilnChatMessage.system('Be terse.'), const KilnChatMessage.user('Hi')]);
      expect(p, contains('Hi'));
      expect(p, contains('terse'));
    });

    test('single turn streams an answer with stats', () async {
      if (model(lfm) == null) return;
      final gen = ctx.chat([const KilnChatMessage.user('What is the capital of France? One word.')],
          sampling: const KilnSamplingParams.greedy(maxTokens: 16));
      final pieces = <KilnPiece>[];
      await for (final p in gen.pieces) {
        pieces.add(p);
      }
      final st = await gen.stats;
      final answer = pieces.where((p) => !p.isReasoning).map((p) => p.text).join();
      expect(answer, contains('Paris'));
      expect(pieces.any((p) => p.isReasoning), isFalse);
      expect(st.promptTokens, greaterThan(0));
      expect(st.cachedTokens, 0);
      expect(st.generatedTokens, inInclusiveRange(1, 16));
      expect(st.genTps, greaterThan(0));
      // ignore: avoid_print
      print('turn 1: "$answer" $st');
    });

    test('second turn reuses the KV prefix', () async {
      if (model(lfm) == null) return;
      await ctx.reset();
      final history = [const KilnChatMessage.user('What is the capital of France? One word.')];
      final g1 = ctx.chat(history, sampling: const KilnSamplingParams.greedy(maxTokens: 16));
      final a1 = await g1.answer;
      await g1.stats;
      history.add(KilnChatMessage.assistant(a1));
      history.add(const KilnChatMessage.user('What is the capital of Germany? One word.'));
      final g2 = ctx.chat(history, sampling: const KilnSamplingParams.greedy(maxTokens: 16));
      final a2 = await g2.answer;
      final st2 = await g2.stats;
      expect(a2, contains('Berlin'));
      expect(st2.cachedTokens, greaterThan(0));
      expect(st2.cachedTokens, lessThan(st2.promptTokens));
      // ignore: avoid_print
      print('turn 2: "$a2" $st2');
    });

    test('cancel stops a long generation', () async {
      if (model(lfm) == null) return;
      await ctx.reset();
      final gen = ctx.chat([const KilnChatMessage.user('Write a very long story about a dragon.')]);
      var n = 0;
      final sub = gen.pieces.listen((_) {
        if (++n == 5) gen.cancel();
      });
      final st = await gen.stats;
      await sub.cancel();
      expect(st.stoppedByUser, isTrue);
      expect(st.generatedTokens, lessThan(60));
    });

    test('context refuses a second concurrent chat', () async {
      if (model(lfm) == null) return;
      await ctx.reset();
      final g = ctx.chat([const KilnChatMessage.user('Count to twenty.')], sampling: const KilnSamplingParams.greedy(maxTokens: 24));
      expect(() => ctx.chat([const KilnChatMessage.user('x')]), throwsStateError);
      await g.stats;
    });

    test('context overflow surfaces as KilnException', () async {
      if (model(lfm) == null) return;
      final small = await m.createContext(params: const KilnContextParams(nCtx: 64));
      try {
        final big = List.filled(600, 'word').join(' ');
        final g = small.chat([KilnChatMessage.user(big)], sampling: const KilnSamplingParams.greedy(maxTokens: 4));
        await expectLater(g.stats, throwsA(isA<KilnException>()));
      } finally {
        await small.dispose();
      }
    });
  });

  group('speed', () {
    // Same job 3 times back to back per model, like kiln-bench: run 1 is cold, run 3 is warmer.
    const prompt = 'Write a detailed, multi-paragraph explanation of how a kiln works, covering history, '
        'heat transfer, the chemistry of firing clay, and electric versus wood-fired designs.';
    for (final entry in speedModels.entries) {
      test(entry.key, () async {
        if (!speed) {
          markTestSkipped('run with --dart-define=KILN_SPEED=true');
          return;
        }
        final f = await fetch(entry.key, entry.value);
        final sw = Stopwatch()..start();
        final m = await kiln.loadModel(f.path);
        final ctx = await m.createContext(params: const KilnContextParams(nCtx: 2048));
        final loadMs = sw.elapsedMilliseconds;
        final rows = <String>[];
        try {
          for (var r = 1; r <= 3; r++) {
            await ctx.reset();
            final g = ctx.chat([KilnChatMessage.user(prompt)], sampling: const KilnSamplingParams(maxTokens: 128, seed: 1));
            await g.pieces.drain<void>();
            final st = await g.stats;
            rows.add('run $r: prompt ${st.promptTokens} tok ${st.promptMs.round()} ms (${st.promptTps.toStringAsFixed(0)} tok/s), '
                'gen ${st.generatedTokens} tok ${st.genMs.round()} ms (${st.genTps.toStringAsFixed(1)} tok/s)');
          }
        } finally {
          await ctx.dispose();
          await m.dispose();
        }
        // ignore: avoid_print
        print('SPEED ${entry.key}: ${m.info} | load $loadMs ms\n  ${rows.join('\n  ')}');
      }, timeout: const Timeout(Duration(minutes: 15)));
    }
  });

  group('Qwen3.5 0.8B thinking', () {
    test('reasoning and answer arrive on separate channels', () async {
      final f = await fetch(qwen, speedModels[qwen]!);   // 533 MB, once per install
      final m = await kiln.loadModel(f.path);
      expect(m.info.supportsThinking, isTrue);
      final ctx = await m.createContext(params: const KilnContextParams(nCtx: 4096));
      try {
        final gen = ctx.chat([const KilnChatMessage.user('What is 17 * 23? Think briefly, then answer.')],
            enableThinking: true, maxReasoningTokens: 256,
            sampling: const KilnSamplingParams(maxTokens: 1500, seed: 7, repeatPenalty: 1.0));
        final reasoning = StringBuffer(), answer = StringBuffer();
        await for (final p in gen.pieces) {
          (p.isReasoning ? reasoning : answer).write(p.text);
        }
        final st = await gen.stats;
        expect(st.reasoningTokens, greaterThan(0));
        expect(st.reasoningTokens, lessThanOrEqualTo(256));
        expect(st.reasoningTruncated, st.reasoningTokens == 256);
        expect(answer.toString().trim(), isNotEmpty);
        expect(answer.toString(), isNot(contains('<think>')));
        expect(reasoning.toString(), isNot(contains('</think>')));
        if (st.stoppedByEos) expect(answer.toString(), contains('391'));
        // ignore: avoid_print
        print('think: ${st.reasoningTokens} reasoning tok, answer="${answer.toString().trim()}"');
      } finally {
        await ctx.dispose();
        await m.dispose();
      }
    }, timeout: const Timeout(Duration(minutes: 5)));
  });
}
