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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:integration_test/integration_test.dart';
import 'package:kiln_flutter/kiln_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'media_fixtures.dart';

const lfm = 'LiquidAI_LFM2.5-350M_Q4_K_M.gguf';
const qwen = 'Qwen3.5-0.8B-Q4_K_M.gguf';
// The small model is fetched automatically when missing so the suite is self-sufficient on
// any device (229 MB, once). Qwen is optional: push it if you want the thinking test.
const lfmUrl = 'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf';
const gemma = 'gemma-3-270m-it-Q8_0.gguf';
// Decision model, fetched on demand by its test (407 MB, once per install). Liquid AI lfm1.0 license.
const d1 = 'd1-omni-600M-Q8_0.gguf';
const d1Url = 'https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/d1-omni-600M-Q8_0.gguf';
// Speed group: these are fetched on demand (533 MB + 292 MB) only when KILN_SPEED=true.
const speedModels = {
  lfm: lfmUrl,
  qwen: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
  gemma: 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf',
};
const bool speed = bool.fromEnvironment('KILN_SPEED');
// Embedding model, fetched on demand by its tests (176 MB, once per install). Apache 2.0.
const gemmaEmbed = 'embeddinggemma-2-UD-Q4_K_XL.gguf';
const gemmaEmbedUrl = 'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf';
// Its image / audio encoders (555 MB), only fetched when KILN_MEDIA=true.
const gemmaProjector = 'mmproj-embeddinggemma-2-Q8_0.gguf';
const gemmaProjectorUrl = 'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/mmproj-Q8_0.gguf';
const bool mediaTests = bool.fromEnvironment('KILN_MEDIA');
// Speech models in whisper.cpp's format. Whisper base (60 MB) is fetched on demand by its test;
// Parakeet (669 MB) only when KILN_SPEECH=true.
const whisperBase = 'ggml-base-q5_1.bin';
const whisperBaseUrl = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin';
const parakeet = 'ggml-parakeet-tdt-0.6b-v3-q8_0.bin';
const parakeetUrl = 'https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-q8_0.bin';
const bool speechTests = bool.fromEnvironment('KILN_SPEECH');
// Text to speech: Pocket TTS backbone (159 MB) and its projector (60 MB), only when KILN_SPEECH=true.
const pocket = 'pocket-tts-en.gguf';
const pocketUrl = 'https://huggingface.co/EryriLabs/pocket-tts-GGUF/resolve/main/pocket-tts-en.gguf';
const pocketProjector = 'mmproj-pocket-tts-en.gguf';
const pocketProjectorUrl = 'https://huggingface.co/EryriLabs/pocket-tts-GGUF/resolve/main/mmproj-pocket-tts-en.gguf';

/// Lower-case letters and digits, one space between words: capitals and punctuation do not
/// count, and "9" is accepted for "nine".
String words(String s) => s.toLowerCase().replaceAll(RegExp('[^a-z0-9]+'), ' ').trim().replaceAll(' 9 ', ' nine ').replaceAll(' 2 ', ' two ').replaceAll(' 1 ', ' one ');


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

    test('embed on a chat model is refused', () async {
      expect(m.info.isEmbeddingModel, isFalse);
      expect(m.info.nEmbd, 0);
      final ctx = await m.createContext();
      try {
        await expectLater(ctx.embed('some text'),
            throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_NOT_SUPPORTED')));
        // a chat model may take a projector since 0.4.0 (Qwen3-ASR is one), so a missing
        // file is a load error, not "not supported"
        await expectLater(m.loadProjector('${dir.path}/nothing.gguf'),
            throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_MODEL_LOAD')));
      } finally {
        await ctx.dispose();
      }
    });

    test('transcribe on a chat model is refused', () async {
      final ctx = await m.createContext();
      try {
        await expectLater(ctx.transcribe(KilnMedia.bytes(speechTrainWav)).stats,
            throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_NOT_SUPPORTED')));
      } finally {
        await ctx.dispose();
      }
    });

    test('decide on a chat model is refused', () async {
      expect(m.info.isDecisionModel, isFalse);
      final ctx = await m.createContext();
      try {
        await expectLater(
            ctx.decide(KilnQuestion.yesNo('Is this a question?'), state: 'state'),
            throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_NOT_SUPPORTED')));
      } finally {
        await ctx.dispose();
      }
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

    // The same system prompt with a new question each time (extraction, classification): kiln
    // keeps the system prompt, also on a hybrid model like Qwen3.5, which needs a checkpoint for it.
    test('the system prompt is reused when only the question changes', () async {
      final f = await fetch(qwen, speedModels[qwen]!);
      final m = await kiln.loadModel(f.path);
      final ctx = await m.createContext();
      try {
        const sys = 'You are a terse geography assistant. Answer with the city name only, one word.';
        Future<(String, KilnGenStats)> ask(String q) async {
          final gen = ctx.chat([const KilnChatMessage.system(sys), KilnChatMessage.user(q)],
              sampling: const KilnSamplingParams(temperature: 0, maxTokens: 16, repeatPenalty: 1.0));
          final answer = StringBuffer();
          await for (final p in gen.pieces) {
            if (!p.isReasoning) answer.write(p.text);
          }
          return (answer.toString(), await gen.stats);
        }

        await ask('What is the capital of France?');
        final (answer, st) = await ask('What is the capital of Italy?');
        // ignore: avoid_print
        print('system reuse: "$answer" (cached ${st.cachedTokens} of ${st.promptTokens})');
        expect(st.cachedTokens, greaterThanOrEqualTo(15));
        expect(st.cachedTokens, lessThan(st.promptTokens));
        expect(answer, contains('Rome'));
      } finally {
        await ctx.dispose();
        await m.dispose();
      }
    }, timeout: const Timeout(Duration(minutes: 5)));
  });

  group('d1-omni decision model', () {
    late KilnModel m;
    late KilnContext ctx;
    const state = 'Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and '
        'nobody answered. I want my money back today or I will dispute the charge with my bank.';
    double sum(List<double> p) => p.fold(0.0, (a, b) => a + b);

    setUpAll(() async {
      final f = await fetch(d1, d1Url);
      m = await kiln.loadModel(f.path);
      ctx = await m.createContext();
    });
    tearDownAll(() async {
      await ctx.dispose();
      await m.dispose();
    });

    test('info says it is a decision model', () {
      expect(m.info.isDecisionModel, isTrue);
      // ignore: avoid_print
      print('model: ${m.info}');
    });

    test('yes / no', () async {
      final yes = await ctx.decide(
          KilnQuestion.yesNo('Is the customer asking for a refund?', id: 'wants_refund'), state: state);
      expect(yes.probabilities, hasLength(2));
      expect(sum(yes.probabilities), closeTo(1, 1e-3));
      expect(yes.yes, isTrue);
      expect(yes.value, greaterThan(0.5));
      expect(yes.bestKey, 'yes');
      // ignore: avoid_print
      print('refund? P(yes)=${yes.value.toStringAsFixed(3)}  $yes');

      final no = await ctx.decide(
          KilnQuestion.yesNo('Is the customer asking about the weather?', id: 'about_weather'), state: state);
      expect(no.yes, isFalse);
      expect(no.value, lessThan(0.5));
    });

    test('choice', () async {
      final d = await ctx.decide(
          const KilnQuestion.choice('Which team should handle this?', [
            KilnOption('billing', 'Payments, refunds and invoices'),
            KilnOption('shipping', 'Delivery status and lost parcels'),
            KilnOption('technical', 'The product does not work as expected'),
          ], id: 'team'),
          state: state);
      expect(d.probabilities, hasLength(3));
      expect(sum(d.probabilities), closeTo(1, 1e-3));
      expect(d.keys, ['billing', 'shipping', 'technical']);
      expect(d.confidence, inInclusiveRange(0, 1));
      expect(d.bestKey, 'billing');
      // ignore: avoid_print
      print('team: $d  ${d.probabilities.map((p) => p.toStringAsFixed(3)).toList()}');
    });

    test('score is deterministic', () async {
      final q = KilnQuestion.score(
          'How urgent is this?', ['Can wait', 'Today', 'Blocking the customer now'], id: 'urgency');
      final a = await ctx.decide(q, state: state);
      final b = await ctx.decide(q, state: state);
      expect(a.probabilities, hasLength(3));
      expect(sum(a.probabilities), closeTo(1, 1e-3));
      expect(a.value, inInclusiveRange(0, 2));
      expect(b.probabilities, a.probabilities);
      // ignore: avoid_print
      print('urgency: expected level ${a.value.toStringAsFixed(2)} of 0..2, best "${a.bestKey}"');
    });

    test('bad input and wrong call fail cleanly', () async {
      await expectLater(ctx.decide(KilnQuestion.score('How urgent?', ['only one']), state: state),
          throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_INVALID_ARG')));
      expect((await ctx.decide(KilnQuestion.yesNo('Is this a question?'))).probabilities, hasLength(2));
      final gen = ctx.chat([const KilnChatMessage.user('hi')]);
      await expectLater(gen.stats,
          throwsA(isA<KilnException>().having((e) => e.status.name, 'status', 'KILN_ERR_NOT_SUPPORTED')));
      // the context is usable again afterwards
      expect((await ctx.decide(KilnQuestion.yesNo('Is this a question?'))).probabilities, hasLength(2));
    });
  });

  group('EmbeddingGemma 2 embedding model', () {
    late KilnModel m;
    late KilnContext ctx;
    double length(KilnEmbedding e) => KilnEmbedding.cosine(e.vector, e.vector) == 0
        ? 0
        : e.vector.fold<double>(0, (a, x) => a + x * x);

    setUpAll(() async {
      final f = await fetch(gemmaEmbed, gemmaEmbedUrl);
      m = await kiln.loadModel(f.path);
      ctx = await m.createContext(params: const KilnContextParams(nCtx: 256));
    });
    tearDownAll(() async {
      await ctx.dispose();
      await m.dispose();
    });

    test('info says it is an embedding model', () {
      expect(m.info.isEmbeddingModel, isTrue);
      expect(m.info.isDecisionModel, isFalse);
      expect(m.info.nEmbd, 768);
      // ignore: avoid_print
      print('model: ${m.info}');
    });

    test('a text becomes a vector of length 1, the same every time', () async {
      final a = await ctx.embed('The cat sat on the mat.');
      final b = await ctx.embed('The cat sat on the mat.');
      expect(a.dimensions, 768);
      expect(a.vector, hasLength(768));
      expect(length(a), closeTo(1, 1e-4));
      expect(a.tokens, inInclusiveRange(5, 15));
      expect(a.prefixed, isFalse);
      expect(a.mediaTokens, 0);
      expect(b.vector, a.vector);
      // ignore: avoid_print
      print('embed: $a');
    });

    test('search ranks the answer above the trap and the unrelated text', () async {
      final q = await ctx.embed('how do I reset my password', kind: KilnEmbedKind.query);
      expect(q.prefixed, isTrue);
      Future<KilnEmbedding> doc(String t) => ctx.embed(t, kind: KilnEmbedKind.document);
      final right = await doc('Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.');
      final trap = await doc('Passwords should contain at least twelve characters including a symbol.');
      final other = await doc('Soda loaf uses baking soda and buttermilk as the raising agent.');
      final sr = q.similarity(right), st = q.similarity(trap), so = q.similarity(other);
      // ignore: avoid_print
      print('password query: answer ${sr.toStringAsFixed(3)}, trap ${st.toStringAsFixed(3)}, unrelated ${so.toStringAsFixed(3)}');
      expect(sr, greaterThan(st));
      expect(st, greaterThan(so));
    });

    test('options: kind, title, dimensions, normalize', () async {
      const text = 'Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.';
      final raw = await ctx.embed(text);
      final doc = await ctx.embed(text, kind: KilnEmbedKind.document);
      final titled = await ctx.embed(text, kind: KilnEmbedKind.document, title: 'Account help');
      expect(doc.vector, isNot(raw.vector));
      expect(titled.vector, isNot(doc.vector));

      final short = await ctx.embed(text, dimensions: 256);
      expect(short.vector, hasLength(256));
      expect(length(short), closeTo(1, 1e-4));
      // the short vector is the start of the full one, scaled back to length 1
      expect(short.vector[0] / short.vector[1], closeTo(raw.vector[0] / raw.vector[1], 1e-3));

      final unscaled = await ctx.embed(text, normalize: false);
      expect(length(unscaled), greaterThan(4));
      expect(unscaled.similarity(raw), closeTo(1, 1e-4));
    });

    test('bad input and wrong calls fail cleanly', () async {
      Matcher status(String name) => throwsA(isA<KilnException>().having((e) => e.status.name, 'status', name));
      await expectLater(ctx.embed('text', dimensions: 769), status('KILN_ERR_INVALID_ARG'));
      await expectLater(ctx.embed(List.filled(400, 'a long sentence about nothing').join(' ')), status('KILN_ERR_CONTEXT_FULL'));
      await expectLater(ctx.embedMedia([KilnMedia.bytes(redCirclePng)]), status('KILN_ERR_NOT_SUPPORTED')); // no projector yet
      expect(() => ctx.embedMedia(const []), throwsArgumentError);
      await expectLater(ctx.decide(KilnQuestion.yesNo('Is this a question?')), status('KILN_ERR_NOT_SUPPORTED'));
      final gen = ctx.chat([const KilnChatMessage.user('hi')]);
      await expectLater(gen.stats, status('KILN_ERR_NOT_SUPPORTED'));
      // the context is usable again afterwards
      expect((await ctx.embed('still works')).vector, hasLength(768));
    });

    test('images and audio share the space of text', () async {
      if (!mediaTests) {
        markTestSkipped('run with --dart-define=KILN_MEDIA=true (downloads a 555 MB projector)');
        return;
      }
      Matcher status(String name) => throwsA(isA<KilnException>().having((e) => e.status.name, 'status', name));
      await expectLater(m.loadProjector('${dir.path}/nothing.gguf'), status('KILN_ERR_MODEL_LOAD'));
      final p = await fetch(gemmaProjector, gemmaProjectorUrl);
      final info = await m.loadProjector(p.path);
      // ignore: avoid_print
      print(info);
      expect(info.supportsImages, isTrue);
      expect(info.supportsAudio, isTrue);

      // an image given as bytes, and the same image given as a file, are the same vector
      final red = await ctx.embedMedia([KilnMedia.bytes(redCirclePng)]);
      final file = File('${dir.path}/red_circle.png')..writeAsBytesSync(redCirclePng);
      final redFromFile = await ctx.embedMedia([KilnMedia.file(file.path)]);
      final blue = await ctx.embedMedia([KilnMedia.bytes(blueSquarePng)]);
      expect(red.vector, hasLength(768));
      expect(length(red), closeTo(1, 1e-4));
      expect(red.mediaTokens, greaterThan(0));
      expect(red.mediaMs, greaterThan(0));
      expect(redFromFile.vector, red.vector);
      // ignore: avoid_print
      print('image: $red');

      final q = await ctx.embed('a red circle', kind: KilnEmbedKind.query);
      final sRed = q.similarity(red), sBlue = q.similarity(blue);
      // ignore: avoid_print
      print('"a red circle": red circle ${sRed.toStringAsFixed(3)}, blue square ${sBlue.toStringAsFixed(3)}');
      expect(sRed, greaterThan(sBlue));

      final tone = await ctx.embedMedia([KilnMedia.bytes(toneWav())]);
      expect(tone.vector, hasLength(768));
      expect(length(tone), closeTo(1, 1e-4));
      expect(tone.mediaTokens, greaterThan(0));
      // ignore: avoid_print
      print('audio: $tone');

      // text and an image together, with the marker saying where the image goes
      final mixed = await ctx.embedMedia([KilnMedia.bytes(redCirclePng)], text: 'A drawing ${KilnMedia.marker} on white paper');
      expect(mixed.tokens, greaterThan(red.tokens));
      expect(mixed.vector, isNot(red.vector));

      await expectLater(ctx.embedMedia([KilnMedia.bytes(redCirclePng), KilnMedia.bytes(blueSquarePng)], text: 'one ${KilnMedia.marker} marker'),
          status('KILN_ERR_INVALID_ARG'));
      await expectLater(ctx.embedMedia([KilnMedia.bytes(Uint8List.fromList(List.filled(64, 7)))]), status('KILN_ERR_INVALID_ARG'));
      await expectLater(m.loadProjector(p.path), status('KILN_ERR_INVALID_ARG')); // already loaded
    }, timeout: const Timeout(Duration(minutes: 15)));
  });

  group('speech to text (whisper.cpp)', () {
    Future<void> check(KilnModel m, {required String language}) async {
      expect(m.info.isSpeechModel, isTrue);
      expect(m.info.isDecisionModel, isFalse);
      expect(m.info.isEmbeddingModel, isFalse);
      expect(m.info.backendName, startsWith('whisper.cpp/'));
      // ignore: avoid_print
      print('model: ${m.info}');
      final ctx = await m.createContext();
      try {
        final file = File('${dir.path}/speech_train.wav')..writeAsBytesSync(speechTrainWav);
        final t = ctx.transcribe(KilnMedia.file(file.path));
        final pieces = <String>[];
        t.pieces.listen(pieces.add);
        final st = await t.stats;
        final text = pieces.join();
        // ignore: avoid_print
        print('transcribe: "$text" ($st)');
        expect(words(text), 'the train to london leaves at nine in the morning');
        expect(text, isNot(startsWith(' ')));
        expect(st.audioSeconds, inInclusiveRange(3.0, 3.3));
        expect(st.generatedTokens, greaterThan(5));
        expect(st.encodeMs, greaterThan(0));
        expect(st.ms, greaterThan(st.encodeMs));
        expect(st.stoppedByEos, isTrue);
        expect(st.stoppedByUser, isFalse);
        expect(st.language, language);
        // the bytes of the file give the same text; a second run too
        expect(await ctx.transcribe(KilnMedia.bytes(speechTrainWav)).text, text);
        expect(await ctx.transcribe(KilnMedia.file(file.path)).text, text);
        // what these models do not do
        Matcher status(String name) => throwsA(isA<KilnException>().having((e) => e.status.name, 'status', name));
        await expectLater(ctx.chat([const KilnChatMessage('user', 'Hi')]).stats, status('KILN_ERR_NOT_SUPPORTED'));
        await expectLater(ctx.embed('text'), status('KILN_ERR_NOT_SUPPORTED'));
        await expectLater(m.loadProjector('${dir.path}/nothing.gguf'), status('KILN_ERR_NOT_SUPPORTED'));
        await expectLater(ctx.transcribe(KilnMedia.bytes(redCirclePng)).stats, status('KILN_ERR_INVALID_ARG'));
        await expectLater(ctx.transcribe(KilnMedia.file('${dir.path}/nothing.wav')).stats, status('KILN_ERR_INVALID_ARG'));
      } finally {
        await ctx.dispose();
      }
    }

    test('Whisper base writes down a clip', () async {
      final f = await fetch(whisperBase, whisperBaseUrl);
      final m = await kiln.loadModel(f.path);
      try {
        await check(m, language: 'English');
      } finally {
        await m.dispose();
      }
    }, timeout: const Timeout(Duration(minutes: 10)));

    test('Pocket TTS speaks, and Parakeet hears the words (KILN_SPEECH)', () async {
      if (!speechTests) {
        markTestSkipped('run with --dart-define=KILN_SPEECH=true (downloads 220 MB of Pocket TTS and 669 MB of Parakeet)');
        return;
      }
      final f = await fetch(pocket, pocketUrl);
      final m = await kiln.loadModel(f.path);
      try {
        expect(m.info.isSpeechModel, isFalse);
        final ctx = await m.createContext();
        final voice = File('${dir.path}/speech_train.wav')..writeAsBytesSync(speechTrainWav);
        const text = 'The train to London leaves at nine in the morning.';
        Matcher status(String name) => throwsA(isA<KilnException>().having((e) => e.status.name, 'status', name));
        await expectLater(ctx.speak(text, voice: KilnMedia.file(voice.path)).stats, status('KILN_ERR_NOT_SUPPORTED'));
        final p = await fetch(pocketProjector, pocketProjectorUrl);
        final info = await m.loadProjector(p.path);
        // ignore: avoid_print
        print(info);
        expect(info.generatesAudio, isTrue);
        expect(info.sampleRate, 24000);

        final s = ctx.speak(text, voice: KilnMedia.file(voice.path), seed: 7);
        final pieces = <KilnAudioPiece>[];
        final wavFuture = s.pieces.map((p) { pieces.add(p); return p; }).toList();
        final st = await s.stats;
        await wavFuture;
        // ignore: avoid_print
        print('speak: $st, ${pieces.length} pieces');
        expect(pieces.length, greaterThanOrEqualTo(2));
        expect(st.sampleRate, 24000);
        expect(st.audioSeconds, inInclusiveRange(1.5, 8.0));
        expect(st.frames, greaterThan(20));
        expect(st.stoppedByEos, isTrue);
        expect(st.stoppedByUser, isFalse);
        final all = Float32List.fromList([for (final p in pieces) ...p.samples]);
        expect(all.length / 24000, closeTo(st.audioSeconds, 0.01));
        expect(all.map((v) => v.abs()).reduce((a, b) => a > b ? a : b), greaterThan(0.05));
        // the same seed gives audio of about the same length (sample-exact on the Mac and the
        // Pixel 8 in the C suite, but a Linux run differed by three frames, so not demanded here)
        final again = await ctx.speak(text, voice: KilnMedia.file(voice.path), seed: 7).wav;
        expect((again.length - 44) ~/ 2, inInclusiveRange((all.length * 0.75).round(), (all.length * 1.25).round()));
        // what it does not take
        await expectLater(ctx.speak('').stats, status('KILN_ERR_INVALID_ARG'));
        await expectLater(ctx.speak(text, voice: KilnMedia.bytes(redCirclePng)).stats, status('KILN_ERR_NOT_SUPPORTED'));   // read as an image; the projector takes audio only
        await ctx.dispose();

        // round trip: Parakeet reads the generated speech back
        final wav = pcmToWav(all, 24000);
        final said = File('${dir.path}/said.wav')..writeAsBytesSync(wav);
        final asr = await kiln.loadModel((await fetch(parakeet, parakeetUrl)).path);
        try {
          final actx = await asr.createContext();
          final heard = await actx.transcribe(KilnMedia.file(said.path)).text;
          // ignore: avoid_print
          print('Parakeet heard: "$heard"');
          expect(words(heard), 'the train to london leaves at nine in the morning');
          await actx.dispose();
        } finally {
          await asr.dispose();
        }
      } finally {
        await m.dispose();
      }
    }, timeout: const Timeout(Duration(minutes: 25)));

    test('Parakeet TDT 0.6B v3 writes down a clip', () async {
      if (!speechTests) {
        markTestSkipped('run with --dart-define=KILN_SPEECH=true (downloads a 669 MB model)');
        return;
      }
      final f = await fetch(parakeet, parakeetUrl);
      final m = await kiln.loadModel(f.path);
      try {
        await check(m, language: '');
      } finally {
        await m.dispose();
      }
    }, timeout: const Timeout(Duration(minutes: 20)));
  });
}
