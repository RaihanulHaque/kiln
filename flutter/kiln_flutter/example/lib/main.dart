// kiln example: a small chat app. Two views in one page: the model library (pick or download
// a model) and the chat. Everything kiln-specific is in _HomePageState; the rest is layout.
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:kiln_flutter/kiln_flutter.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';

void main() => runApp(const KilnApp());

// ---------------------------------------------------------------- theme
// One accent (the ember of the kiln logo) on warm neutrals. Light and dark follow the system.

ThemeData _theme(Brightness b) {
  final dark = b == Brightness.dark;
  final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFFE8590C), brightness: b).copyWith(
    primary: dark ? const Color(0xFFFF8A4C) : const Color(0xFFC2410C),
    onPrimary: dark ? const Color(0xFF2A1004) : Colors.white,
    primaryContainer: dark ? const Color(0xFF45220F) : const Color(0xFFFFE2D1),
    onPrimaryContainer: dark ? const Color(0xFFFFDCC8) : const Color(0xFF4A1C06),
    secondaryContainer: dark ? const Color(0xFF3A2A20) : const Color(0xFFF6E3D6),
    onSecondaryContainer: dark ? const Color(0xFFFFDCC8) : const Color(0xFF4A1C06),
    surface: dark ? const Color(0xFF12100F) : const Color(0xFFFBF8F5),
    surfaceContainerLow: dark ? const Color(0xFF191715) : const Color(0xFFF5F0EB),
    surfaceContainer: dark ? const Color(0xFF1F1C19) : const Color(0xFFEFE9E3),
    surfaceContainerHigh: dark ? const Color(0xFF27231F) : const Color(0xFFE8E1DA),
    surfaceContainerHighest: dark ? const Color(0xFF312C27) : const Color(0xFFE0D8D0),
    onSurface: dark ? const Color(0xFFEDE6E0) : const Color(0xFF1F1A17),
    onSurfaceVariant: dark ? const Color(0xFFA69C94) : const Color(0xFF6B625B),
    outlineVariant: dark ? const Color(0xFF37312D) : const Color(0xFFDDD4CC),
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: scheme.surface,
    appBarTheme: AppBarTheme(
      backgroundColor: scheme.surface,
      surfaceTintColor: Colors.transparent,
      scrolledUnderElevation: 0,
      centerTitle: false,
    ),
    dividerTheme: DividerThemeData(color: scheme.outlineVariant, space: 1, thickness: 1),
    snackBarTheme: const SnackBarThemeData(behavior: SnackBarBehavior.floating),
  );
}

class KilnApp extends StatelessWidget {
  const KilnApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'kiln',
        debugShowCheckedModeBanner: false,
        theme: _theme(Brightness.light),
        darkTheme: _theme(Brightness.dark),
        // follows the system; --dart-define=KILN_THEME=light|dark forces one (for screenshots)
        themeMode: switch (const String.fromEnvironment('KILN_THEME')) {
          'light' => ThemeMode.light,
          'dark' => ThemeMode.dark,
          _ => ThemeMode.system,
        },
        home: const HomePage(),
      );
}

// ---------------------------------------------------------------- data

enum _Kind { chat, decision, embedding, speech, tts }

/// A model the app offers to download. All three are small enough for a mid-range phone.
class _CatalogModel {
  const _CatalogModel(this.name, this.blurb, this.megabytes, this.url,
      [this.alias, this.kind = _Kind.chat, this.projectorUrl, this.projectorFile, this.projectorMegabytes = 0]);

  /// Embedding models: where to get the image / audio encoders (the "mmproj" file), if any.
  final String? projectorUrl;

  /// The name the projector is saved under. Projector files of different models often share
  /// one name upstream ("mmproj-Q8_0.gguf"), so each gets its own here.
  final String? projectorFile;
  final int projectorMegabytes;

  /// Which screen the model opens: chat, Decide, Search or Transcribe.
  final _Kind kind;
  final String name, blurb, url;
  final int megabytes;

  /// Another file name the same model is known by (the name the repo's test scripts use).
  final String? alias;
  String get fileName => url.split('/').last;
  bool matches(String file) => file == fileName || file == alias;
}

const _catalog = [
  _CatalogModel('LFM2.5 350M', 'Fastest. Good for short answers.', 229,
      'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf',
      'LiquidAI_LFM2.5-350M_Q4_K_M.gguf'),
  _CatalogModel('Gemma 3 270M', 'Tiny and quick.', 292,
      'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf'),
  _CatalogModel('Qwen3.5 0.8B', 'Smartest of the chat models. Can think before answering.', 533,
      'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf'),
  // Not a chat model: it opens the Decide screen. Weights are under Liquid AI's lfm1.0 license.
  _CatalogModel('LFM2 d1-omni 600M', 'Decision model. Judges a text instead of chatting.', 407,
      'https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/d1-omni-600M-Q8_0.gguf', null, _Kind.decision),
  // Not a chat model either: it opens the Search screen. Apache 2.0.
  _CatalogModel('EmbeddingGemma 2', 'Embedding model. Finds notes by meaning.', 176,
      'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf',
      null, _Kind.embedding,
      'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/mmproj-Q8_0.gguf',
      'mmproj-embeddinggemma-2-Q8_0.gguf', 555),
  // Speech models, in whisper.cpp's file format: they open the Transcribe screen. The encoder
  // is built in, so no projector. Parakeet: CC-BY-4.0 (NVIDIA). Whisper: MIT (OpenAI).
  _CatalogModel('Parakeet TDT 0.6B v3', 'Speech to text. The most accurate small model; 25 European languages.', 669,
      'https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-q8_0.bin', null, _Kind.speech),
  _CatalogModel('Whisper base', 'Speech to text in 99 languages. Small and quick, less exact.', 60,
      'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin', null, _Kind.speech),
  // Text to speech: a small text model plus its projector, like the embedding media models.
  // Kyutai's Pocket TTS, English, CC-BY-4.0. It opens the Speak screen.
  _CatalogModel('Pocket TTS', 'Text to speech in English, in the voice of a short recording.', 159,
      'https://huggingface.co/EryriLabs/pocket-tts-GGUF/resolve/main/pocket-tts-en.gguf', null, _Kind.tts,
      'https://huggingface.co/EryriLabs/pocket-tts-GGUF/resolve/main/mmproj-pocket-tts-en.gguf',
      'mmproj-pocket-tts-en.gguf', 60),
];

const _sampleSentences = [
  'The train to London leaves at nine in the morning.',
  'Your parcel will arrive on Thursday, between two and four.',
  'Knead the dough and let it rise for an hour before baking.',
];

const _audioExtensions = ['wav', 'mp3', 'flac'];

/// One audio clip on the Transcribe screen: recorded here or picked from the device.
class _Clip {
  _Clip(this.path, this.name);
  final String path, name;
  String text = '';
  String info = 'listening…';
  bool done = false, failed = false;
}

/// Ready-made questions for the Decide screen, one of each kind.
class _Preset {
  const _Preset(this.label, this.type, this.question, [this.options = '']);
  final String label, question, options;
  final KilnQuestionType type;
}

const _sampleState =
    'Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and nobody '
    'answered. I want my money back today or I will dispute the charge with my bank.';

/// Notes for the Search screen, and questions that share almost no words with their answer.
const _sampleNotes = '''
Knead the dough and let it rise for an hour before baking.
The train to Leeds leaves from platform 4 at 9:15.
To get back into your account, open Settings, choose Security, then Forgot credentials.
Water the tomatoes every second day and keep them in full sun.
The dentist moved my appointment to Thursday afternoon.
Ahmed owes me 40 for the concert tickets.
The wifi code for the flat is written under the router.
Replace the bike chain when it starts skipping gears.''';

const _mediaExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'wav', 'mp3', 'flac'];
const _imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif'];

/// A photo or an audio clip the user added to the Search screen.
class _MediaItem {
  _MediaItem(this.path, this.name);
  final String path, name;
  Float32List? vector; // null while it is being embedded, or if that failed
  String info = 'reading…';
  bool failed = false;
  bool get isImage => _imageExtensions.contains(name.split('.').last.toLowerCase());
}

/// One line of a search result: a note, or a media item.
class _Hit {
  const _Hit(this.label, this.score, [this.media]);
  final String label;
  final double score;
  final _MediaItem? media;
}

const _sampleQueries = ['how do I make bread', 'I forgot my password', 'who needs to pay me back', 'when is my trip'];

const _presets = [
  _Preset('Refund?', KilnQuestionType.yesNo, 'Is the customer asking for a refund?'),
  _Preset('Which team?', KilnQuestionType.choice, 'Which team should handle this?',
      'billing: Payments, refunds and invoices\nshipping: Delivery status and lost parcels\n'
      'technical: The product does not work as expected'),
  _Preset('How urgent?', KilnQuestionType.score, 'How urgent is this?',
      'Can wait\nToday\nBlocking the customer now'),
];

const _suggestions = [
  'Why is the sky blue? One sentence.',
  'What is 17 × 23?',
  'Write a two-line poem about a kiln.',
  'Give me three names for a pottery studio.',
];

class _Turn {
  _Turn(this.role, this.text);
  final String role;
  String text;
  String reasoning = '';
  String? error;
  KilnGenStats? stats;
  bool showReasoning = false;
}

String _fileName(File f) => f.uri.pathSegments.last;
String _prettyName(String file) {
  for (final m in _catalog) {
    if (m.matches(file)) return m.name;
  }
  return file.replaceAll('.gguf', '').replaceAll('.bin', '');
}
String _size(num bytes) =>
    bytes >= 1e9 ? '${(bytes / 1e9).toStringAsFixed(2)} GB' : '${(bytes / 1e6).round()} MB';

// ---------------------------------------------------------------- page

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Kiln? _kiln;
  KilnDeviceInfo? _device;
  KilnModel? _model;
  KilnContext? _ctx;
  KilnGeneration? _gen;
  Directory? _modelDir;
  List<File> _models = [];
  String? _loadedPath;
  String? _loadingPath;
  bool _showLibrary = true;
  bool _thinking = false;

  // one download at a time
  String? _downloading; // file name
  double? _progress; // null = size unknown
  http.Client? _downloadClient;

  final _turns = <_Turn>[];
  final _input = TextEditingController();
  final _url = TextEditingController();
  final _scroll = ScrollController();

  // Decide screen (shown instead of the chat when the loaded model is a decision model)
  final _stateText = TextEditingController(text: _sampleState);
  final _question = TextEditingController(text: _presets[0].question);
  final _options = TextEditingController(text: _presets[0].options);
  KilnQuestionType _qType = _presets[0].type;
  KilnDecision? _decision;
  bool _deciding = false;
  final _decideScroll = ScrollController();

  bool get _isDecision => _model?.info.isDecisionModel ?? false;

  // Search screen (shown when the loaded model is an embedding model)
  final _notes = TextEditingController(text: _sampleNotes);
  final _query = TextEditingController(text: _sampleQueries[0]);
  final _noteVectors = <String, Float32List>{}; // note text -> its vector, so a note is embedded once
  List<_Hit>? _hits; // best first
  final _media = <_MediaItem>[];
  KilnProjectorInfo? _projector; // set once the image / audio encoders are loaded
  bool _loadingProjector = false;
  String _searchInfo = '';
  bool _searching = false;
  final _searchScroll = ScrollController();

  bool get _isEmbedding => _model?.info.isEmbeddingModel ?? false;

  // Transcribe screen (shown when the loaded model is a speech model: Whisper or Parakeet)
  final _clips = <_Clip>[];
  final _recorder = AudioRecorder();
  bool _recording = false;
  bool _transcribing = false;
  final _transcribeScroll = ScrollController();
  bool get _isSpeech => _model?.info.isSpeechModel ?? false;

  // Speak screen (shown when the loaded model is a text-to-speech model from the catalog)
  final _sayText = TextEditingController(text: _sampleSentences[0]);
  String? _voicePath;          // the voice to imitate: a recording, a picked file, or the bundled sample
  String _voiceLabel = 'sample voice';
  bool _speaking = false;
  bool _recordingVoice = false;
  String _speakInfo = '';
  String? _lastWavPath;
  final _player = AudioPlayer();
  bool get _isTts => _loadedCatalog?.kind == _Kind.tts;

  bool get _inChat => _ctx != null && !_showLibrary;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    _downloadClient?.close();
    _kiln?.dispose();
    _input.dispose();
    _url.dispose();
    _scroll.dispose();
    _stateText.dispose();
    _question.dispose();
    _options.dispose();
    _decideScroll.dispose();
    _notes.dispose();
    _query.dispose();
    _searchScroll.dispose();
    _transcribeScroll.dispose();
    _recorder.dispose();
    _sayText.dispose();
    _player.dispose();
    super.dispose();
  }

  Future<void> _init() async {
    try {
      _kiln = await Kiln.create();
      _device = await _kiln!.probeDevice();
      // App-specific external dir on Android: no permission needed, and reachable with
      //   adb push model.gguf /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
      _modelDir = (Platform.isAndroid ? await getExternalStorageDirectory() : null) ??
          await getApplicationDocumentsDirectory();
      _modelDir!.createSync(recursive: true);
      _refreshModels();
    } catch (e) {
      _toast('Could not start kiln: $e');
    }
  }

  void _refreshModels() {
    final dir = _modelDir;
    if (dir == null) return;
    final files = dir
        .listSync()
        .whereType<File>()
        // a projector (mmproj) is an add-on to an embedding model, not a model to open;
        // .bin is whisper.cpp's format (Whisper, Parakeet)
        .where((f) => (f.path.endsWith('.gguf') || f.path.endsWith('.bin')) && !_fileName(f).startsWith('mmproj'))
        .toList()
      ..sort((a, b) => _prettyName(_fileName(a)).toLowerCase().compareTo(_prettyName(_fileName(b)).toLowerCase()));
    if (mounted) setState(() => _models = files);
  }

  void _toast(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  // ---- models

  /// Streams to a .part file and renames at the end, so a dropped connection never leaves a
  /// truncated file that would later be loaded as a model.
  Future<void> _download(String url, {String? saveAs}) async {
    final uri = Uri.tryParse(url.trim());
    if (uri == null || !uri.hasScheme || !(uri.path.endsWith('.gguf') || uri.path.endsWith('.bin'))) {
      _toast('That does not look like a link to a .gguf (or whisper.cpp .bin) file.');
      return;
    }
    final name = saveAs ?? uri.pathSegments.last;
    final out = File('${_modelDir!.path}/$name');
    final part = File('${out.path}.part');
    final client = http.Client();
    setState(() { _downloading = name; _progress = null; _downloadClient = client; });
    try {
      final resp = await client.send(http.Request('GET', uri));
      if (resp.statusCode != 200) throw HttpException('server answered ${resp.statusCode}');
      final total = resp.contentLength ?? 0;
      var got = 0, shown = 0;
      final sink = part.openWrite();
      try {
        await for (final chunk in resp.stream) {
          sink.add(chunk);
          got += chunk.length;
          // repaint about once per percent, not once per chunk
          if (total > 0 && got - shown > total / 100) {
            shown = got;
            if (mounted) setState(() => _progress = got / total);
          }
        }
      } finally {
        await sink.close();
      }
      if (total > 0 && got != total) throw const HttpException('connection dropped');
      part.renameSync(out.path);
      _url.clear();
    } catch (e) {
      if (part.existsSync()) part.deleteSync();
      // closing the client (Cancel) also lands here; only report real failures
      if (_downloadClient == client) _toast('Download failed: $e');
    } finally {
      client.close();
      if (mounted) setState(() { _downloading = null; _progress = null; _downloadClient = null; });
      _refreshModels();
    }
  }

  void _cancelDownload() {
    final c = _downloadClient;
    _downloadClient = null; // marks the failure in _download as a cancel
    c?.close();
  }

  Future<void> _load(File f) async {
    if (_loadingPath != null || _gen != null) return;
    if (f.path == _loadedPath) {
      _openChat();
      return;
    }
    setState(() => _loadingPath = f.path);
    try {
      await _unload();
      final model = await _kiln!.loadModel(f.path);
      _model = model;
      // A decision model reads one short prompt per question, so the default context is enough.
      _ctx = await model.createContext(
          params: model.info.isDecisionModel || model.info.isEmbeddingModel || model.info.isSpeechModel
              ? const KilnContextParams()
              : const KilnContextParams(nCtx: 4096));
      _decision = null;
      _hits = null;
      _noteVectors.clear(); // vectors of one model mean nothing to another
      _media.clear();
      _clips.clear();
      _speakInfo = '';
      _lastWavPath = null;
      _projector = null; // the encoders belong to the model and went away with it
      _loadedPath = f.path;
      _thinking = false;
      _showLibrary = false;
    } catch (e) {
      await _unload();
      _toast('Could not load ${_fileName(f)}: ${e is KilnException ? e.message : e}');
    } finally {
      if (mounted) setState(() => _loadingPath = null);
    }
  }

  /// The chat list is rebuilt when coming back from the library, so put it at the latest message.
  void _openChat() {
    setState(() => _showLibrary = false);
    _scrollDown(force: true);
  }

  Future<void> _unload() async {
    final ctx = _ctx, model = _model;
    _ctx = null;
    _model = null;
    _loadedPath = null;
    _turns.clear();
    await ctx?.dispose();
    await model?.dispose();
  }

  Future<void> _confirmDelete(File f) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Delete model?'),
        content: Text('${_prettyName(_fileName(f))} (${_size(f.lengthSync())}) will be removed from this device.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;
    if (f.path == _loadedPath) await _unload();
    if (f.existsSync()) f.deleteSync();
    _refreshModels();
  }

  // ---- chat

  Future<void> _send([String? preset]) async {
    final text = (preset ?? _input.text).trim();
    final ctx = _ctx;
    if (text.isEmpty || ctx == null || _gen != null) return;
    _input.clear();
    final reply = _Turn('assistant', '');
    final history = [
      for (final t in _turns)
        if (t.error == null) KilnChatMessage(t.role, t.text),
      KilnChatMessage.user(text),
    ];
    // 512-token thinking budget: small models such as Qwen3.5 0.8B rarely stop thinking by themselves
    final think = _thinking && _model!.info.supportsThinking;
    final gen = ctx.chat(history,
        enableThinking: think,
        maxReasoningTokens: 512,
        // With thinking on, the repeat penalty makes Qwen3.5 0.8B loop in its answer more often
        // (measured in the thinking-budget work), so it is switched off for those turns.
        sampling: KilnSamplingParams(repeatPenalty: think ? 1.0 : 1.1, maxTokens: 2048));
    setState(() {
      _turns..add(_Turn('user', text))..add(reply);
      _gen = gen;
    });
    _scrollDown(force: true);
    gen.pieces.listen((p) {
      setState(() {
        if (p.isReasoning) {
          reply.reasoning += p.text;
          reply.showReasoning = true;
        } else {
          if (reply.text.isEmpty) reply.showReasoning = false; // fold the thinking once the answer starts
          reply.text += p.text;
        }
      });
      _scrollDown();
    }, onError: (_) {}); // the same error arrives through gen.stats below
    try {
      reply.stats = await gen.stats;
    } catch (e) {
      reply.error = e is KilnException ? e.message : '$e';
    }
    if (mounted) setState(() => _gen = null);
    _scrollDown();
  }

  Future<void> _newChat() async {
    _gen?.cancel();
    try {
      await _ctx?.reset();
    } catch (_) {/* a running generation is being cancelled; the next chat re-reads the prompt */}
    if (mounted) setState(_turns.clear);
  }

  /// Follows the stream only while the user is already at the bottom, so scrolling up to
  /// re-read is not fought by new tokens.
  void _scrollDown({bool force = false}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      final pos = _scroll.position;
      if (force || pos.maxScrollExtent - pos.pixels < 120) _scroll.jumpTo(pos.maxScrollExtent);
    });
  }

  // ---------------------------------------------------------------- build

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return PopScope(
      canPop: !_inChat,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) setState(() => _showLibrary = true);
      },
      child: Scaffold(
        appBar: _inChat ? _chatBar(scheme) : _libraryBar(scheme),
        body: SafeArea(
            child: !_inChat
                ? _libraryView(scheme)
                : _isDecision
                    ? _decideView(scheme)
                    : _isEmbedding
                        ? _searchView(scheme)
                        : _isSpeech
                            ? _transcribeView(scheme)
                            : _isTts
                                ? _speakView(scheme)
                                : _chatView(scheme)),
      ),
    );
  }

  // ---- library view

  PreferredSizeWidget _libraryBar(ColorScheme scheme) => AppBar(
        titleSpacing: 20,
        title: Row(children: [
          Icon(Icons.local_fire_department_rounded, color: scheme.primary, size: 26),
          const SizedBox(width: 8),
          const Text('kiln', style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: -0.5)),
        ]),
        actions: [
          if (_ctx != null)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: FilledButton.tonalIcon(
                onPressed: _openChat,
                icon: const Icon(Icons.chat_bubble_outline_rounded, size: 18),
                label: Text(_isDecision || _isEmbedding || _isSpeech || _isTts ? 'Back to model' : 'Back to chat'),
              ),
            ),
        ],
      );

  Widget _libraryView(ColorScheme scheme) {
    final have = [for (final f in _models) _fileName(f)];
    final toGet = [for (final m in _catalog) if (!have.any(m.matches)) m];
    return ListView(padding: const EdgeInsets.fromLTRB(20, 4, 20, 32), children: [
      _deviceCard(scheme),
      _sectionTitle('On this device', scheme),
      if (_modelDir == null)
        const Padding(padding: EdgeInsets.all(24), child: Center(child: CircularProgressIndicator()))
      else if (_models.isEmpty)
        _hint('No models yet. Download one below to start chatting.', scheme)
      else
        _card(scheme, [for (final f in _models) _localTile(f, scheme)]),
      if (toGet.isNotEmpty || _downloading != null) ...[
        _sectionTitle('Download', scheme),
        _card(scheme, [for (final m in toGet) _catalogTile(m, scheme)]),
      ],
      _sectionTitle('Any other GGUF', scheme),
      _customUrl(scheme),
    ]);
  }

  Widget _sectionTitle(String s, ColorScheme scheme) => Padding(
        padding: const EdgeInsets.fromLTRB(4, 28, 4, 10),
        child: Text(s.toUpperCase(),
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, letterSpacing: 1.1, color: scheme.onSurfaceVariant)),
      );

  Widget _hint(String s, ColorScheme scheme) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4),
        child: Text(s, style: TextStyle(color: scheme.onSurfaceVariant, height: 1.4)),
      );

  /// A rounded group of rows separated by hairlines.
  Widget _card(ColorScheme scheme, List<Widget> rows) => Material(
        color: scheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(18),
        clipBehavior: Clip.antiAlias,
        child: Column(children: [
          for (var i = 0; i < rows.length; i++) ...[
            if (i > 0) const Divider(indent: 16, endIndent: 16),
            rows[i],
          ],
        ]),
      );

  Widget _deviceCard(ColorScheme scheme) {
    final d = _device;
    if (d == null) return const SizedBox(height: 8);
    final features = [
      if (d.hasFp16) 'fp16',
      if (d.hasDotprod) 'dotprod',
      if (d.hasI8mm) 'i8mm',
      if (d.hasSve) 'sve',
      if (d.hasGpu) d.gpuName.isEmpty ? 'GPU' : d.gpuName,
    ];
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('THIS DEVICE',
            style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.1, color: scheme.onSurfaceVariant)),
        const SizedBox(height: 6),
        Text(d.soc.isEmpty ? d.abi : d.soc, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600)),
        const SizedBox(height: 2),
        Text('${d.cpuCores} cores, ${d.cpuFastCores} fast  ·  ${_size(d.ramTotalBytes)} RAM  ·  ${d.abi}',
            style: TextStyle(color: scheme.onSurfaceVariant)),
        if (features.isNotEmpty) ...[
          const SizedBox(height: 12),
          Wrap(spacing: 6, runSpacing: 6, children: [
            for (final f in features)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                decoration: BoxDecoration(color: scheme.surfaceContainerHigh, borderRadius: BorderRadius.circular(8)),
                child: Text(f, style: TextStyle(fontSize: 12, fontFamily: 'monospace', color: scheme.onSurfaceVariant)),
              ),
          ]),
        ],
      ]),
    );
  }

  Widget _localTile(File f, ColorScheme scheme) {
    final loaded = f.path == _loadedPath, loading = f.path == _loadingPath;
    return ListTile(
      contentPadding: const EdgeInsets.fromLTRB(16, 4, 4, 4),
      enabled: _loadingPath == null,
      onTap: () => _load(f),
      title: Text(_prettyName(_fileName(f)), maxLines: 1, overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.w600)),
      subtitle: Text(
        loading
            ? 'Loading…'
            : [
                _size(f.lengthSync()),
                if (_catalog.any((m) => m.kind == _Kind.decision && m.matches(_fileName(f)))) 'decision model',
                if (_catalog.any((m) => m.kind == _Kind.embedding && m.matches(_fileName(f)))) 'embedding model',
                if (_fileName(f).endsWith('.bin')) 'speech model',
                if (_catalog.any((m) => m.kind == _Kind.tts && m.matches(_fileName(f)))) 'text to speech',
                if (loaded) 'loaded',
              ].join('  ·  '),
        style: TextStyle(color: loaded || loading ? scheme.primary : scheme.onSurfaceVariant),
      ),
      trailing: loading
          ? const Padding(
              padding: EdgeInsets.all(14),
              child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.4)))
          : IconButton(
              tooltip: 'Delete',
              icon: Icon(Icons.delete_outline_rounded, color: scheme.onSurfaceVariant),
              onPressed: _loadingPath == null && _gen == null ? () => _confirmDelete(f) : null,
            ),
    );
  }

  Widget _catalogTile(_CatalogModel m, ColorScheme scheme) {
    final active = _downloading == m.fileName;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 12, 12),
      child: Column(children: [
        Row(children: [
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(m.name, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              const SizedBox(height: 2),
              Text('${m.megabytes} MB  ·  ${m.blurb}', style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant)),
            ]),
          ),
          const SizedBox(width: 8),
          active
              ? TextButton(onPressed: _cancelDownload, child: const Text('Cancel'))
              : FilledButton.tonal(
                  onPressed: _downloading == null && _modelDir != null ? () => _download(m.url) : null,
                  child: const Text('Get'),
                ),
        ]),
        if (active) _progressBar(scheme),
      ]),
    );
  }

  Widget _progressBar(ColorScheme scheme) => Padding(
        padding: const EdgeInsets.only(top: 12, right: 4),
        child: Row(children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(value: _progress, minHeight: 6, backgroundColor: scheme.surfaceContainerHighest),
            ),
          ),
          SizedBox(
            width: 48,
            child: Text(_progress == null ? '' : '${(_progress! * 100).round()}%',
                textAlign: TextAlign.right, style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
          ),
        ]),
      );

  Widget _customUrl(ColorScheme scheme) {
    final custom = _downloading != null && !_catalog.any((m) => m.fileName == _downloading);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Expanded(
          child: TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            autocorrect: false,
            style: const TextStyle(fontSize: 14),
            decoration: InputDecoration(
              hintText: 'Paste a link to a .gguf or whisper.cpp .bin file',
              isDense: true,
              filled: true,
              fillColor: scheme.surfaceContainerLow,
              contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none),
            ),
          ),
        ),
        const SizedBox(width: 8),
        custom
            ? TextButton(onPressed: _cancelDownload, child: const Text('Cancel'))
            : FilledButton.tonal(
                onPressed: _downloading == null && _modelDir != null ? () => _download(_url.text) : null,
                child: const Text('Get'),
              ),
      ]),
      if (custom) _progressBar(scheme),
      if (Platform.isAndroid && _modelDir != null)
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 12, 4, 0),
          child: Text('Or copy a file over USB:\nadb push model.gguf ${_modelDir!.path}/',
              style: TextStyle(fontSize: 12, height: 1.5, color: scheme.onSurfaceVariant)),
        ),
    ]);
  }

  // ---- decide view

  void _usePreset(_Preset p) => setState(() {
        _qType = p.type;
        _question.text = p.question;
        _options.text = p.options;
        _decision = null;
      });

  /// Options are typed one per line. For a choice, "key: what it means"; for a score, the
  /// levels from lowest to highest.
  KilnQuestion? _buildQuestion() {
    final q = _question.text.trim();
    if (q.isEmpty) {
      _toast('Type a question first.');
      return null;
    }
    final lines = [for (final l in _options.text.split('\n')) if (l.trim().isNotEmpty) l.trim()];
    switch (_qType) {
      case KilnQuestionType.yesNo:
        return KilnQuestion.yesNo(q);
      case KilnQuestionType.choice:
        if (lines.length < 2) {
          _toast('A choice needs at least two options, one per line.');
          return null;
        }
        return KilnQuestion.choice(q, [
          for (final l in lines)
            l.contains(':')
                ? KilnOption(l.substring(0, l.indexOf(':')).trim(), l.substring(l.indexOf(':') + 1).trim())
                : KilnOption(l),
        ]);
      case KilnQuestionType.score:
        if (lines.length < 2 || lines.length > 10) {
          _toast('A score needs 2 to 10 levels, one per line, lowest first.');
          return null;
        }
        return KilnQuestion.score(q, lines);
    }
  }

  Future<void> _decide() async {
    final q = _buildQuestion();
    final ctx = _ctx;
    if (q == null || ctx == null || _deciding) return;
    FocusScope.of(context).unfocus();
    setState(() => _deciding = true);
    try {
      final text = _stateText.text.trim();
      final d = await ctx.decide(q, state: text.isEmpty ? null : text);
      if (!mounted) return;
      setState(() => _decision = d);
      // bring the answer card into view
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_decideScroll.hasClients) {
          _decideScroll.animateTo(_decideScroll.position.maxScrollExtent,
              duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
        }
      });
    } catch (e) {
      _toast(e is KilnException ? e.message : '$e');
    } finally {
      if (mounted) setState(() => _deciding = false);
    }
  }

  InputDecoration _fieldStyle(ColorScheme scheme, String hint) => InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(color: scheme.onSurfaceVariant),
        isDense: true,
        filled: true,
        fillColor: scheme.surfaceContainerLow,
        contentPadding: const EdgeInsets.all(16),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none),
      );

  Widget _decideView(ColorScheme scheme) {
    void edited(String _) {
      if (_decision != null) setState(() => _decision = null); // the answer no longer matches the inputs
    }

    return ListView(controller: _decideScroll, padding: const EdgeInsets.fromLTRB(20, 4, 20, 32), children: [
      _hint('A decision model does not write text. It reads the text below and scores every '
          'answer to your question in one pass.', scheme),
      _sectionTitle('Text to judge', scheme),
      TextField(
        controller: _stateText,
        minLines: 3,
        maxLines: 8,
        onChanged: edited,
        style: const TextStyle(fontSize: 15, height: 1.4),
        decoration: _fieldStyle(scheme, 'A message, a review, a note…'),
      ),
      _sectionTitle('Question', scheme),
      Wrap(spacing: 8, runSpacing: 4, children: [
        for (final p in _presets)
          ActionChip(
            label: Text(p.label),
            side: BorderSide(color: scheme.outlineVariant),
            shape: const StadiumBorder(),
            onPressed: () => _usePreset(p),
          ),
      ]),
      const SizedBox(height: 12),
      SizedBox(
        width: double.infinity,
        child: SegmentedButton<KilnQuestionType>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: KilnQuestionType.yesNo, label: Text('Yes / no')),
            ButtonSegment(value: KilnQuestionType.choice, label: Text('Choice')),
            ButtonSegment(value: KilnQuestionType.score, label: Text('Score')),
          ],
          selected: {_qType},
          onSelectionChanged: (v) => setState(() {
            _qType = v.first;
            _decision = null;
          }),
        ),
      ),
      const SizedBox(height: 12),
      TextField(
        controller: _question,
        minLines: 1,
        maxLines: 3,
        onChanged: edited,
        textCapitalization: TextCapitalization.sentences,
        decoration: _fieldStyle(scheme, 'Ask something about the text'),
      ),
      if (_qType != KilnQuestionType.yesNo) ...[
        const SizedBox(height: 12),
        TextField(
          controller: _options,
          minLines: 3,
          maxLines: 10,
          onChanged: edited,
          style: const TextStyle(fontSize: 14.5, height: 1.5),
          decoration: _fieldStyle(scheme, ''),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 8, 4, 0),
          child: Text(
              _qType == KilnQuestionType.choice
                  ? 'One option per line. Add what it means after a colon.'
                  : 'One level per line, from lowest to highest (2 to 10).',
              style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
        ),
      ],
      const SizedBox(height: 20),
      FilledButton(
        onPressed: _deciding ? null : _decide,
        style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(52), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        child: _deciding
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.4))
            : const Text('Decide', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
      ),
      if (_decision != null) _decisionCard(_decision!, scheme),
    ]);
  }

  Widget _decisionCard(KilnDecision d, ColorScheme scheme) {
    final headline = switch (d.type) {
      KilnQuestionType.yesNo => d.yes ? 'Yes' : 'No',
      KilnQuestionType.choice => d.bestKey,
      KilnQuestionType.score => d.bestKey,
    };
    final detail = switch (d.type) {
      KilnQuestionType.score =>
        'level ${d.best + 1} of ${d.keys.length}  ·  average ${(d.value + 1).toStringAsFixed(1)}',
      _ => '${(d.probabilities[d.best] * 100).toStringAsFixed(1)}% sure',
    };
    return Container(
      margin: const EdgeInsets.only(top: 24),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('ANSWER',
            style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.1, color: scheme.onSurfaceVariant)),
        const SizedBox(height: 6),
        Text(headline, style: TextStyle(fontSize: 24, fontWeight: FontWeight.w700, color: scheme.primary)),
        const SizedBox(height: 2),
        Text(detail, style: TextStyle(color: scheme.onSurfaceVariant)),
        const SizedBox(height: 16),
        for (var i = 0; i < d.keys.length; i++)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Expanded(
                  child: Text(d.type == KilnQuestionType.yesNo ? (i == 1 ? 'Yes' : 'No') : d.keys[i],
                      maxLines: 1, overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontWeight: i == d.best ? FontWeight.w600 : FontWeight.w400)),
                ),
                Text('${(d.probabilities[i] * 100).toStringAsFixed(1)}%',
                    style: TextStyle(fontFamily: 'monospace', fontSize: 13, color: scheme.onSurfaceVariant)),
              ]),
              const SizedBox(height: 6),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: d.probabilities[i].clamp(0.0, 1.0),
                  minHeight: 8,
                  backgroundColor: scheme.surfaceContainerHigh,
                  color: i == d.best ? scheme.primary : scheme.onSurfaceVariant.withValues(alpha: 0.45),
                ),
              ),
            ]),
          ),
        Text('${d.promptTokens} tokens read  ·  ${d.ms.round()} ms',
            style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
      ]),
    );
  }

  // ---- search view

  /// Embeds every note that has no vector yet, embeds the query, and sorts the notes by how
  /// close they are to it. Notes are "documents" and the typed text is a "query": the model
  /// was trained with a different prefix for each, and kiln adds it.
  Future<void> _search([String? preset]) async {
    if (preset != null) _query.text = preset;
    final q = _query.text.trim();
    final ctx = _ctx;
    final notes = [for (final l in _notes.text.split('\n')) if (l.trim().isNotEmpty) l.trim()];
    final ready = [for (final m in _media) if (m.vector != null) m];
    if (ctx == null || _searching) return;
    if (q.isEmpty || (notes.isEmpty && ready.isEmpty)) {
      _toast(q.isEmpty ? 'Type what you are looking for.' : 'Add some notes first, one per line.');
      return;
    }
    FocusScope.of(context).unfocus();
    setState(() => _searching = true);
    try {
      final sw = Stopwatch()..start();
      var fresh = 0;
      for (final n in notes) {
        if (_noteVectors.containsKey(n)) continue;
        _noteVectors[n] = (await ctx.embed(n, kind: KilnEmbedKind.document)).vector;
        fresh++;
      }
      final indexMs = sw.elapsedMilliseconds;
      final qe = await ctx.embed(q, kind: KilnEmbedKind.query);
      // Text, photos and audio are in one space, so they are ranked in one list.
      final hits = [
        for (final n in notes) _Hit(n, KilnEmbedding.cosine(qe.vector, _noteVectors[n]!)),
        for (final m in ready) _Hit(m.name, KilnEmbedding.cosine(qe.vector, m.vector!), m),
      ]..sort((a, b) => b.score.compareTo(a.score));
      if (!mounted) return;
      setState(() {
        _hits = hits;
        _searchInfo = [
          if (fresh > 0) '$fresh notes read in $indexMs ms',
          'query ${qe.ms.round()} ms',
          '${qe.dimensions} numbers each',
        ].join('  ·  ');
      });
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_searchScroll.hasClients) {
          _searchScroll.animateTo(_searchScroll.position.maxScrollExtent,
              duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
        }
      });
    } catch (e) {
      _toast(e is KilnException ? e.message : '$e');
    } finally {
      if (mounted) setState(() => _searching = false);
    }
  }

  // ---- photos and audio on the search screen

  /// The catalog entry of the loaded model, when it is one the app knows.
  _CatalogModel? get _loadedCatalog {
    final file = _loadedPath?.split('/').last;
    for (final m in _catalog) {
      if (file != null && m.matches(file)) return m;
    }
    return null;
  }

  File? get _projectorOnDisk {
    final name = _loadedCatalog?.projectorFile;
    if (name == null || _modelDir == null) return null;
    final f = File('${_modelDir!.path}/$name');
    return f.existsSync() ? f : null;
  }

  /// Loads the model's image / audio encoders. They stay in memory until the model is unloaded.
  Future<void> _enableMedia() async {
    final f = _projectorOnDisk, model = _model;
    if (f == null || model == null || _loadingProjector) return;
    setState(() => _loadingProjector = true);
    try {
      final info = await model.loadProjector(f.path);
      if (mounted) setState(() => _projector = info);
    } catch (e) {
      _toast(e is KilnException ? e.message : '$e');
    } finally {
      if (mounted) setState(() => _loadingProjector = false);
    }
  }

  Future<void> _getProjector() async {
    final m = _loadedCatalog;
    if (m?.projectorUrl == null) return;
    await _download(m!.projectorUrl!, saveAs: m.projectorFile);
    if (_projectorOnDisk != null) await _enableMedia();
  }

  /// Lets the user pick photos and audio clips, then embeds them one after the other. An image
  /// takes a few seconds on a phone, so each row shows its own progress and the screen stays usable.
  Future<void> _addMedia() async {
    final ctx = _ctx;
    if (ctx == null || _projector == null) return;
    final List<PlatformFile> picked;
    try {
      picked = await FilePicker.pickFiles(type: FileType.custom, allowedExtensions: _mediaExtensions);
    } catch (e) {
      _toast('Could not open the file picker: $e');
      return;
    }
    final items = [
      for (final f in picked)
        if (f.path != null && !_media.any((m) => m.path == f.path)) _MediaItem(f.path!, f.name),
    ];
    if (items.isEmpty || !mounted) return;
    setState(() {
      _media.addAll(items);
      _hits = null;
    });
    for (final item in items) {
      if (!identical(ctx, _ctx)) return; // the model was changed meanwhile
      try {
        final e = await ctx.embedMedia([KilnMedia.file(item.path)]);
        item.vector = e.vector;
        item.info = '${e.mediaTokens} tokens  ·  ${(e.ms / 1000).toStringAsFixed(1)} s';
      } catch (e) {
        item.failed = true;
        item.info = e is KilnException ? e.message : '$e';
      }
      if (mounted) setState(() {});
    }
  }

  Widget _thumb(_MediaItem m, ColorScheme scheme, {double size = 44}) => ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: size,
          height: size,
          color: scheme.surfaceContainerHigh,
          child: m.isImage
              ? Image.file(File(m.path), fit: BoxFit.cover, cacheWidth: 160,
                  errorBuilder: (_, _, _) => Icon(Icons.image_outlined, color: scheme.onSurfaceVariant))
              : Icon(Icons.graphic_eq_rounded, color: scheme.onSurfaceVariant, size: size * 0.5),
        ),
      );

  /// The "Photos and audio" block: get the encoders, switch them on, then add files.
  Widget _mediaSection(ColorScheme scheme) {
    final cat = _loadedCatalog;
    if (cat?.projectorUrl == null) {
      return _hint('This model is not in the app\'s list, so the app does not know where its image '
          'and audio encoders are. It searches text only.', scheme);
    }
    if (_projector == null) {
      final onDisk = _projectorOnDisk != null;
      final downloading = _downloading == cat!.projectorFile;
      return Container(
        padding: const EdgeInsets.fromLTRB(16, 14, 12, 14),
        decoration: BoxDecoration(color: scheme.surfaceContainerLow, borderRadius: BorderRadius.circular(18)),
        child: Column(children: [
          Row(children: [
            Expanded(
              child: Text(
                  onDisk
                      ? 'Switch on the image and audio encoders to search photos and voice notes too. They use about 0.6 GB of memory.'
                      : 'To search photos and voice notes too, this model needs its image and audio encoders (${cat.projectorMegabytes} MB).',
                  style: TextStyle(fontSize: 13.5, height: 1.4, color: scheme.onSurfaceVariant)),
            ),
            const SizedBox(width: 10),
            if (downloading)
              TextButton(onPressed: _cancelDownload, child: const Text('Cancel'))
            else if (_loadingProjector)
              const Padding(
                  padding: EdgeInsets.all(12),
                  child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.4)))
            else
              FilledButton.tonal(
                onPressed: _downloading != null ? null : (onDisk ? _enableMedia : _getProjector),
                child: Text(onDisk ? 'Turn on' : 'Get'),
              ),
          ]),
          if (downloading) _progressBar(scheme),
        ]),
      );
    }
    final kinds = [if (_projector!.supportsImages) 'photos', if (_projector!.supportsAudio) 'audio clips'].join(' or ');
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      if (_media.isNotEmpty)
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: _card(scheme, [
            for (final m in _media)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 10, 4, 10),
                child: Row(children: [
                  _thumb(m, scheme),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(m.name, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w600)),
                      Text(m.info,
                          maxLines: 2, overflow: TextOverflow.ellipsis,
                          style: TextStyle(fontSize: 12.5, color: m.failed ? scheme.error : scheme.onSurfaceVariant)),
                    ]),
                  ),
                  if (m.vector == null && !m.failed)
                    const Padding(
                        padding: EdgeInsets.all(14),
                        child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2.2)))
                  else
                    IconButton(
                      tooltip: 'Remove',
                      icon: Icon(Icons.close_rounded, size: 20, color: scheme.onSurfaceVariant),
                      onPressed: () => setState(() {
                        _media.remove(m);
                        _hits = null;
                      }),
                    ),
                ]),
              ),
          ]),
        ),
      OutlinedButton.icon(
        onPressed: _addMedia,
        icon: const Icon(Icons.add_photo_alternate_outlined, size: 20),
        label: Text('Add $kinds'),
        style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(48),
            side: BorderSide(color: scheme.outlineVariant),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
      ),
    ]);
  }

  Widget _searchView(ColorScheme scheme) {
    void edited(String _) {
      if (_hits != null) setState(() => _hits = null); // the ranking no longer matches the inputs
    }

    // Not a ListView: a lazy list only estimates its length while the results are off screen,
    // and scrolling to the end would stop short.
    return SingleChildScrollView(
        controller: _searchScroll,
        padding: const EdgeInsets.fromLTRB(20, 4, 20, 32),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _hint('An embedding model does not write text. It turns each note, photo or audio clip into '
          'numbers, so they can be found by what they mean, even with no words in common.', scheme),
      _sectionTitle('Notes', scheme),
      TextField(
        controller: _notes,
        minLines: 4,
        maxLines: 9,
        onChanged: edited,
        style: const TextStyle(fontSize: 14.5, height: 1.5),
        decoration: _fieldStyle(scheme, 'One note per line'),
      ),
      _sectionTitle('Photos and audio', scheme),
      _mediaSection(scheme),
      _sectionTitle('Find', scheme),
      Wrap(spacing: 8, runSpacing: 4, children: [
        for (final q in _sampleQueries)
          ActionChip(
            label: Text(q),
            side: BorderSide(color: scheme.outlineVariant),
            shape: const StadiumBorder(),
            onPressed: _searching ? null : () => _search(q),
          ),
      ]),
      const SizedBox(height: 12),
      TextField(
        controller: _query,
        onChanged: edited,
        textInputAction: TextInputAction.search,
        onSubmitted: (_) => _search(),
        decoration: _fieldStyle(scheme, 'What are you looking for?'),
      ),
      const SizedBox(height: 20),
      FilledButton(
        onPressed: _searching ? null : _search,
        style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(52), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        child: _searching
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.4))
            : const Text('Search', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
      ),
      if (_hits != null) _hitsCard(_hits!, scheme),
    ]));
  }

  Widget _hitsCard(List<_Hit> hits, ColorScheme scheme) {
    // Scores of unrelated notes sit around 0.5, so the bars show the spread between the
    // weakest and the best note; the number next to each bar is the real similarity.
    final lo = hits.last.score, hi = hits.first.score;
    double bar(double s) => hi - lo < 1e-6 ? 1 : 0.08 + 0.92 * (s - lo) / (hi - lo);
    return Container(
      margin: const EdgeInsets.only(top: 24),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('BEST MATCH',
            style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.1, color: scheme.onSurfaceVariant)),
        const SizedBox(height: 6),
        Row(children: [
          if (hits.first.media != null) ...[_thumb(hits.first.media!, scheme, size: 64), const SizedBox(width: 14)],
          Expanded(
            child: Text(hits.first.label,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, height: 1.3, color: scheme.primary)),
          ),
        ]),
        const SizedBox(height: 16),
        for (var i = 0; i < hits.length; i++)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
                if (hits[i].media != null) ...[_thumb(hits[i].media!, scheme, size: 30), const SizedBox(width: 10)],
                Expanded(
                  child: Text(hits[i].label,
                      maxLines: 2, overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 13.5, fontWeight: i == 0 ? FontWeight.w600 : FontWeight.w400)),
                ),
                const SizedBox(width: 12),
                Text(hits[i].score.toStringAsFixed(3),
                    style: TextStyle(fontFamily: 'monospace', fontSize: 13, color: scheme.onSurfaceVariant)),
              ]),
              const SizedBox(height: 6),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: bar(hits[i].score),
                  minHeight: 6,
                  backgroundColor: scheme.surfaceContainerHigh,
                  color: i == 0 ? scheme.primary : scheme.onSurfaceVariant.withValues(alpha: 0.45),
                ),
              ),
            ]),
          ),
        Text(_searchInfo, style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
      ]),
    );
  }

  // ---- chat view

  PreferredSizeWidget _chatBar(ColorScheme scheme) {
    final info = _model!.info;
    return AppBar(
      titleSpacing: 0,
      leading: IconButton(
        tooltip: 'Models',
        icon: const Icon(Icons.arrow_back_rounded),
        onPressed: () => setState(() => _showLibrary = true),
      ),
      title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(_prettyName(_loadedPath!.split('/').last),
            maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        Text('${info.quant}  ·  ${info.backendName}', style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
      ]),
      actions: [
        if (info.supportsThinking)
          Padding(
            padding: const EdgeInsets.only(right: 4),
            child: FilterChip(
              label: const Text('Think'),
              avatar: Icon(Icons.psychology_alt_outlined, size: 18, color: _thinking ? scheme.onPrimaryContainer : scheme.onSurfaceVariant),
              showCheckmark: false,
              selected: _thinking,
              selectedColor: scheme.primaryContainer,
              side: BorderSide(color: _thinking ? Colors.transparent : scheme.outlineVariant),
              shape: const StadiumBorder(),
              visualDensity: VisualDensity.compact,
              onSelected: (v) => setState(() => _thinking = v),
            ),
          ),
        if (!info.isDecisionModel && !info.isEmbeddingModel)
          IconButton(
            tooltip: 'New chat',
            icon: const Icon(Icons.edit_square),
            onPressed: _turns.isEmpty ? null : _newChat,
          ),
        const SizedBox(width: 4),
      ],
    );
  }

  // ---- transcribe view

  /// Records from the microphone into a 16 kHz wav, the format speech models want, then
  /// writes it down. Tap again to stop.
  Future<void> _toggleRecording() async {
    if (_recording) {
      final path = await _recorder.stop();
      if (mounted) setState(() => _recording = false);
      if (path != null) await _transcribeFile(path, 'Recording ${_clips.length + 1}');
      return;
    }
    if (!await _recorder.hasPermission()) {
      _toast('kiln needs the microphone to record.');
      return;
    }
    final dir = Directory('${_modelDir!.path}/clips')..createSync(recursive: true);
    final path = '${dir.path}/rec_${DateTime.now().millisecondsSinceEpoch}.wav';
    try {
      await _recorder.start(
          const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1), path: path);
      if (mounted) setState(() => _recording = true);
    } catch (e) {
      _toast('Could not record: $e');
    }
  }

  Future<void> _pickClip() async {
    final List<PlatformFile> picked;
    try {
      picked = await FilePicker.pickFiles(type: FileType.custom, allowedExtensions: _audioExtensions);
    } catch (e) {
      _toast('Could not open the file picker: $e');
      return;
    }
    for (final f in picked) {
      if (f.path != null) await _transcribeFile(f.path!, f.name);
    }
  }

  /// One clip at a time: the words stream into the card as the model writes them.
  Future<void> _transcribeFile(String path, String name) async {
    final ctx = _ctx;
    if (ctx == null || _transcribing) return;
    final clip = _Clip(path, name);
    setState(() { _clips.insert(0, clip); _transcribing = true; });
    try {
      final t = ctx.transcribe(KilnMedia.file(path));
      t.pieces.listen((piece) {
        clip.text += piece;
        if (mounted) setState(() {});
      }, onError: (_) {});
      final st = await t.stats;
      clip.info = [
        '${st.audioSeconds.toStringAsFixed(1)} s of audio',
        '${(st.ms / 1000).toStringAsFixed(2)} s',
        '${st.speed.toStringAsFixed(1)}x real time',
        if (st.language.isNotEmpty) st.language,
      ].join('  ·  ');
      if (clip.text.trim().isEmpty) clip.text = '(no speech heard)';
    } catch (e) {
      clip.failed = true;
      clip.info = e is KilnException ? e.message : '$e';
    } finally {
      clip.done = true;
      if (mounted) setState(() => _transcribing = false);
    }
  }

  Widget _transcribeView(ColorScheme scheme) {
    final name = _loadedCatalog?.name ?? _prettyName(_fileName(File(_loadedPath ?? '')));
    return SingleChildScrollView(
        controller: _transcribeScroll,
        padding: const EdgeInsets.fromLTRB(20, 4, 20, 32),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _hint('$name is a speech model: it writes down what is said in a recording, on the phone, '
          'with nothing sent anywhere. Record something, or pick a wav, mp3 or flac file.', scheme),
      const SizedBox(height: 16),
      FilledButton.icon(
        onPressed: _transcribing && !_recording ? null : _toggleRecording,
        icon: Icon(_recording ? Icons.stop_rounded : Icons.mic_rounded),
        label: Text(_recording ? 'Stop and transcribe' : 'Record', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(52),
            backgroundColor: _recording ? scheme.error : null,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
      ),
      const SizedBox(height: 10),
      OutlinedButton.icon(
        onPressed: _transcribing || _recording ? null : _pickClip,
        icon: const Icon(Icons.audio_file_outlined, size: 20),
        label: const Text('Add an audio file'),
        style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(48),
            side: BorderSide(color: scheme.outlineVariant),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
      ),
      for (final c in _clips) _clipCard(c, scheme),
    ]));
  }

  Widget _clipCard(_Clip c, ColorScheme scheme) => Container(
        margin: const EdgeInsets.only(top: 16),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: scheme.outlineVariant),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(Icons.graphic_eq_rounded, size: 18, color: scheme.onSurfaceVariant),
            const SizedBox(width: 8),
            Expanded(
              child: Text(c.name, maxLines: 1, overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, letterSpacing: 0.4, color: scheme.onSurfaceVariant)),
            ),
            if (!c.done) const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
          ]),
          const SizedBox(height: 10),
          Text(c.text.isEmpty && !c.done ? '…' : c.text,
              style: TextStyle(fontSize: 17, height: 1.4, color: c.failed ? scheme.error : scheme.onSurface)),
          const SizedBox(height: 10),
          Text(c.info, style: TextStyle(fontSize: 12, color: c.failed ? scheme.error : scheme.onSurfaceVariant)),
        ]),
      );

  // ---- speak view

  Future<String> _sampleVoiceFile() async {
    final f = File('${_modelDir!.path}/clips/sample_voice.wav');
    if (!f.existsSync()) {
      f.parent.createSync(recursive: true);
      f.writeAsBytesSync((await rootBundle.load('assets/sample_voice.wav')).buffer.asUint8List());
    }
    return f.path;
  }

  /// Records a few seconds of the user's own voice, which the model then imitates.
  Future<void> _toggleVoiceRecording() async {
    if (_recordingVoice) {
      final path = await _recorder.stop();
      if (!mounted) return;
      setState(() {
        _recordingVoice = false;
        if (path != null) { _voicePath = path; _voiceLabel = 'your recording'; }
      });
      return;
    }
    if (!await _recorder.hasPermission()) {
      _toast('kiln needs the microphone to record a voice.');
      return;
    }
    final dir = Directory('${_modelDir!.path}/clips')..createSync(recursive: true);
    final path = '${dir.path}/voice_${DateTime.now().millisecondsSinceEpoch}.wav';
    try {
      await _recorder.start(const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1), path: path);
      if (mounted) setState(() => _recordingVoice = true);
    } catch (e) {
      _toast('Could not record: $e');
    }
  }

  Future<void> _pickVoice() async {
    final List<PlatformFile> picked;
    try {
      picked = await FilePicker.pickFiles(type: FileType.custom, allowedExtensions: _audioExtensions);
    } catch (e) {
      _toast('Could not open the file picker: $e');
      return;
    }
    final f = picked.isEmpty ? null : picked.first;
    if (f?.path != null && mounted) setState(() { _voicePath = f!.path; _voiceLabel = f.name; });
  }

  /// Generates the sentence, writes it to a wav and plays it. The audio arrives in pieces
  /// while the model is still speaking; this simple version waits for all of them.
  Future<void> _speak() async {
    final ctx = _ctx;
    final text = _sayText.text.trim();
    if (ctx == null || _speaking || _projector == null) return;
    if (text.isEmpty) { _toast('Type something to say.'); return; }
    FocusScope.of(context).unfocus();
    setState(() { _speaking = true; _speakInfo = 'speaking…'; });
    try {
      final voice = _voicePath ?? await _sampleVoiceFile();
      final s = ctx.speak(text, voice: KilnMedia.file(voice));
      final wav = await s.wav;
      final st = await s.stats;
      final out = File('${_modelDir!.path}/clips/said_${DateTime.now().millisecondsSinceEpoch}.wav')..writeAsBytesSync(wav);
      _lastWavPath = out.path;
      await _player.play(DeviceFileSource(out.path));
      if (!mounted) return;
      setState(() => _speakInfo = '${st.audioSeconds.toStringAsFixed(1)} s of speech in ${(st.ms / 1000).toStringAsFixed(2)} s  ·  '
          '${st.speed.toStringAsFixed(1)}x real time  ·  ${st.frames} frames');
    } catch (e) {
      if (mounted) setState(() => _speakInfo = e is KilnException ? e.message : '$e');
    } finally {
      if (mounted) setState(() => _speaking = false);
    }
  }

  Widget _speakView(ColorScheme scheme) {
    final cat = _loadedCatalog!;
    final ready = _projector?.generatesAudio ?? false;
    return SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 4, 20, 32),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _hint('${cat.name} turns text into speech, on the phone, in the voice of a short recording: '
          'the bundled sample, a file, or three seconds of your own voice.', scheme),
      if (!ready) ...[
        _sectionTitle('Voice engine', scheme),
        if (_projectorOnDisk == null)
          FilledButton.tonal(
            onPressed: _downloading == null ? _getProjector : null,
            child: Text(_downloading == cat.projectorFile ? 'Downloading…' : 'Get the voice engine (${cat.projectorMegabytes} MB)'),
          )
        else
          FilledButton.tonal(onPressed: _loadingProjector ? null : _enableMedia, child: Text(_loadingProjector ? 'Loading…' : 'Load the voice engine')),
        if (_downloading == cat.projectorFile) _progressBar(scheme),
      ] else ...[
        _sectionTitle('Say', scheme),
        Wrap(spacing: 8, runSpacing: 4, children: [
          for (final t in _sampleSentences)
            ActionChip(
              label: Text(t, maxLines: 1, overflow: TextOverflow.ellipsis),
              side: BorderSide(color: scheme.outlineVariant),
              shape: const StadiumBorder(),
              onPressed: _speaking ? null : () => setState(() => _sayText.text = t),
            ),
        ]),
        const SizedBox(height: 12),
        TextField(controller: _sayText, minLines: 2, maxLines: 5, decoration: _fieldStyle(scheme, 'What should it say?')),
        _sectionTitle('Voice', scheme),
        Row(children: [
          Expanded(child: Text('Imitating: $_voiceLabel', style: TextStyle(color: scheme.onSurfaceVariant))),
          TextButton.icon(
            onPressed: _speaking ? null : _toggleVoiceRecording,
            icon: Icon(_recordingVoice ? Icons.stop_rounded : Icons.mic_rounded, size: 18),
            label: Text(_recordingVoice ? 'Stop' : 'Record'),
          ),
          TextButton.icon(onPressed: _speaking || _recordingVoice ? null : _pickVoice, icon: const Icon(Icons.audio_file_outlined, size: 18), label: const Text('File')),
        ]),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: _speaking || _recordingVoice ? null : _speak,
          icon: const Icon(Icons.volume_up_rounded),
          label: Text(_speaking ? 'Speaking…' : 'Speak', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        ),
        if (_speakInfo.isNotEmpty) ...[
          const SizedBox(height: 14),
          Row(children: [
            Expanded(child: Text(_speakInfo, style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant))),
            if (_lastWavPath != null)
              IconButton(tooltip: 'Play again', icon: const Icon(Icons.replay_rounded), onPressed: () => _player.play(DeviceFileSource(_lastWavPath!))),
          ]),
        ],
      ],
    ]));
  }

  Widget _chatView(ColorScheme scheme) => Column(children: [
        Expanded(child: _turns.isEmpty ? _emptyChat(scheme) : _turnList(scheme)),
        _composer(scheme),
      ]);

  Widget _emptyChat(ColorScheme scheme) => Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(28),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.local_fire_department_rounded, size: 44, color: scheme.primary),
            const SizedBox(height: 14),
            const Text('Ask anything', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            Text('Runs on this phone. Nothing leaves it.', style: TextStyle(color: scheme.onSurfaceVariant)),
            const SizedBox(height: 28),
            for (final s in _suggestions)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Material(
                  color: scheme.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(14),
                  clipBehavior: Clip.antiAlias,
                  child: InkWell(
                    onTap: () => _send(s),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      child: Row(children: [
                        Expanded(child: Text(s)),
                        Icon(Icons.north_east_rounded, size: 16, color: scheme.onSurfaceVariant),
                      ]),
                    ),
                  ),
                ),
              ),
          ]),
        ),
      );

  Widget _turnList(ColorScheme scheme) => ListView.builder(
        controller: _scroll,
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        itemCount: _turns.length,
        itemBuilder: (_, i) {
          final t = _turns[i];
          return t.role == 'user' ? _userBubble(t, scheme) : _reply(t, scheme, live: i == _turns.length - 1 && _gen != null);
        },
      );

  Widget _userBubble(_Turn t, ColorScheme scheme) => Align(
        alignment: Alignment.centerRight,
        child: Container(
          constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.8),
          margin: const EdgeInsets.only(top: 14, bottom: 6),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
          decoration: BoxDecoration(
            color: scheme.primaryContainer,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(20), topRight: Radius.circular(20),
              bottomLeft: Radius.circular(20), bottomRight: Radius.circular(6)),
          ),
          child: Text(t.text, style: TextStyle(color: scheme.onPrimaryContainer, fontSize: 15.5, height: 1.4)),
        ),
      );

  /// The model's reply sits directly on the background, without a bubble.
  Widget _reply(_Turn t, ColorScheme scheme, {required bool live}) {
    final waiting = live && t.text.isEmpty && t.reasoning.isEmpty;
    final st = t.stats;
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 6, right: 12),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        if (t.reasoning.isNotEmpty) _reasoning(t, scheme, live: live && t.text.isEmpty),
        if (waiting) Padding(padding: const EdgeInsets.symmetric(vertical: 8), child: _Dots(color: scheme.onSurfaceVariant)),
        if (t.text.isNotEmpty)
          SelectionArea(child: _Markdown(t.text.trim(), scheme: scheme)),
        if (t.error != null)
          Container(
            margin: const EdgeInsets.only(top: 6),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(color: scheme.errorContainer, borderRadius: BorderRadius.circular(12)),
            child: Text(t.error!, style: TextStyle(color: scheme.onErrorContainer, fontSize: 13.5)),
          ),
        if (st != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Row(children: [
              Flexible(
                child: Text(
                  [
                    '${st.genTps.toStringAsFixed(1)} tok/s',
                    st.reasoningTokens > 0
                        ? '${st.generatedTokens} tokens, ${st.reasoningTokens} thinking'
                        : '${st.generatedTokens} tokens',
                    if (st.cachedTokens > 0) '${st.cachedTokens} cached',
                    if (st.stoppedByUser) 'stopped',
                  ].join('  ·  '),
                  style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
                ),
              ),
              if (t.text.isNotEmpty)
                InkWell(
                  borderRadius: BorderRadius.circular(8),
                  onTap: () {
                    Clipboard.setData(ClipboardData(text: t.text.trim()));
                    _toast('Copied');
                  },
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    child: Icon(Icons.copy_rounded, size: 15, color: scheme.onSurfaceVariant),
                  ),
                ),
            ]),
          ),
      ]),
    );
  }

  Widget _reasoning(_Turn t, ColorScheme scheme, {required bool live}) {
    final muted = scheme.onSurfaceVariant;
    final cut = t.stats?.reasoningTruncated ?? false;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: () => setState(() => t.showReasoning = !t.showReasoning),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.psychology_alt_outlined, size: 16, color: live ? scheme.primary : muted),
              const SizedBox(width: 6),
              Text(live ? 'Thinking…' : cut ? 'Thought (stopped at the budget)' : 'Thought',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: live ? scheme.primary : muted)),
              Icon(t.showReasoning ? Icons.expand_less_rounded : Icons.expand_more_rounded, size: 18, color: muted),
            ]),
          ),
        ),
        if (t.showReasoning)
          Container(
            margin: const EdgeInsets.only(top: 4, left: 7),
            padding: const EdgeInsets.only(left: 14),
            decoration: BoxDecoration(border: Border(left: BorderSide(color: scheme.outlineVariant, width: 2))),
            child: _Markdown(t.reasoning.trim(), scheme: scheme, color: muted, fontSize: 13.5),
          ),
      ]),
    );
  }

  Widget _composer(ColorScheme scheme) => Padding(
        padding: const EdgeInsets.fromLTRB(12, 6, 12, 10),
        child: Container(
          padding: const EdgeInsets.fromLTRB(18, 2, 6, 2),
          decoration: BoxDecoration(color: scheme.surfaceContainer, borderRadius: BorderRadius.circular(26)),
          child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
            Expanded(
              child: TextField(
                controller: _input,
                minLines: 1,
                maxLines: 5,
                textCapitalization: TextCapitalization.sentences,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _send(),
                decoration: InputDecoration(
                  hintText: 'Message',
                  hintStyle: TextStyle(color: scheme.onSurfaceVariant),
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const SizedBox(width: 6),
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: _gen == null
                  ? ListenableBuilder(
                      listenable: _input,
                      builder: (_, _) => IconButton.filled(
                        tooltip: 'Send',
                        onPressed: _input.text.trim().isEmpty ? null : _send,
                        icon: const Icon(Icons.arrow_upward_rounded),
                      ),
                    )
                  : IconButton.filled(
                      tooltip: 'Stop',
                      onPressed: () => _gen?.cancel(),
                      icon: const Icon(Icons.stop_rounded),
                    ),
            ),
          ]),
        ),
      );
}

/// Three pulsing dots shown while the model reads the prompt, before the first token.
class _Dots extends StatefulWidget {
  const _Dots({required this.color});
  final Color color;
  @override
  State<_Dots> createState() => _DotsState();
}

class _DotsState extends State<_Dots> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
        animation: _c,
        builder: (_, _) => Row(mainAxisSize: MainAxisSize.min, children: [
          for (var i = 0; i < 3; i++)
            Container(
              width: 7,
              height: 7,
              margin: const EdgeInsets.only(right: 5),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                // each dot peaks a third of a cycle after the previous one
                color: widget.color.withValues(alpha: 0.25 + 0.75 * (1 - ((_c.value - i / 3) % 1.0 - 0.5).abs() * 2).clamp(0.0, 1.0)),
              ),
            ),
        ]),
      );
}

/// Just enough markdown for what small models write: headings, bullets, **bold**, `code`
/// and fenced code blocks. Inline maths is shown as plain text. No dependency needed.
class _Markdown extends StatelessWidget {
  const _Markdown(this.text, {required this.scheme, this.color, this.fontSize = 15.5});
  final String text;
  final ColorScheme scheme;
  final Color? color;
  final double fontSize;

  static final _inline = RegExp(r'\*\*(.+?)\*\*|`([^`]+)`');
  static final _math = RegExp(r'\$\$?([^$\n]*[\\=×+^_][^$\n]*)\$\$?');
  static final _mathNumber = RegExp(r'\$\$?(\d[\d.,]*)(?<![.,])\$\$?');
  static final _bullet = RegExp(r'^\s*[*\-•]\s+');
  static final _heading = RegExp(r'^#{1,6}\s+');

  static String _plainMath(String s) => s
      .replaceAll(r'\times', '×').replaceAll(r'\cdot', '·').replaceAll(r'\div', '÷').replaceAll(r'\approx', '≈')
      .replaceAllMapped(_math, (m) => m[1]!)
      .replaceAllMapped(_mathNumber, (m) => m[1]!);

  List<InlineSpan> _spans(String line) {
    final out = <InlineSpan>[];
    var at = 0;
    for (final m in _inline.allMatches(line)) {
      if (m.start > at) out.add(TextSpan(text: line.substring(at, m.start)));
      out.add(m[1] != null
          ? TextSpan(text: m[1], style: const TextStyle(fontWeight: FontWeight.w700))
          : TextSpan(
              text: m[2],
              style: TextStyle(fontFamily: 'monospace', fontSize: fontSize - 1.5, backgroundColor: scheme.surfaceContainerHigh)));
      at = m.end;
    }
    if (at < line.length) out.add(TextSpan(text: line.substring(at)));
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final base = TextStyle(fontSize: fontSize, height: 1.5, color: color ?? scheme.onSurface);
    final blocks = <Widget>[];
    final para = <String>[];
    void flush() {
      if (para.isEmpty) return;
      blocks.add(Text.rich(TextSpan(children: _spans(para.join('\n'))), style: base));
      para.clear();
    }

    List<String>? code; // non-null while inside a fenced block
    for (final raw in text.split('\n')) {
      if (raw.trimLeft().startsWith('```')) {
        if (code == null) {
          flush();
          code = [];
        } else {
          blocks.add(_codeBlock(code.join('\n')));
          code = null;
        }
        continue;
      }
      if (code != null) {
        code.add(raw);
        continue;
      }
      final line = _plainMath(raw);
      if (line.trim().isEmpty) {
        flush();
      } else if (_heading.hasMatch(line)) {
        flush();
        blocks.add(Text.rich(TextSpan(children: _spans(line.replaceFirst(_heading, ''))),
            style: base.copyWith(fontSize: fontSize + 1.5, fontWeight: FontWeight.w700)));
      } else if (_bullet.hasMatch(line)) {
        flush();
        final indent = (line.length - line.trimLeft().length).clamp(0, 8) * 4.0;
        blocks.add(Padding(
          padding: EdgeInsets.only(left: indent),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            SizedBox(width: 18, child: Text('•', style: base.copyWith(color: scheme.primary))),
            Expanded(child: Text.rich(TextSpan(children: _spans(line.replaceFirst(_bullet, ''))), style: base)),
          ]),
        ));
      } else {
        para.add(line.trimLeft());
      }
    }
    if (code != null) blocks.add(_codeBlock(code.join('\n'))); // still streaming
    flush();

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      for (var i = 0; i < blocks.length; i++)
        Padding(padding: EdgeInsets.only(top: i == 0 ? 0 : 8), child: blocks[i]),
    ]);
  }

  Widget _codeBlock(String code) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: scheme.surfaceContainerLow, borderRadius: BorderRadius.circular(12)),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Text(code, style: TextStyle(fontFamily: 'monospace', fontSize: fontSize - 2, height: 1.45, color: color ?? scheme.onSurface)),
        ),
      );
}
