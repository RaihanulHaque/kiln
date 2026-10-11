// kiln example: a small chat app. Two views in one page: the model library (pick or download
// a model) and the chat. Everything kiln-specific is in _HomePageState; the rest is layout.
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:kiln_flutter/kiln_flutter.dart';
import 'package:path_provider/path_provider.dart';

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

/// A model the app offers to download. All three are small enough for a mid-range phone.
class _CatalogModel {
  const _CatalogModel(this.name, this.blurb, this.megabytes, this.url, [this.alias]);
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
  _CatalogModel('Qwen3.5 0.8B', 'Smartest of the three. Can think before answering.', 533,
      'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf'),
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
  return file.replaceAll('.gguf', '');
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
    final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.gguf')).toList()
      ..sort((a, b) => a.path.compareTo(b.path));
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
  Future<void> _download(String url) async {
    final uri = Uri.tryParse(url.trim());
    if (uri == null || !uri.hasScheme || !uri.path.endsWith('.gguf')) {
      _toast('That does not look like a link to a .gguf file.');
      return;
    }
    final name = uri.pathSegments.last;
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
      if (model.info.isDecisionModel) {
        await model.dispose();
        _toast('This is a decision model. It answers typed questions with decide() and cannot chat.');
        return;
      }
      _model = model;
      _ctx = await model.createContext(params: const KilnContextParams(nCtx: 4096));
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
        body: SafeArea(child: _inChat ? _chatView(scheme) : _libraryView(scheme)),
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
                label: const Text('Back to chat'),
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
        loading ? 'Loading…' : loaded ? '${_size(f.lengthSync())}  ·  loaded' : _size(f.lengthSync()),
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
              hintText: 'Paste a link to a .gguf file',
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
        IconButton(
          tooltip: 'New chat',
          icon: const Icon(Icons.edit_square),
          onPressed: _turns.isEmpty ? null : _newChat,
        ),
        const SizedBox(width: 4),
      ],
    );
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
