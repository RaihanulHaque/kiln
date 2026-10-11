import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:kiln_flutter/kiln_flutter.dart';
import 'package:path_provider/path_provider.dart';

void main() => runApp(const KilnApp());

class KilnApp extends StatelessWidget {
  const KilnApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'kiln',
        theme: ThemeData(colorSchemeSeed: const Color(0xFFD9480F), brightness: Brightness.dark, useMaterial3: true),
        home: const HomePage(),
      );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _Turn {
  _Turn(this.role, this.text);
  final String role;
  String text;
  String reasoning = '';
  KilnGenStats? stats;
  bool showReasoning = false;
}

class _HomePageState extends State<HomePage> {
  Kiln? _kiln;
  KilnDeviceInfo? _device;
  KilnModel? _model;
  KilnContext? _ctx;
  KilnGeneration? _gen;
  Directory? _modelDir;
  List<File> _models = [];
  String? _status;
  double? _downloadProgress;
  bool _thinking = false;
  final _turns = <_Turn>[];
  final _input = TextEditingController();
  final _url = TextEditingController(
      text: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf');
  final _scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    _kiln = await Kiln.create();
    _device = await _kiln!.probeDevice();
    // App-specific external dir: no permission needed, and reachable with
    //   adb push model.gguf /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
    _modelDir = (await getExternalStorageDirectory()) ?? await getApplicationDocumentsDirectory();
    _modelDir!.createSync(recursive: true);
    await _refreshModels();
    setState(() {});
  }

  Future<void> _refreshModels() async {
    final dir = _modelDir;
    if (dir == null) return;
    final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.gguf')).toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    setState(() => _models = files);
  }

  Future<void> _download() async {
    final url = _url.text.trim();
    final name = url.split('/').last.split('?').first;
    final out = File('${_modelDir!.path}/$name');
    setState(() { _downloadProgress = 0; _status = 'downloading $name'; });
    try {
      final req = http.Request('GET', Uri.parse(url));
      final resp = await http.Client().send(req);
      final total = resp.contentLength ?? 0;
      var got = 0;
      final sink = out.openWrite();
      await for (final chunk in resp.stream) {
        sink.add(chunk);
        got += chunk.length;
        if (total > 0) setState(() => _downloadProgress = got / total);
      }
      await sink.close();
      setState(() { _downloadProgress = null; _status = null; });
      await _refreshModels();
    } catch (e) {
      setState(() { _downloadProgress = null; _status = 'download failed: $e'; });
    }
  }

  Future<void> _load(File f) async {
    setState(() => _status = 'loading ${f.uri.pathSegments.last}');
    try {
      await _ctx?.dispose();
      await _model?.dispose();
      final sw = Stopwatch()..start();
      _model = await _kiln!.loadModel(f.path);
      _ctx = await _model!.createContext(params: const KilnContextParams(nCtx: 4096));
      _turns.clear();
      setState(() => _status = 'loaded in ${sw.elapsedMilliseconds} ms');
    } catch (e) {
      setState(() => _status = 'load failed: $e');
    }
  }

  Future<void> _send([String? preset]) async {
    final text = (preset ?? _input.text).trim();
    if (text.isEmpty || _ctx == null || _gen != null) return;
    _input.clear();
    final reply = _Turn('assistant', '');
    setState(() { _turns.add(_Turn('user', text)); _turns.add(reply); });
    final history = [
      for (final t in _turns)
        if (!identical(t, reply)) KilnChatMessage(t.role, t.text),
    ];
    // 512-token thinking budget: small models such as Qwen3.5 0.8B rarely stop thinking on their own
    final gen = _ctx!.chat(history, enableThinking: _thinking && _model!.info.supportsThinking, maxReasoningTokens: 512);
    _gen = gen;
    gen.pieces.listen((p) {
      setState(() {
        if (p.isReasoning) { reply.reasoning += p.text; reply.showReasoning = true; }
        else { reply.text += p.text; }
      });
      _scrollDown();
    });
    try {
      reply.stats = await gen.stats;
    } catch (e) {
      reply.text += '\n[error: $e]';
    }
    setState(() => _gen = null);
  }

  void _scrollDown() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('kiln'),
        actions: [
          if (_model?.info.supportsThinking ?? false)
            Row(children: [
              const Text('think'),
              Switch(value: _thinking, onChanged: (v) => setState(() => _thinking = v)),
            ]),
          IconButton(icon: const Icon(Icons.refresh), tooltip: 'reset context', onPressed: () async { await _ctx?.reset(); setState(_turns.clear); }),
        ],
      ),
      body: Column(children: [
        _header(theme),
        const Divider(height: 1),
        Expanded(child: _ctx == null ? _modelPicker(theme) : _chat(theme)),
        if (_ctx != null) _composer(theme),
      ]),
    );
  }

  Widget _header(ThemeData theme) {
    final d = _device;
    final i = _model?.info;
    final small = theme.textTheme.bodySmall;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(d == null ? 'probing device…' : d.toString(), style: small),
        if (d != null)
          Text('fp16=${d.hasFp16} dotprod=${d.hasDotprod} i8mm=${d.hasI8mm} sve=${d.hasSve}', style: small?.copyWith(color: theme.hintColor)),
        if (i != null) Text(i.toString(), style: small?.copyWith(color: theme.colorScheme.primary)),
        if (_status != null) Text(_status!, style: small?.copyWith(color: theme.colorScheme.tertiary)),
        if (_downloadProgress != null) LinearProgressIndicator(value: _downloadProgress),
      ]),
    );
  }

  Widget _modelPicker(ThemeData theme) => ListView(padding: const EdgeInsets.all(12), children: [
        Text(_modelDir == null ? 'starting…' : 'Models in ${_modelDir!.path}', style: theme.textTheme.titleSmall),
        const SizedBox(height: 4),
        if (_models.isEmpty) const Text('none yet. Download one below or adb push a .gguf into that folder.'),
        for (final f in _models)
          ListTile(
            leading: const Icon(Icons.memory),
            title: Text(f.uri.pathSegments.last),
            subtitle: Text('${(f.lengthSync() / 1e6).toStringAsFixed(0)} MB'),
            onTap: () => _load(f),
          ),
        const Divider(),
        TextField(controller: _url, decoration: const InputDecoration(labelText: 'GGUF URL'), maxLines: 2),
        const SizedBox(height: 8),
        FilledButton.icon(onPressed: _downloadProgress == null ? _download : null, icon: const Icon(Icons.download), label: const Text('Download')),
      ]);

  static const _suggestions = ['Why is the sky blue? One sentence.', 'What is 17 × 23?', 'And 19 × 21?', 'Write a two-line poem about a kiln.'];

  Widget _chat(ThemeData theme) => Column(children: [
        Expanded(child: _turnList(theme)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Wrap(spacing: 6, children: [
            for (final s in _suggestions) ActionChip(label: Text(s), onPressed: _gen == null ? () => _send(s) : null),
          ]),
        ),
      ]);

  Widget _turnList(ThemeData theme) => ListView.builder(
        controller: _scroll,
        padding: const EdgeInsets.all(12),
        itemCount: _turns.length,
        itemBuilder: (_, i) {
          final t = _turns[i];
          final isUser = t.role == 'user';
          return Align(
            alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
            child: Container(
              constraints: const BoxConstraints(maxWidth: 340),
              margin: const EdgeInsets.symmetric(vertical: 4),
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: isUser ? theme.colorScheme.primaryContainer : theme.colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                if (t.reasoning.isNotEmpty)
                  InkWell(
                    onTap: () => setState(() => t.showReasoning = !t.showReasoning),
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text('${t.showReasoning ? '▾' : '▸'} thinking', style: theme.textTheme.labelSmall?.copyWith(color: theme.hintColor)),
                        if (t.showReasoning)
                          Text(t.reasoning, style: theme.textTheme.bodySmall?.copyWith(color: theme.hintColor, fontStyle: FontStyle.italic)),
                      ]),
                    ),
                  ),
                if (t.text.isEmpty && !isUser && t.stats == null) const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                if (t.text.isNotEmpty) SelectableText(t.text),
                if (t.stats != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(t.stats.toString(), style: theme.textTheme.labelSmall?.copyWith(color: theme.hintColor)),
                  ),
              ]),
            ),
          );
        },
      );

  Widget _composer(ThemeData theme) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
          child: Row(children: [
            Expanded(
              child: TextField(
                controller: _input,
                decoration: const InputDecoration(hintText: 'Message', border: OutlineInputBorder()),
                onSubmitted: (_) => _send(),
                textInputAction: TextInputAction.send,
              ),
            ),
            const SizedBox(width: 8),
            _gen == null
                ? IconButton.filled(onPressed: _send, icon: const Icon(Icons.send))
                : IconButton.filled(onPressed: () => _gen?.cancel(), icon: const Icon(Icons.stop)),
          ]),
        ),
      );
}
