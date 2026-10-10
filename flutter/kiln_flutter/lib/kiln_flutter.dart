/// kiln: on-device LLM runtime for Flutter. Honest about what it runs on.
///
/// ```dart
/// final kiln = await Kiln.create();
/// final model = await kiln.loadModel('/path/to/model.gguf');
/// print(model.info);                       // what actually loaded
/// final ctx = await model.createContext();
/// final gen = ctx.chat([KilnChatMessage.user('Hi')]);
/// await for (final p in gen.pieces) stdout.write(p.text);
/// print(await gen.stats);                  // real tok/s, cache hits, why it stopped
/// ```
library;

export 'src/kiln.dart';
export 'src/types.dart' hide cStr;
