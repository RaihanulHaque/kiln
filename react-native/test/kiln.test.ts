// The JS layer against a fake native module: routing of streamed pieces to the right call,
// stopping, cancelling, defaults, and the shape of what reaches native code. The native side is
// the dev.kiln Kotlin library, tested in ../android.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_SAMPLING, KilnLibrary, greedy, isCancelled, type NativeKilnLike } from '../src/kiln';
import type { NativePieceEvent } from '../src/NativeKiln';

type Call = { name: string; args: unknown[] };

const stats = {
  promptTokens: 10, cachedTokens: 0, generatedTokens: 3, reasoningTokens: 0, promptMs: 1, genMs: 2,
  promptTps: 10, genTps: 50, stoppedByEos: true, stoppedByUser: false, reasoningTruncated: false,
};

function fake() {
  const calls: Call[] = [];
  let emit: (e: NativePieceEvent) => void = () => {};
  const stoppedIds = new Set<number>();
  const cancelled = new Set<number>();
  // a chat that streams `pieces` for its request id and resolves, unless stopped or cancelled
  let pieces: { channel: string; text: string }[] = [];
  const record = (name: string) => (...args: unknown[]) => {
    calls.push({ name, args });
  };
  const native: NativeKilnLike = {
    version: () => '0.5.0',
    probeDevice: async () => { throw new Error('unused'); },
    setLogLevel: record('setLogLevel'),
    loadModel: async (path) => {
      calls.push({ name: 'loadModel', args: [path] });
      if (path.includes('missing')) throw Object.assign(new Error('failed to load'), { code: 'MODEL_LOAD' });
      return {
        handle: 7, backendName: 'llama.cpp/cpu', arch: 'gemma3', quant: 'Q8_0', nParams: 1, nCtxTrain: 1, nVocab: 1,
        nGpuLayersUsed: 0, modelBytes: 1, supportsThinking: false, isDecisionModel: false, isEmbeddingModel: false, nEmbd: 0,
        isSpeechModel: false,
      };
    },
    freeModel: async (m) => { calls.push({ name: 'freeModel', args: [m] }); },
    loadProjector: async () => ({ supportsImages: true, supportsAudio: false, generatesAudio: false, sampleRate: 0 }),
    tokenize: async () => [1, 2],
    detokenize: async () => 'x',
    applyChatTemplate: async () => '',
    newContext: async (m, ...rest) => { calls.push({ name: 'newContext', args: [m, ...rest] }); return 3; },
    freeContext: async (c) => { calls.push({ name: 'freeContext', args: [c] }); },
    resetContext: async () => {},
    cancel: (c) => { calls.push({ name: 'cancel', args: [c] }); cancelled.add(c); },
    stop: (id) => { stoppedIds.add(id); },
    chat: async (ctx, id, messages, options, sampling) => {
      calls.push({ name: 'chat', args: [ctx, id, messages, options, sampling] });
      // "say:xyz" streams x, y, z: lets a test give each call its own pieces
      const last = (messages as { content: string }[]).at(-1)!.content;
      const own = last.startsWith('say:') ? [...last.slice(4)].map((text) => ({ channel: 'answer', text })) : pieces;
      for (const p of own) {
        await Promise.resolve();
        if (cancelled.delete(ctx)) throw Object.assign(new Error('cancelled'), { code: 'CANCELLED' });
        if (stoppedIds.has(id)) return { ...stats, stoppedByEos: false, stoppedByUser: true };
        emit({ requestId: id, channel: p.channel, text: p.text, tokenId: 1 });
      }
      return stats;
    },
    generate: async () => stats,
    decide: async (ctx, state, q) => { calls.push({ name: 'decide', args: [ctx, state, q] }); return { probs: [0.2, 0.8], best: 1, value: 0.8, confidence: 0.6, promptTokens: 5, ms: 1 }; },
    embed: async (ctx, text, o) => { calls.push({ name: 'embed', args: [ctx, text, o] }); return { vector: [1, 0], tokens: 2, mediaTokens: 0, dimensions: 2, prefixed: true, mediaMs: 0, ms: 1 }; },
    embedMedia: async (ctx, text, media, o) => { calls.push({ name: 'embedMedia', args: [ctx, text, media, o] }); return { vector: [1], tokens: 1, mediaTokens: 1, dimensions: 1, prefixed: false, mediaMs: 1, ms: 1 }; },
    transcribe: async (ctx, id, audio, o) => {
      calls.push({ name: 'transcribe', args: [ctx, id, audio, o] });
      emit({ requestId: id, channel: 'answer', text: 'The train', tokenId: 0 });
      emit({ requestId: id, channel: 'answer', text: ' leaves.', tokenId: 0 });
      return { text: 'The train leaves.', audioSeconds: 3, audioTokens: 1, promptTokens: 1, generatedTokens: 1, encodeMs: 1, promptMs: 1, genMs: 1, ms: 1, language: '', stoppedByEos: true, stoppedByUser: false };
    },
    speak: async (ctx, text, o, out) => { calls.push({ name: 'speak', args: [ctx, text, o, out] }); return { path: out, sampleRate: 24000, audioSeconds: 2, frames: 25, promptTokens: 1, promptMs: 1, genMs: 1, ms: 1, stoppedByEos: true, stoppedByUser: false }; },
    onPiece: (handler) => { emit = handler; return { remove() {} }; },
  };
  return { native, calls, setPieces: (p: typeof pieces) => { pieces = p; } };
}

test('chat streams pieces, splits channels and returns the whole reply', async () => {
  const f = fake();
  f.setPieces([{ channel: 'reasoning', text: 'hmm' }, { channel: 'answer', text: 'Red, ' }, { channel: 'answer', text: 'blue' }]);
  const kiln = new KilnLibrary(f.native);
  const model = await kiln.loadModel('/m.gguf');
  assert.equal(model.info.arch, 'gemma3');
  assert.equal((model.info as Record<string, unknown>).handle, undefined, 'the handle stays private');
  const ctx = await model.newContext();
  const seen: string[] = [];
  const reply = await ctx.chat([{ role: 'user', content: 'colours?' }], { onPiece: (p) => { seen.push(`${p.channel}:${p.text}`); } });
  assert.deepEqual(seen, ['reasoning:hmm', 'answer:Red, ', 'answer:blue']);
  assert.equal(reply.answer, 'Red, blue');
  assert.equal(reply.reasoning, 'hmm');
  assert.equal(reply.stats.generatedTokens, 3);
  // kiln's defaults fill in what the caller left out
  const chat = f.calls.find((c) => c.name === 'chat')!;
  assert.deepEqual(chat.args[4], DEFAULT_SAMPLING);
  assert.deepEqual(chat.args[3], { addAssistantPrefix: true, enableThinking: false, maxReasoningTokens: 0 });
  assert.deepEqual(f.calls.find((c) => c.name === 'newContext')!.args, [7, 0, 512, true]);
});

test('two calls at once each get only their own pieces', async () => {
  const f = fake();
  const kiln = new KilnLibrary(f.native);
  const model = await kiln.loadModel('/m.gguf');
  const [c1, c2] = [await model.newContext(), await model.newContext()];
  const [r1, r2] = await Promise.all([c1.chat([{ role: 'user', content: 'say:abc' }]), c2.chat([{ role: 'user', content: 'say:xyz' }])]);
  assert.equal(r1.answer, 'abc');
  assert.equal(r2.answer, 'xyz');
});

test('returning false from onPiece stops quietly', async () => {
  const f = fake();
  f.setPieces([{ channel: 'answer', text: 'one' }, { channel: 'answer', text: 'two' }, { channel: 'answer', text: 'three' }]);
  const ctx = await (await new KilnLibrary(f.native).loadModel('/m.gguf')).newContext();
  let n = 0;
  const r = await ctx.chat([{ role: 'user', content: 'count' }], { sampling: greedy(50), onPiece: () => ++n < 1 });
  assert.equal(n, 1);
  assert.equal(r.answer, 'one');
  assert.ok(r.stats.stoppedByUser);
  const sent = f.calls.find((c) => c.name === 'chat')!.args[4] as Record<string, number>;
  assert.equal(sent.temperature, 0);
  assert.equal(sent.maxTokens, 50);
  assert.equal(sent.topK, DEFAULT_SAMPLING.topK);
});

test('an AbortSignal cancels: the call rejects with CANCELLED', async () => {
  const f = fake();
  f.setPieces([{ channel: 'answer', text: 'x' }, { channel: 'answer', text: 'y' }]);
  const ctx = await (await new KilnLibrary(f.native).loadModel('/m.gguf')).newContext();
  const ac = new AbortController();
  await assert.rejects(
    ctx.chat([{ role: 'user', content: 'go' }], { signal: ac.signal, onPiece: () => { ac.abort(); } }),
    (e: unknown) => isCancelled(e),
  );
});

test('errors keep kiln\'s status as code', async () => {
  const kiln = new KilnLibrary(fake().native);
  await assert.rejects(kiln.loadModel('/missing.gguf'), (e: unknown) => (e as { code: string }).code === 'MODEL_LOAD');
});

test('decide, embed, media, transcribe and speak send what native expects', async () => {
  const f = fake();
  const ctx = await (await new KilnLibrary(f.native).loadModel('/m.gguf')).newContext();

  const d = await ctx.decide('my parcel never came', {
    id: 'team', type: 'choice', instructions: 'Which team?', options: [{ key: 'billing' }, { key: 'shipping', description: 'Delivery' }],
  });
  assert.equal(d.best, 1);
  assert.deepEqual((f.calls.find((c) => c.name === 'decide')!.args[2] as { options: unknown }).options, [
    { key: 'billing', description: null },
    { key: 'shipping', description: 'Delivery' },
  ]);

  await ctx.embed('query text', { kind: 'query' });
  assert.deepEqual(f.calls.find((c) => c.name === 'embed')!.args[2], { kind: 'query', title: null, dimensions: 0, normalize: true });

  await ctx.embedMedia(null, [{ path: '/a.png' }, { base64: 'AAAA' }]);
  assert.deepEqual(f.calls.find((c) => c.name === 'embedMedia')!.args[2], [
    { path: '/a.png', base64: null },
    { path: null, base64: 'AAAA' },
  ]);

  const words: string[] = [];
  const t = await ctx.transcribe({ path: '/clip.wav' }, { onText: (w) => { words.push(w); } });
  assert.deepEqual(words, ['The train', ' leaves.']);
  assert.equal(t.text, 'The train leaves.');

  const s = await ctx.speak('Hello.', '/tmp/out.wav', { voice: { path: '/voice.wav' }, seed: 7 });
  assert.equal(s.path, '/tmp/out.wav');
  assert.deepEqual(f.calls.find((c) => c.name === 'speak')!.args[2], {
    voice: { path: '/voice.wav', base64: null }, language: null, maxFrames: 0, topK: 0, topP: 0, seed: 7,
  });
});

test('closed handles refuse calls and close only once', async () => {
  const f = fake();
  const model = await new KilnLibrary(f.native).loadModel('/m.gguf');
  const ctx = await model.newContext();
  await ctx.close();
  await ctx.close();
  assert.equal(f.calls.filter((c) => c.name === 'freeContext').length, 1);
  assert.throws(() => ctx.embed('x'), /closed/);
  await model.close();
  assert.throws(() => model.tokenize('x'), /closed/);
});

test('aborting one call leaves another call on the same context running', async () => {
  const f = fake();
  const ctx = await (await new KilnLibrary(f.native).loadModel('/m.gguf')).newContext();
  const ac = new AbortController();
  const first = ctx.chat([{ role: 'user', content: 'say:abcd' }], { signal: ac.signal, onPiece: () => { ac.abort(); } });
  const second = ctx.chat([{ role: 'user', content: 'say:wxyz' }]);
  await assert.rejects(first, (e: unknown) => isCancelled(e));
  assert.equal((await second).answer, 'wxyz');
  assert.equal(f.calls.filter((c) => c.name === 'cancel').length, 0, 'no context-wide cancel');
});

test('an already aborted signal rejects without calling native code', async () => {
  const f = fake();
  const ctx = await (await new KilnLibrary(f.native).loadModel('/m.gguf')).newContext();
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(ctx.chat([{ role: 'user', content: 'go' }], { signal: ac.signal }), (e: unknown) => isCancelled(e));
  assert.equal(f.calls.filter((c) => c.name === 'chat').length, 0);
});
