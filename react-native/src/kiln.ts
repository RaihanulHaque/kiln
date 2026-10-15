// The API apps use. It wraps the TurboModule (NativeKiln.ts) in classes, routes streamed pieces
// to the call they belong to, and fills in kiln's defaults. It takes the native module as a
// parameter so the routing can be tested in Node with a fake (test/kiln.test.ts).
import type {
  NativeChatOptions,
  NativeEmbedOptions,
  NativeEmbedding,
  NativeGenStats,
  NativeMedia,
  NativePieceEvent,
  NativeSampling,
  Spec,
} from './NativeKiln';

/** What kiln.ts needs from the native module: the spec, minus React Native's TurboModule base. */
export type NativeKilnLike = Omit<Spec, 'getConstants'>;

export type Backend = 'auto' | 'cpu' | 'gpu' | 'npu';
export type Role = 'system' | 'user' | 'assistant';
export type Channel = 'answer' | 'reasoning';

export interface Message {
  role: Role;
  content: string;
}

export interface ModelParams {
  backend?: Backend;
  /** -1 = all layers if a GPU is chosen, 0 = none. */
  nGpuLayers?: number;
  /** 0 = from the device probe. */
  nThreads?: number;
  useMmap?: boolean;
}

export interface ContextParams {
  /** 0 = the model's default, capped by RAM. */
  nCtx?: number;
  nBatch?: number;
  flashAttn?: boolean;
}

export interface ChatOptions {
  /** Reasoning models only. false = answer directly. */
  enableThinking?: boolean;
  /** After this many reasoning tokens kiln closes the think block and the model answers. <= 0 = unlimited. */
  maxReasoningTokens?: number;
  addAssistantPrefix?: boolean;
}

/** How the next token is picked. kiln's defaults are in DEFAULT_SAMPLING. */
export interface Sampling {
  /** <= 0 = greedy: always the most likely token. */
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  /** 0 = random. */
  seed?: number;
  /** <= 0 = until the model stops or the context is full. */
  maxTokens?: number;
}

/** kiln_sampling_params_default() in src/core/kiln.cpp. */
export const DEFAULT_SAMPLING: Required<Sampling> = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  repeatPenalty: 1.1,
  repeatLastN: 64,
  seed: 0,
  maxTokens: 0,
};

/** Always the most likely token, no repeat penalty: the same prompt gives the same text. */
export function greedy(maxTokens = 0): Sampling {
  return { temperature: 0, repeatPenalty: 1, maxTokens };
}

export interface Piece {
  channel: Channel;
  /** Whole characters only. */
  text: string;
  tokenId: number;
}

export type GenStats = NativeGenStats;

export interface Reply {
  answer: string;
  reasoning: string;
  stats: GenStats;
}

export interface StreamOptions {
  /** Each piece as it is written. Return false to stop: the call then resolves with stoppedByUser. */
  onPiece?: (piece: Piece) => boolean | void;
  /** Aborting rejects the call with code "CANCELLED". */
  signal?: AbortSignal;
}

/** An image or audio clip: a file path, or the file's bytes as base64. */
export type Media = { path: string } | { base64: string };

export type QuestionType = 'yes_no' | 'choice' | 'score';

export interface Question {
  /** Short name, e.g. "wants_refund". */
  id: string;
  type: QuestionType;
  instructions: string;
  /** Choice: key is the label returned. Score: the levels, lowest first (descriptions only). */
  options?: { key?: string; description?: string }[];
}

export interface Decision {
  /** One probability per option. Yes / no: [no, yes]. */
  probs: number[];
  best: number;
  /** Yes / no: P(yes). Score: the expected level, 0 .. n-1. Choice: P(best). */
  value: number;
  confidence: number;
  promptTokens: number;
  ms: number;
}

export interface EmbedOptions {
  /** 'query' and 'document' add the model's task prefix; use them for search. */
  kind?: 'raw' | 'query' | 'document';
  title?: string;
  /** Keep only the first N numbers. <= 0 = the full vector. */
  dimensions?: number;
  /** Scale to length 1, so similarity is the dot product. Default true. */
  normalize?: boolean;
}

export type Embedding = NativeEmbedding;

export interface TranscribeOptions {
  /** The language spoken, e.g. "English", when known. */
  language?: string;
  /** Qwen3-ASR: replaces the instruction. Whisper: text the words may follow on from. */
  prompt?: string;
  maxTokens?: number;
  /** The words as they are written down. Return false to stop. */
  onText?: (text: string) => boolean | void;
  signal?: AbortSignal;
}

export interface SpeakOptions {
  /** A few seconds of someone speaking; the output imitates that voice. Pocket TTS needs one. */
  voice?: Media;
  /** Qwen3-TTS: "en", "zh", ... Pocket TTS: ignored. */
  language?: string;
  maxFrames?: number;
  topK?: number;
  topP?: number;
  /** A value makes the same text give the same audio; unset = random. */
  seed?: number;
}

/** The error a kiln call rejects with: `code` is kiln's status ("MODEL_LOAD", "CANCELLED", ...). */
export interface KilnError extends Error {
  code: string;
}

export function isCancelled(e: unknown): boolean {
  return typeof e === 'object' && e !== null && (e as { code?: unknown }).code === 'CANCELLED';
}

// ---------------------------------------------------------------- the library

/** The entry point: `Kiln` in index.ts is this, bound to the real native module. */
export class KilnLibrary {
  private nextRequest = 1;
  private readonly listeners = new Map<number, (e: NativePieceEvent) => void>();
  private subscribed = false;

  constructor(private readonly native: NativeKilnLike) {}

  /** kiln's version, e.g. "0.5.0". */
  get version(): string {
    return this.native.version();
  }

  probeDevice() {
    return this.native.probeDevice();
  }

  /** 0 = errors, 1 = warnings (default), 2 = info, 3 = debug. On Android the lines go to logcat. */
  setLogLevel(level: 0 | 1 | 2 | 3) {
    this.native.setLogLevel(level);
  }

  async loadModel(path: string, params: ModelParams = {}): Promise<KilnModel> {
    const info = await this.native.loadModel(path, params.backend ?? 'auto', params.nGpuLayers ?? -1, params.nThreads ?? 0, params.useMmap ?? true);
    const { handle, ...rest } = info;
    return new KilnModel(this, handle, rest);
  }

  /** @internal */
  get raw(): NativeKilnLike {
    return this.native;
  }

  /** @internal Starts a streamed call: a fresh id, and pieces with that id go to `onEvent`. */
  begin(onEvent: (e: NativePieceEvent) => void): { id: number; end: () => void } {
    if (!this.subscribed) {
      this.native.onPiece((e) => this.listeners.get(e.requestId)?.(e));
      this.subscribed = true;
    }
    const id = this.nextRequest++;
    this.listeners.set(id, onEvent);
    return { id, end: () => this.listeners.delete(id) };
  }
}

export type ModelInfo = Omit<Awaited<ReturnType<NativeKilnLike['loadModel']>>, 'handle'>;

/** A loaded model. Close it when done, after its contexts. */
export class KilnModel {
  private closed = false;

  constructor(
    private readonly lib: KilnLibrary,
    private readonly handle: number,
    readonly info: ModelInfo,
  ) {}

  private h(): number {
    if (this.closed) throw new Error('the model is closed');
    return this.handle;
  }

  /** The image / audio encoders of this model (its "mmproj" file). Call before creating contexts. */
  loadProjector(path: string, imageMaxTokens = 0) {
    return this.lib.raw.loadProjector(this.h(), path, imageMaxTokens);
  }

  async newContext(params: ContextParams = {}): Promise<KilnContext> {
    const c = await this.lib.raw.newContext(this.h(), params.nCtx ?? 0, params.nBatch ?? 512, params.flashAttn ?? true);
    return new KilnContext(this.lib, c, this.info);
  }

  tokenize(text: string, addSpecial = false) {
    return this.lib.raw.tokenize(this.h(), text, addSpecial);
  }

  detokenize(tokens: number[]) {
    return this.lib.raw.detokenize(this.h(), tokens);
  }

  applyChatTemplate(messages: Message[], options: ChatOptions = {}) {
    return this.lib.raw.applyChatTemplate(this.h(), messages, chatOptions(options));
  }

  /** Frees the model. Rejects with "INVALID_STATE" while a context of it is still open. */
  async close(): Promise<void> {
    if (this.closed) return;
    await this.lib.raw.freeModel(this.handle);
    this.closed = true;
  }
}

/** One conversation and its KV cache. One call runs at a time; a second waits for the first. */
export class KilnContext {
  private closed = false;

  constructor(
    private readonly lib: KilnLibrary,
    private readonly handle: number,
    readonly modelInfo: ModelInfo,
  ) {}

  private h(): number {
    if (this.closed) throw new Error('the context is closed');
    return this.handle;
  }

  /** Stops the running call: its promise rejects with code "CANCELLED". */
  cancel() {
    if (!this.closed) this.lib.raw.cancel(this.handle);
  }

  reset() {
    return this.lib.raw.resetContext(this.h());
  }

  /**
   * Answers the last message. Pass the whole conversation every turn; kiln reuses what is already
   * in its cache (stats.cachedTokens). Pieces stream to `onPiece`; the promise gives the whole reply.
   */
  async chat(messages: Message[], opts: { options?: ChatOptions; sampling?: Sampling } & StreamOptions = {}): Promise<Reply> {
    const h = this.h();
    const reply = { answer: '', reasoning: '' };
    return this.stream(opts, reply, (id) =>
      this.lib.raw.chat(h, id, messages, chatOptions(opts.options ?? {}), sampling(opts.sampling)),
    ).then((stats) => ({ ...reply, stats }));
  }

  /** Continues a raw prompt already in the model's template. */
  async generate(prompt: string, opts: { sampling?: Sampling } & StreamOptions = {}): Promise<Reply> {
    const h = this.h();
    const reply = { answer: '', reasoning: '' };
    return this.stream(opts, reply, (id) => this.lib.raw.generate(h, id, prompt, sampling(opts.sampling))).then((stats) => ({
      ...reply,
      stats,
    }));
  }

  private async stream<T>(
    opts: StreamOptions,
    reply: { answer: string; reasoning: string },
    call: (id: number) => Promise<T>,
  ): Promise<T> {
    let stopped = false;
    const req = this.lib.begin((e) => {
      const piece: Piece = { channel: e.channel === 'reasoning' ? 'reasoning' : 'answer', text: e.text, tokenId: e.tokenId };
      if (piece.channel === 'answer') reply.answer += piece.text;
      else reply.reasoning += piece.text;
      if (!stopped && opts.onPiece?.(piece) === false) {
        stopped = true;
        this.lib.raw.stop(req.id);
      }
    });
    const onAbort = () => this.cancel();
    opts.signal?.addEventListener('abort', onAbort);
    try {
      if (opts.signal?.aborted) this.cancel();
      return await call(req.id);
    } finally {
      opts.signal?.removeEventListener('abort', onAbort);
      req.end();
    }
  }

  /** Decision models: the probability of each answer to one typed question about `state`. */
  decide(state: string | null, question: Question): Promise<Decision> {
    return this.lib.raw.decide(this.h(), state, {
      id: question.id,
      type: question.type,
      instructions: question.instructions,
      options: (question.options ?? []).map((o) => ({ key: o.key ?? null, description: o.description ?? null })),
    });
  }

  /** Embedding models: the vector for `text`. */
  embed(text: string, options: EmbedOptions = {}): Promise<Embedding> {
    return this.lib.raw.embed(this.h(), text, embedOptions(options));
  }

  /** Embedding models with a projector: one vector for text with images / audio, or media alone. */
  embedMedia(text: string | null, media: Media[], options: EmbedOptions = {}): Promise<Embedding> {
    return this.lib.raw.embedMedia(this.h(), text, media.map(nativeMedia), embedOptions(options));
  }

  /** Speech models: writes down one audio clip (wav, mp3, flac). Each call clears the context. */
  async transcribe(audio: Media, options: TranscribeOptions = {}) {
    const h = this.h();
    const { onText, signal, ...rest } = options;
    return this.stream(
      { signal, onPiece: onText && ((p) => onText(p.text)) },
      { answer: '', reasoning: '' },
      (id) =>
        this.lib.raw.transcribe(h, id, nativeMedia(audio), {
          language: rest.language ?? null,
          prompt: rest.prompt ?? null,
          maxTokens: rest.maxTokens ?? 0,
        }),
    );
  }

  /** Text-to-speech models (projector loaded): speaks `text` into a 16-bit wav file at `outPath`. */
  speak(text: string, outPath: string, options: SpeakOptions = {}) {
    return this.lib.raw.speak(
      this.h(),
      text,
      {
        voice: options.voice ? nativeMedia(options.voice) : null,
        language: options.language ?? null,
        maxFrames: options.maxFrames ?? 0,
        topK: options.topK ?? 0,
        topP: options.topP ?? 0,
        seed: options.seed ?? -1,
      },
      outPath,
    );
  }

  /** Stops a running call, waits for it, and frees the context. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.lib.raw.freeContext(this.handle);
  }
}

function chatOptions(o: ChatOptions): NativeChatOptions {
  return {
    addAssistantPrefix: o.addAssistantPrefix ?? true,
    enableThinking: o.enableThinking ?? false,
    maxReasoningTokens: o.maxReasoningTokens ?? 0,
  };
}

function sampling(s: Sampling = {}): NativeSampling {
  return { ...DEFAULT_SAMPLING, ...s };
}

function embedOptions(o: EmbedOptions): NativeEmbedOptions {
  return { kind: o.kind ?? 'raw', title: o.title ?? null, dimensions: o.dimensions ?? 0, normalize: o.normalize ?? true };
}

function nativeMedia(m: Media): NativeMedia {
  return 'path' in m ? { path: m.path, base64: null } : { path: null, base64: m.base64 };
}
