// The TurboModule spec: React Native's codegen reads this file and generates the Kotlin base
// class (NativeKilnSpec) that android/src/main/java/dev/kiln/reactnative/KilnModule.kt fills in.
// Keep it to codegen's types: number, boolean, string, arrays, plain object types, Promise.
// Apps use the classes in kiln.ts, not this module directly.
import type { CodegenTypes, TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type NativeDeviceInfo = {
  soc: string;
  abi: string;
  cpuCores: number;
  cpuBigCores: number;
  /** Bitmask of the fast cores, as a decimal string (it can exceed 2^53). */
  cpuFastMask: string;
  ramTotalBytes: number;
  ramAvailableBytes: number;
  hasFp16: boolean;
  hasDotprod: boolean;
  hasI8mm: boolean;
  hasSve: boolean;
  hasGpu: boolean;
  gpuName: string;
  hasNpu: boolean;
  npuName: string;
};

export type NativeModelInfo = {
  handle: number;
  backendName: string;
  arch: string;
  quant: string;
  nParams: number;
  nCtxTrain: number;
  nVocab: number;
  nGpuLayersUsed: number;
  modelBytes: number;
  supportsThinking: boolean;
  isDecisionModel: boolean;
  isEmbeddingModel: boolean;
  nEmbd: number;
  isSpeechModel: boolean;
};

export type NativeProjectorInfo = {
  supportsImages: boolean;
  supportsAudio: boolean;
  generatesAudio: boolean;
  sampleRate: number;
};

export type NativeMessage = { role: string; content: string };

export type NativeChatOptions = {
  addAssistantPrefix: boolean;
  enableThinking: boolean;
  maxReasoningTokens: number;
};

export type NativeSampling = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  repeatLastN: number;
  seed: number;
  maxTokens: number;
};

export type NativeGenStats = {
  promptTokens: number;
  cachedTokens: number;
  generatedTokens: number;
  reasoningTokens: number;
  promptMs: number;
  genMs: number;
  promptTps: number;
  genTps: number;
  stoppedByEos: boolean;
  stoppedByUser: boolean;
  reasoningTruncated: boolean;
};

export type NativeOption = { key: string | null; description: string | null };

export type NativeQuestion = {
  id: string;
  /** "yes_no" | "choice" | "score" */
  type: string;
  instructions: string;
  options: NativeOption[];
};

export type NativeDecision = {
  probs: number[];
  best: number;
  value: number;
  confidence: number;
  promptTokens: number;
  ms: number;
};

export type NativeEmbedOptions = {
  /** "raw" | "query" | "document" */
  kind: string;
  title: string | null;
  dimensions: number;
  normalize: boolean;
};

export type NativeEmbedding = {
  vector: number[];
  tokens: number;
  mediaTokens: number;
  dimensions: number;
  prefixed: boolean;
  mediaMs: number;
  ms: number;
};

/** A file path, or the file's bytes as base64. */
export type NativeMedia = { path: string | null; base64: string | null };

export type NativeTranscribeOptions = { language: string | null; prompt: string | null; maxTokens: number };

export type NativeTranscript = {
  text: string;
  audioSeconds: number;
  audioTokens: number;
  promptTokens: number;
  generatedTokens: number;
  encodeMs: number;
  promptMs: number;
  genMs: number;
  ms: number;
  language: string;
  stoppedByEos: boolean;
  stoppedByUser: boolean;
};

export type NativeSpeakOptions = {
  voice: NativeMedia | null;
  language: string | null;
  maxFrames: number;
  topK: number;
  topP: number;
  /** -1 = random */
  seed: number;
};

export type NativeSpeech = {
  /** The wav file written (16-bit mono). */
  path: string;
  sampleRate: number;
  audioSeconds: number;
  frames: number;
  promptTokens: number;
  promptMs: number;
  genMs: number;
  ms: number;
  stoppedByEos: boolean;
  stoppedByUser: boolean;
};

/** One streamed piece of a chat, generate or transcribe call, tagged with the call's id. */
export type NativePieceEvent = { requestId: number; channel: string; text: string; tokenId: number };

export interface Spec extends TurboModule {
  version(): string;
  probeDevice(): Promise<NativeDeviceInfo>;
  setLogLevel(level: number): void;

  loadModel(path: string, backend: string, nGpuLayers: number, nThreads: number, useMmap: boolean): Promise<NativeModelInfo>;
  freeModel(model: number): Promise<void>;
  loadProjector(model: number, path: string, imageMaxTokens: number): Promise<NativeProjectorInfo>;
  tokenize(model: number, text: string, addSpecial: boolean): Promise<number[]>;
  detokenize(model: number, tokens: number[]): Promise<string>;
  applyChatTemplate(model: number, messages: NativeMessage[], options: NativeChatOptions): Promise<string>;

  newContext(model: number, nCtx: number, nBatch: number, flashAttn: boolean): Promise<number>;
  freeContext(context: number): Promise<void>;
  resetContext(context: number): Promise<void>;
  /** Stops the running call on this context; that call's promise rejects with code "CANCELLED". */
  cancel(context: number): void;
  /** Stops the call with this id quietly: it resolves with stoppedByUser. */
  stop(requestId: number): void;

  chat(
    context: number,
    requestId: number,
    messages: NativeMessage[],
    options: NativeChatOptions,
    sampling: NativeSampling,
  ): Promise<NativeGenStats>;
  generate(context: number, requestId: number, prompt: string, sampling: NativeSampling): Promise<NativeGenStats>;
  decide(context: number, state: string | null, question: NativeQuestion): Promise<NativeDecision>;
  embed(context: number, text: string, options: NativeEmbedOptions): Promise<NativeEmbedding>;
  embedMedia(context: number, text: string | null, media: NativeMedia[], options: NativeEmbedOptions): Promise<NativeEmbedding>;
  transcribe(context: number, requestId: number, audio: NativeMedia, options: NativeTranscribeOptions): Promise<NativeTranscript>;
  speak(context: number, text: string, options: NativeSpeakOptions, outPath: string): Promise<NativeSpeech>;

  readonly onPiece: CodegenTypes.EventEmitter<NativePieceEvent>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('Kiln');
