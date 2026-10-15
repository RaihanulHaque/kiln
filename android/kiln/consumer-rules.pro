# Native code finds these by name: the JNI functions on dev.kiln.Native, the callback methods
# onPiece / onAudio, and KilnException's (int, byte[]) constructor.
-keep class dev.kiln.Native { *; }
-keep class * implements dev.kiln.PieceSink { boolean onPiece(int, byte[], int); }
-keep class * implements dev.kiln.AudioSink { boolean onAudio(float[], int); }
-keep class dev.kiln.KilnException { <init>(...); }
