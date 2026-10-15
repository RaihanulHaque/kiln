package dev.kiln

/** Library-wide calls. Loading the native library happens on first use of any kiln class. */
object Kiln {
    /** kiln's version, e.g. "0.5.0". */
    val version: String by lazy { Native.version().str() }

    /** What kiln finds on this device: processor, fast cores, memory, GPU. */
    fun probeDevice(): DeviceInfo {
        val n = LongArray(11)
        val s = Native.probe(n)
        return DeviceInfo(
            soc = s[0].str(), abi = s[1].str(), cpuCores = n[0].toInt(), cpuBigCores = n[1].toInt(), cpuFastMask = n[2],
            ramTotalBytes = n[3], ramAvailableBytes = n[4], hasFp16 = n[5] != 0L, hasDotprod = n[6] != 0L, hasI8mm = n[7] != 0L,
            hasSve = n[8] != 0L, hasGpu = n[9] != 0L, gpuName = s[2].str(), hasNpu = n[10] != 0L, npuName = s[3].str(),
        )
    }

    /** The most detailed level kiln logs (default: warnings). On Android the lines go to logcat. */
    fun setLogLevel(level: LogLevel) = Native.setLogLevel(level.ordinal)
}
