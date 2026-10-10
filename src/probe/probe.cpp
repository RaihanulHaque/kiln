// CPU / RAM / ABI probe. GPU and NPU fields are filled by backends via annotate_device().
#include "../core/internal.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <mach/mach.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#  include <sys/auxv.h>
#  include <fstream>
#  if defined(__aarch64__)
#    include <asm/hwcap.h>
#  endif
#  if defined(__ANDROID__)
#    include <sys/system_properties.h>
#  endif
#endif

namespace {

void copy_str(char* dst, size_t cap, const std::string& s) {
    strncpy(dst, s.c_str(), cap - 1); dst[cap - 1] = 0;
}

#if defined(__APPLE__)
bool sysctl_bool(const char* name) {
    int v = 0; size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, nullptr, 0) == 0 && v != 0;
}
int sysctl_int(const char* name) {
    int v = 0; size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, nullptr, 0) == 0 ? v : 0;
}
#endif

#if defined(__linux__)
std::string read_cpuinfo_field(const char* key) {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            auto p = line.find(':');
            if (p != std::string::npos) {
                std::string v = line.substr(p + 1);
                size_t a = v.find_first_not_of(" \t");
                return a == std::string::npos ? "" : v.substr(a);
            }
        }
    }
    return "";
}
uint64_t read_meminfo_kb(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            unsigned long long kb = 0;
            sscanf(line.c_str() + strlen(key), ": %llu", &kb);
            return kb;
        }
    }
    return 0;
}
#endif

} // namespace

namespace kiln {

void probe_cpu(kiln_device_info& d) {
    d.cpu_cores = (int32_t)std::thread::hardware_concurrency();

#if defined(__aarch64__) || defined(__arm64__)
    copy_str(d.abi, sizeof d.abi, "arm64-v8a");
#elif defined(__x86_64__)
    copy_str(d.abi, sizeof d.abi, "x86_64");
#else
    copy_str(d.abi, sizeof d.abi, "unknown");
#endif

#if defined(__APPLE__)
    char brand[128] = {0}; size_t sz = sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &sz, nullptr, 0) == 0)
        copy_str(d.soc, sizeof d.soc, brand);
    d.cpu_big_cores = sysctl_int("hw.perflevel0.physicalcpu");
    uint64_t mem = 0; sz = sizeof mem;
    if (sysctlbyname("hw.memsize", &mem, &sz, nullptr, 0) == 0) d.ram_total_bytes = mem;
    {
        vm_size_t page = 0; host_page_size(mach_host_self(), &page);
        vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) == KERN_SUCCESS)
            d.ram_available_bytes = (uint64_t)(vm.free_count + vm.inactive_count) * page;
    }
    d.has_fp16    = sysctl_bool("hw.optional.arm.FEAT_FP16");
    d.has_dotprod = sysctl_bool("hw.optional.arm.FEAT_DotProd");
    d.has_i8mm    = sysctl_bool("hw.optional.arm.FEAT_I8MM");
    d.has_sve     = false;

#elif defined(__linux__)
    std::string soc;
#  if defined(__ANDROID__)
    char prop[PROP_VALUE_MAX] = {0};
    std::string manu, model;
    if (__system_property_get("ro.soc.manufacturer", prop) > 0) manu = prop;
    if (__system_property_get("ro.soc.model", prop) > 0) model = prop;
    if (!model.empty()) soc = manu.empty() ? model : manu + " " + model;
    if (soc.empty() && __system_property_get("ro.hardware", prop) > 0) soc = prop;
#  endif
    if (soc.empty()) soc = read_cpuinfo_field("Hardware");
    if (soc.empty()) soc = read_cpuinfo_field("model name");
    copy_str(d.soc, sizeof d.soc, soc.empty() ? "unknown" : soc);

    d.ram_total_bytes     = read_meminfo_kb("MemTotal") * 1024ull;
    d.ram_available_bytes = read_meminfo_kb("MemAvailable") * 1024ull;
    if (!d.ram_total_bytes) { struct sysinfo si; if (sysinfo(&si) == 0) d.ram_total_bytes = (uint64_t)si.totalram * si.mem_unit; }

#  if defined(__aarch64__)
    unsigned long hw1 = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    d.has_fp16    = (hw1 & HWCAP_FPHP) != 0;
    d.has_dotprod = (hw1 & HWCAP_ASIMDDP) != 0;
    d.has_sve     = (hw1 & HWCAP_SVE) != 0;
#    ifdef HWCAP2_I8MM
    d.has_i8mm    = (hw2 & HWCAP2_I8MM) != 0;
#    else
    (void)hw2;
#    endif
#  endif
    // big.LITTLE: group cores by max freq. "Fast" = every core not in the slowest cluster.
    // On Tensor G3 that is X3 + 4x A715 (5 cores); the 4x A510 are excluded.
    {
        std::vector<long> freqs(d.cpu_cores, 0);
        long minf = 0, maxf = 0;
        for (int i = 0; i < d.cpu_cores && i < 64; ++i) {
            char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
            std::ifstream f(p); long v = 0; if (f >> v) freqs[i] = v;
            if (v) { if (!minf || v < minf) minf = v; if (v > maxf) maxf = v; }
        }
        if (maxf > minf) {
            int fast = 0; uint64_t mask = 0;
            for (int i = 0; i < d.cpu_cores && i < 64; ++i)
                if (freqs[i] > minf) { ++fast; mask |= (1ull << i); }
            d.cpu_big_cores = fast; d.cpu_fast_mask = mask;
        }
    }
#endif
    if (d.cpu_big_cores <= 0) d.cpu_big_cores = d.cpu_cores;
}

} // namespace kiln
