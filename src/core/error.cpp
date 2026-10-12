#include "internal.h"
#include <cstdarg>
#include <cstdio>
#include <atomic>

namespace {
thread_local std::string g_last_error;
std::atomic<kiln_log_cb> g_log_cb{nullptr};
std::atomic<void*>       g_log_user{nullptr};
std::atomic<int>         g_log_level{KILN_LOG_WARN};

const char* k_status_names[] = {
    "ok", "invalid argument", "no backend", "model load failed", "out of memory",
    "context full", "cancelled", "backend error", "not supported"
};
}

namespace kiln {

kiln_status set_error(kiln_status s, const std::string& msg) {
    g_last_error = msg;
    if (s != KILN_OK) log(KILN_LOG_ERROR, "%s: %s", kiln_status_string(s), msg.c_str());
    return s;
}

void log(kiln_log_level lvl, const char* fmt, ...) {
    if (lvl > g_log_level.load()) return;
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    kiln_log_cb cb = g_log_cb.load();
    if (cb) cb(lvl, buf, g_log_user.load());
    else    fprintf(stderr, "[kiln] %s\n", buf);
}

} // namespace kiln

extern "C" {

const char* kiln_last_error(void) { return g_last_error.c_str(); }

const char* kiln_status_string(kiln_status s) {
    if (s == KILN_ERR_INTERNAL) return "internal error";
    if (s < 0 || s > KILN_ERR_NOT_SUPPORTED) return "unknown";
    return k_status_names[s];
}

const char* kiln_version(void) { return "0.3.1"; }

void kiln_set_log_callback(kiln_log_cb cb, void* user) { g_log_cb = cb; g_log_user = user; }
void kiln_set_log_level(kiln_log_level level) { g_log_level = level; }

}
