// Cleans up what a speech model writes before the caller sees it.
// Qwen3-ASR does not answer with the words alone. It first names the language it heard:
//     language English<asr_text>The train leaves at seven.
// The caller wants "The train leaves at seven." and, separately, "English". The header may
// arrive split across token pieces, so the first pieces are held back until it is clear
// whether they are a header. Models that write the words directly pass through unchanged.
#pragma once
#include "kiln/kiln.h"
#include <algorithm>
#include <string>

namespace kiln {

class TranscriptFilter {
public:
    TranscriptFilter(kiln_token_cb cb, void* user) : cb_(cb), user_(user) {}

    // Has the signature of kiln_token_cb, with a TranscriptFilter as `user`.
    static bool on_piece(kiln_channel, const char* piece, int32_t, void* self) {
        return static_cast<TranscriptFilter*>(self)->feed(piece);
    }

    // Returns false if the caller's callback asked to stop.
    bool feed(const std::string& piece) {
        if (passing_) return emit(piece);
        pending_ += piece;
        const size_t tag = pending_.find(kTag);
        if (tag != std::string::npos) {
            if (pending_.compare(0, kLang.size(), kLang) == 0) language_ = trim(pending_.substr(kLang.size(), tag - kLang.size()));
            std::string rest = pending_.substr(tag + kTag.size());
            pending_.clear();
            passing_ = true;
            return emit(rest);
        }
        if (may_be_header()) return true;
        return flush();
    }

    // Call when the model has stopped: hands over anything still held back.
    bool finish() { return passing_ ? true : flush(); }

    // The language as the model named it ("English"), or empty when it did not say.
    const std::string& language() const { return language_; }
    bool stopped() const { return stopped_; }

private:
    // True while what has arrived so far could still turn out to be "language X<asr_text>".
    bool may_be_header() const {
        if (pending_.size() > kMaxHeader || pending_.find('\n') != std::string::npos) return false;
        const size_t n = std::min(pending_.size(), kLang.size());
        if (pending_.compare(0, n, kLang, 0, n) == 0) return true;      // "lang", "language Eng"
        return kTag.compare(0, pending_.size(), pending_) == 0;         // "<asr_", for a model that skips the language
    }

    bool flush() {
        std::string all;
        all.swap(pending_);
        passing_ = true;
        return emit(all);
    }

    bool emit(const std::string& s) {
        // the words start right after the header, or after a space the model put first
        size_t from = 0;
        if (!started_) {
            while (from < s.size() && (s[from] == ' ' || s[from] == '\n')) ++from;
            if (from == s.size()) return true;
            started_ = true;
        }
        if (!cb_) return true;
        if (!cb_(KILN_CHANNEL_ANSWER, s.c_str() + from, -1, user_)) { stopped_ = true; return false; }
        return true;
    }

    static std::string trim(std::string s) {
        while (!s.empty() && s.back() == ' ') s.pop_back();
        size_t b = 0;
        while (b < s.size() && s[b] == ' ') ++b;
        return s.substr(b);
    }

    static constexpr size_t kMaxHeader = 64;
    static inline const std::string kLang = "language ";
    static inline const std::string kTag  = "<asr_text>";

    kiln_token_cb cb_;
    void*         user_;
    std::string   pending_;
    std::string   language_;
    bool          passing_ = false;   // header dealt with; pieces go straight through
    bool          started_ = false;   // first visible character has been sent
    bool          stopped_ = false;
};

} // namespace kiln
