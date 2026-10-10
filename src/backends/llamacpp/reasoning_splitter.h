// Streams model output into ANSWER / REASONING channels by watching for the template's
// think tags, which may arrive split across token pieces.
#pragma once
#include "kiln/kiln.h"
#include <functional>
#include <string>
#include <vector>

namespace kiln {

class ReasoningSplitter {
public:
    using Emit = std::function<bool(kiln_channel, const std::string&)>;   // false = stop

    ReasoningSplitter(std::string start_tag, std::vector<std::string> end_tags, bool start_in_reasoning)
        : start_(std::move(start_tag)), ends_(std::move(end_tags)),
          in_reasoning_(start_in_reasoning && !start_.empty()), trim_leading_(start_in_reasoning) {}

    bool enabled() const { return !start_.empty() && !ends_.empty(); }
    bool in_reasoning() const { return in_reasoning_; }

    // Returns false if emit asked to stop.
    bool feed(const std::string& piece, const Emit& emit) {
        if (!enabled()) return emit(KILN_CHANNEL_ANSWER, piece);
        pending_ += piece;
        for (;;) {
            if (in_reasoning_) {
                size_t best = std::string::npos, best_len = 0;
                for (auto& e : ends_) { size_t p = pending_.find(e); if (p != std::string::npos && p < best) { best = p; best_len = e.size(); } }
                if (best != std::string::npos) {
                    if (best > 0 && !flush_to(best, KILN_CHANNEL_REASONING, emit)) return false;
                    pending_.erase(0, best_len);
                    in_reasoning_ = false; trim_leading_ = true;
                    continue;
                }
                size_t keep = max_partial_suffix(ends_);
                if (pending_.size() > keep && !flush_to(pending_.size() - keep, KILN_CHANNEL_REASONING, emit)) return false;
                return true;
            } else {
                size_t p = pending_.find(start_);
                if (p != std::string::npos) {
                    if (p > 0 && !flush_to(p, KILN_CHANNEL_ANSWER, emit)) return false;
                    pending_.erase(0, start_.size());
                    in_reasoning_ = true; trim_leading_ = true;
                    continue;
                }
                size_t keep = max_partial_suffix({start_});
                if (pending_.size() > keep && !flush_to(pending_.size() - keep, KILN_CHANNEL_ANSWER, emit)) return false;
                return true;
            }
        }
    }

    bool finish(const Emit& emit) {
        if (pending_.empty()) return true;
        return flush_to(pending_.size(), in_reasoning_ ? KILN_CHANNEL_REASONING : KILN_CHANNEL_ANSWER, emit);
    }

private:
    bool flush_to(size_t n, kiln_channel ch, const Emit& emit) {
        std::string out = pending_.substr(0, n);
        pending_.erase(0, n);
        if (trim_leading_) {
            size_t i = 0; while (i < out.size() && (out[i] == '\n' || out[i] == '\r' || out[i] == ' ')) ++i;
            if (i == out.size()) return true;    // all whitespace, keep trimming on next piece
            out.erase(0, i); trim_leading_ = false;
        }
        return emit(ch, out);
    }

    // longest suffix of pending_ that is a proper prefix of any tag
    size_t max_partial_suffix(const std::vector<std::string>& tags) const {
        size_t best = 0;
        for (auto& t : tags)
            for (size_t k = std::min(pending_.size(), t.size() - 1); k > best; --k)
                if (pending_.compare(pending_.size() - k, k, t, 0, k) == 0) { best = k; break; }
        return best;
    }

    std::string start_;
    std::vector<std::string> ends_;
    std::string pending_;
    bool in_reasoning_;
    bool trim_leading_;
};

} // namespace kiln
