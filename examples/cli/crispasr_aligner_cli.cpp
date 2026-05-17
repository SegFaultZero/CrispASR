// crispasr_aligner_cli.cpp — CLI aligner adapter.
//
// Calls the shared library's `crispasr_align_words` and converts the
// resulting `CrispasrAlignedWord` vector to the CLI's `crispasr_word`
// shape consumed by crispasr_run.cpp and downstream output formatters.

#include "crispasr_aligner_cli.h"

static std::vector<crispasr_word> crispasr_words_from_lib(std::vector<CrispasrAlignedWord> lib) {
    std::vector<crispasr_word> out;
    out.reserve(lib.size());
    for (auto& w : lib) {
        crispasr_word cw;
        cw.text = std::move(w.text);
        cw.t0 = w.t0_cs;
        cw.t1 = w.t1_cs;
        out.push_back(std::move(cw));
    }
    return out;
}

std::vector<crispasr_word> crispasr_ctc_align(const std::string& aligner_model, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs, int n_threads) {
    auto lib = crispasr_align_words(aligner_model, transcript, samples, n_samples, t_offset_cs, n_threads);
    return crispasr_words_from_lib(std::move(lib));
}

std::vector<crispasr_word> crispasr_ctc_align(crispasr_aligner_runtime* runtime, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs) {
    auto lib = crispasr_align_words_runtime(runtime, transcript, samples, n_samples, t_offset_cs);
    return crispasr_words_from_lib(std::move(lib));
}
