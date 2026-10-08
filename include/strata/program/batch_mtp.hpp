#pragma once
// When --batch-mtp / STRATA_BATCH_MTP=1 can run, and how wide its windows are.  One home for the rules, so the engine
// and its test agree.
namespace strata::program {

/// nullptr when each batch slot may verify an MTP proposal beside its current token, else the reason it may not.
/// A layer split is allowed since 0.1.41: the slots' drafters live on the last stage (beside the solo drafter) and the
/// stage chain carries grouped rows.  Refused: stages sharing one GPU, and --batch-groups G > 1 on a split (the
/// pipelined groups carry one row per slot: they never verify a draft, and --batch > 8 would overrun their buffers).
inline const char* batch_mtp_refusal(int batch, bool has_mtp, int spec, bool split_same, bool serve,
                                     int batch_groups = 1, bool has_split = false) {
    if (batch < 2) return "it needs --batch 2 or more";
    if (!has_mtp) return "it needs --mtp";
    if (spec < 2) return "it needs --spec T (T >= 2)";
    if (split_same) return "it needs each stage of a layer split on its own GPU";
    if (has_split && batch_groups > 1) return "it does not combine with --batch-groups on a layer split";
    if (!serve) return "it needs --serve";
    return nullptr;
}

/// The row capacity every stage's verifier is sized for: --batch-mtp windows hold two rows per slot, up to the
/// verifier's maximum (`verify_max_t`, kVerifyMaxT); otherwise the larger of the draft width and the slot count.
inline int batch_window_rows(bool batch_mtp, int spec, int batch, int verify_max_t) {
    if (batch_mtp) return verify_max_t;
    return spec > batch ? spec : batch;
}

}  // namespace strata::program
