#pragma once
// When --batch-mtp / STRATA_BATCH_MTP=1 can run.  One home for the rule, so the engine and its test agree.
namespace strata::program {

/// nullptr when each batch slot may verify an MTP proposal beside its current token, else the reason it may not.
/// A layer split is allowed since 0.1.41: the slots' drafters live on the last stage (beside the solo drafter) and the
/// stage chain carries grouped rows.  Only a split whose stages share one GPU is still refused.
inline const char* batch_mtp_refusal(int batch, bool has_mtp, int spec, bool split_same, bool serve) {
    if (batch < 2) return "it needs --batch 2 or more";
    if (!has_mtp) return "it needs --mtp";
    if (spec < 2) return "it needs --spec T (T >= 2)";
    if (split_same) return "it needs each stage of a layer split on its own GPU";
    if (!serve) return "it needs --serve";
    return nullptr;
}

}  // namespace strata::program
