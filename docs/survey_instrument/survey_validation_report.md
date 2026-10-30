# Survey Instrument Validation Report

**Status:** Draft instrument structurally validated; not yet faculty/content validated or field-tested.

## Scope and Counts

- Four Likert questions for each of six constructs: PE, EE, HM, SI, PP, BI = **24**.
- Smartphone aspect satisfaction: battery, camera, performance, design, display = **5**.
- Overall smartphone satisfaction = **1**.
- Optional open-text questions for PE, PV, BI = **3**.
- Total respondent-facing items listed in the codebook = **33** (30 five-point agreement items and 3 optional paragraph items).
- Proposed construct-specific SSR-style anchor statements = **30** (five for each of six constructs; separate CSV).
- Likert labels are consistent and ordered 1–5; all current items are forward-scored, with reverse-scoring status `No`.

## Structural Checks

The generated deliverables are checked after creation for exact required item counts, unique question IDs/export column names, valid five-point scales, valid rating values 1–5, five anchors per construct, and explicit anchor provenance labeling. The codebook uses export names PE01, EE01, HM01, SI01, PP01, BI01, five AS_* fields, SAT01, and three FT_* fields.

## Construct and Method Notes

- The 24 Likert questions cover only the six requested constructs. Price Value (PV) has one optional open-text question and no added Likert items, preserving the requested 24-item structure.
- **Perceived Protection** is provisionally interpreted as perceived protection of personal data/accounts by the respondent's current smartphone. This is a substantive operationalization assumption and needs faculty/framework confirmation before fielding.
- All items refer to the respondent's current smartphone. No identifying fields are requested; consent and eligibility instructions belong in the form build and must follow approved institutional wording.
- The five anchors per construct are newly written proposed smartphone-domain adaptations. They are not the original six Maier et al. purchase-intent anchor sets. The paper's exact original anchor text remains unavailable; see [original_anchor_source_audit.md](../ssr_verification/original_anchor_source_audit.md).
- The proposed construct anchors have not been embedded, scored, or empirically validated. No Google Form was created, no human responses were collected, and no personas or synthetic survey responses were generated.

## Required Before Fielding

1. Confirm construct definitions and item-to-construct mapping with faculty, especially Perceived Protection and Behavioral Intention's current-use/replacement framing.
2. Conduct expert content review and cognitive interviews with target smartphone users; revise only with documented approval.
3. Pilot the final instrument, inspect completion time, missingness, response distributions, and item comprehension; then assess reliability/validity using the planned sample and analysis protocol.
4. Obtain ethics/consent approval and verify Google Forms privacy/export settings before launch.
5. Independently validate any proposed SSR anchor adaptation against the intended constructs; do not claim it is the original paper's anchor set.
