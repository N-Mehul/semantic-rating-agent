# PP Construct Mismatch Impact Analysis and Resolution Proposal

## Audit Overview

| Construct_Code   | Canonical_Definition                                  | Synthetic_Prompt_Definition                                  |   Total_Affected_Responses | Affected_Persona_IDs               | Properly_Excluded_In_Advanced_Optimization   | Impact_On_Global_Wasserstein_If_Included     | Recommended_Scientific_Resolution                                                                                                                              |
|:-----------------|:------------------------------------------------------|:-------------------------------------------------------------|---------------------------:|:-----------------------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PP               | Perceived Protection (Security, privacy, data safety) | Price / Perceived Value (Affordability, worth, cost-benefit) |                         57 | P1 (16), P2 (12), P3 (13), P4 (16) | True                                         | W1 increases from 0.4881 to 0.4905 (+0.0024) | Execute separate regeneration of 57 responses under authentic Perceived Protection prompts, or re-label and evaluate them against genuine Price Value anchors. |

### Impact Assessment:
- In the advanced optimization pipeline, the 57 PP responses were **successfully excluded** from the valid benchmark set ($N=343$), preventing distortion of calibration parameters.
- However, in Phase 3, anchor statements were embedded as 'Perceived Protection' while synthetic responses discussed price.

### Formal Resolution Proposal:
1. **Do not modify the existing 400 responses** to maintain audit integrity.
2. Create a separate dataset `synthetic_responses_v2.csv` where the 57 PP responses are regenerated using genuine Perceived Protection prompts (e.g. data encryption, biometric security, malware defense).
3. Alternatively, create a validated 'Price Value' (PV) anchor set and score the 57 responses under PV.
