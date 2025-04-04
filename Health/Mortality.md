## Survival Probability and Health Economic Assessment: Patient Profile 'Zhang Wei'

### Executive Summary

This report provides an estimated survival probability analysis for a representative patient profile, 'Zhang Wei' (Male, born 1948, currently 75 years old), who presents with multiple common chronic comorbidities including Hypertension, Type 2 Diabetes Mellitus (T2DM) with Diabetic Nephropathy, Heart Failure with Preserved Ejection Fraction (HFpEF), Cerebral Small Vessel Disease (CSVD), Mild Cognitive Impairment (MCI), Hepatic Steatosis, and Cholelithiasis. Baseline survival is derived from standard Chinese mortality tables, adjusted based on the cumulative impact of these conditions using published hazard ratios. The analysis indicates a significantly reduced survival probability compared to the general male population of the same age. Furthermore, the report discusses the implications for Quality-Adjusted Life Years (QALYs) and the cost-effectiveness of managing these conditions, highlighting the substantial potential health and economic value of effective interventions.

### Patient Profile Recap: Zhang Wei

*   Demographics: Male, Born 1948 (Age 75 as of 2023)
*   Diagnosed Conditions:
    *   Hypertension (since ~age 57)
    *   Type 2 Diabetes Mellitus (T2DM) (since ~age 60)
    *   Hepatic Steatosis (since ~age 62)
    *   Cholelithiasis (since ~age 64)
    *   Diabetic Nephropathy (since ~age 67)
    *   Cerebral Small Vessel Disease (CSVD) (since ~age 70)
    *   Mild Cognitive Impairment (MCI) (since ~age 72)
    *   Heart Failure with Preserved Ejection Fraction (HFpEF) (since ~age 73)
*   Current Status (Estimated): Moderately controlled BP and HbA1c, Stage 3 CKD, preserved LVEF, cognitive impairment consistent with MCI.

### Methodology & Assumptions

1.  Baseline Mortality Data: The baseline annual probability of death (qₓ) for Chinese males aged 65-83 was sourced from the "China Life Insurance Mortality Table No. 1 (1990-1993) Non-pension Business – Male". While dated, this provides a structured reference point. Annual survival probability (pₓ) = 1 - qₓ. Cumulative survival S(x) = Product(pᵢ) for i = 65 to x-1.
2.  Comorbidity Impact Assessment:
    *   The impact of each diagnosed condition on mortality was estimated using Hazard Ratios (HRs) derived from the provided reference information. Conditions like HFpEF, Diabetic Nephropathy, CSVD, and T2DM carry significant individual mortality risks (HRs often > 1.5-2.0). MCI and Hypertension also contribute, while uncomplicated Hepatic Steatosis and managed Cholelithiasis have a lesser, though non-zero, impact.
    *   Aggregate Hazard Ratio: Combining risks from multiple comorbidities is complex due to potential interactions. A simple multiplicative approach likely overestimates risk. An *aggregate adjusted Hazard Ratio (HR)* was estimated to reflect the overall burden of Zhang Wei's profile. Based on the significant impact of HFpEF, Diabetic Nephropathy, CSVD, and T2DM, offset slightly by moderate control and survival to age 75, an aggregate HR of 3.8 was applied relative to the baseline mortality rate. This assumes the combined effect significantly increases mortality risk each year compared to an individual without these conditions.
3.  Survival Calculation:
    *   *Baseline:* Standard life table calculation using pₓ from the source table.
    *   *Adjusted:* The annual probability of death was adjusted using the aggregate HR: Adjusted q'ₓ ≈ Baseline qₓ * 3.8 (capped at 1.0). Adjusted p'ₓ = 1 - q'ₓ. Adjusted cumulative survival S'(x) = Product(p'ᵢ) for i = 65 to x-1.
4.  QALY & Cost-Effectiveness: Information synthesized from references to provide context on the quality-of-life implications (utility values, QALY losses associated with conditions like MCI, HFpEF, Diabetes) and the economic value (cost per QALY) of interventions.

### Estimated Survival Probability (Age 65-83)

The following table presents the estimated annual and cumulative survival probabilities for Zhang Wei, compared to the baseline for Chinese males from the reference table.

| Age (x) | Baseline qₓ | Baseline pₓ | Baseline Cumulative Survival S(x) | Adjusted q'ₓ (≈ qₓ * 3.8) | Adjusted p'ₓ | Adjusted Cumulative Survival S'(x) |
| :-----: | :---------: | :---------: | :-------------------------------: | :-----------------------: | :----------: | :--------------------------------: |
| 65  | 0.021912    | 0.978088    | 1.0000                            | 0.083266                  | 0.916734     | 1.0000                             |
| 66  | 0.024021    | 0.975979    | 0.9781                            | 0.091280                  | 0.908720     | 0.9167                             |
| 67  | 0.026310    | 0.973690    | 0.9546                            | 0.099978                  | 0.900022     | 0.8331                             |
| 68  | 0.028787    | 0.971213    | 0.9298                            | 0.109391                  | 0.890609     | 0.7498                             |
| 69  | 0.031462    | 0.968538    | 0.9032                            | 0.119556                  | 0.880444     | 0.6678                             |
| 70  | 0.034353    | 0.965647    | 0.8748                            | 0.130541                  | 0.869459     | 0.5879                             |
| 71  | 0.037481    | 0.962519    | 0.8448                            | 0.142428                  | 0.857572     | 0.5112                             |
| 72  | 0.040877    | 0.959123    | 0.8130                            | 0.155333                  | 0.844667     | 0.4383                             |
| 73  | 0.044578    | 0.955422    | 0.7798                            | 0.169396                  | 0.830604     | 0.3703                             |
| 74  | 0.048628    | 0.951372    | 0.7451                            | 0.184786                  | 0.815214     | 0.3075                             |
| 75  | 0.054501    | 0.945499    | 0.7090                            | 0.207104                  | 0.792896     | 0.2508                             |
| 76  | 0.059644    | 0.940356    | 0.6704                            | 0.226647                  | 0.773353     | 0.1988                             |
| 77  | 0.065173    | 0.934827    | 0.6304                            | 0.247657                  | 0.752343     | 0.1537                             |
| 78  | 0.071120    | 0.928880    | 0.5893                            | 0.270256                  | 0.729744     | 0.1157                             |
| 79  | 0.077522    | 0.922478    | 0.5474                            | 0.294584                  | 0.705416     | 0.0844                             |
| 80  | 0.084427    | 0.915573    | 0.5051                            | 0.320823                  | 0.679177     | 0.0596                             |
| 81  | 0.091892    | 0.908108    | 0.4624                            | 0.349190                  | 0.650810     | 0.0405                             |
| 82  | 0.099982    | 0.900018    | 0.4200                            | 0.379932                  | 0.620068     | 0.0263                             |
| 83  | 0.108777    | 0.891223    | 0.3780                            | 0.413353                  | 0.586647     | 0.0163                             |

*Note: Baseline qₓ from source table. Adjusted q'ₓ capped if qₓ * HR > 1. Cumulative survival starts at 1.0000 at age 65.*

Visual Representation:

![Estimated Cumulative Survival Probability (Age 65-83)](Estimated%20Cumulative%20Survival%20Probability%20(Age%2065-83).svg)
