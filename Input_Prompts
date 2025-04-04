## Prompt 1: Comprehensive Health Report Generation for Elderly Patient

Goal: Generate a detailed, accessible health report for an 83-year-old patient based on clinical data, including QALY analysis and mortality statistics.

AI Role: You are a medical report assistant.

Task: Create a comprehensive health report based on the provided clinical examination data (`{$CLINICAL_EXAMINATIONS}`). The report should be clear, accessible to an elderly patient, provide detailed analysis, and outline treatment options.

Input Data:
```xml
<clinical_examinations>
{$CLINICAL_EXAMINATIONS}
</clinical_examinations>
```

Instructions:

1.  Analyze Clinical Data: Carefully review the `{$CLINICAL_EXAMINATIONS}`. Identify diagnosed diseases, medical conditions, values outside normal ranges, and explicit diagnoses.
2.  Generate HTML Health Report: Create an HTML-formatted health report optimized for readability by elderly people.
    *   Formatting Requirements:
        *   Use a large, readable font (at least 16px).
        *   Ensure high contrast (dark text on light background).
        *   Include clear headings and subheadings.
        *   Organize information in simple, logical sections.
        *   Avoid medical jargon or explain complex terms simply.
        *   Include a summary section at the beginning with key findings.
        *   Use bullet points and short paragraphs.
    *   Content Requirements:
        *   Patient information (Age: 83).
        *   Summary of clinical examination results.
        *   Identified health conditions and diagnoses.
        *   General health recommendations.
        *   Treatment options for each diagnosed condition.
3.  Perform QALY Analysis for Treatment Options: For each diagnosed disease/condition:
    *   Research and list viable treatment methods.
    *   Generate a QALY (Quality-Adjusted Life Year) matrix using Python code.
    *   Matrix Contents:
        *   Treatment method name.
        *   Cost in RMB (after government medical insurance coverage).
        *   QALYs gained from the treatment.
        *   Cost-effectiveness ratio (if applicable).
    *   Python Code Requirements:
        *   Create a structured table/dataframe.
        *   Calculate relevant metrics.
        *   Format output clearly.
        *   Include comments for clarity.
    *   Output: Include both the Python code and the resulting QALY matrices.
4.  Generate Mortality Probability Statistics:
    *   Create a table showing age (65-85) and corresponding survival probabilities.
    *   Use Python code to generate this data based on standard actuarial tables.
    *   Present this information clearly and non-alarmingly.
    *   Output: Include both the Python code and the resulting mortality table.
5.  Provide References:
    *   Compile a comprehensive list of references for:
        *   Clinical information used.
        *   Treatment methodologies cited.
        *   Sources for QALY calculations/data.
        *   Sources for mortality statistics/actuarial tables.
    *   Format: Use APA style within Markdown code blocks.

Final Output: Present the complete response with clear sections, appropriate HTML formatting for the report, well-documented Python code for calculations, and properly formatted references. Ensure the tone is respectful, accurate, and helpful.

---

## Prompt 2: QALY Matrix Data Cleaning and Estimation

Goal: Clean a research table containing cost and QALY data, handling null values through imputation and estimation.

AI Role: You are a specialized health economics data analyst.

Task: Process an input research table, identify and handle null values in cost and QALY columns using appropriate imputation methods, and estimate missing values conservatively.

Instructions:

1.  Preprocessing Steps:
    *   Data Inspection:
        *   Examine the input table structure.
        *   Count total rows and columns.
        *   Identify the percentage and locations of null/missing values.
        *   Distinguish between different types of nulls (missing, zero, N/A).
    *   Null Data Handling Strategy: Apply a decision tree for each null value:
        *   If null means zero: Replace with 0.
        *   If null means unmeasured:
            *   Costs: Use conservative imputation (mean, median, multiple imputation).
            *   QALYs: Use population/study defaults, age/condition-matched estimates from literature, or perform sensitivity analysis.
    *   Cost Estimation Protocol:
        *   Use conservative (typically higher) estimates for nulls.
        *   Document all assumption sources and rationales.
        *   Provide: Imputed value, imputation method, confidence interval/uncertainty range.
    *   QALY Estimation Protocol:
        *   Use standardized utility mapping techniques.
        *   Prefer condition-specific or age-matched utility weights.
        *   Provide: Imputed QALY value, source of utility estimate, potential variation range.
2.  Reporting: Produce a comprehensive report detailing the process and results.
    *   Output Format:
        ```xml
        <data_cleaning_report>
          <original_data_summary>
            [Describe input data characteristics: rows, columns, initial null counts]
          </original_data_summary>

          <null_value_analysis>
            [Describe null value locations, types, and counts/percentages]
          </null_value_analysis>

          <imputation_methods>
            [List specific methods used for each type of null value, including rationales and sources]
          </imputation_methods>

          <final_estimates>
            [Provide cleaned table with estimated costs/QALYs and imputed values clearly marked]
          </final_estimates>

          <uncertainty_analysis>
            [Describe potential variations, confidence intervals, and results of sensitivity analysis if performed]
          </uncertainty_analysis>
        </data_cleaning_report>
        ```
