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
    *   Generate a QALY (Quality-Adjusted Life Year) matrix using Python code. (See also Prompt 2 for detailed QALY data handling if needed).
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
    *   Use Python code to generate mortality probability data based on standard actuarial tables. (See also Prompt 3 for specific table generation).
    *   The data should cover ages 65-85 showing corresponding survival probabilities.
    *   Output: Include both the Python code and the resulting mortality table. Incorporate this information clearly and non-alarmingly into the main health report.
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

Important Constraints:

*   Transparency: Always document assumptions clearly.
*   Conservatism: Prefer conservative estimation methods.
*   Bias Avoidance: Avoid introducing bias through imputation.
*   Rationale: Provide clear justification for each imputation decision.

---

## Prompt 3: Mortality Probability Table Generation

Goal: Generate a table showing age-specific survival probabilities based on standard actuarial data.

AI Role: You are a data analysis assistant capable of retrieving and processing statistical data.

Task: Create a clear table displaying survival probabilities for a specified age range using Python.

Instructions:

1.  Data Source: Use standard actuarial life tables (e.g., from a public health agency, social security administration, or reputable demographic research source). Clearly state the source used.
2.  Age Range: Generate data for ages 65 through 85, inclusive.
3.  Metric: Calculate and display the survival probability for each year within the specified age range. This typically represents the probability of an individual of a given age surviving to the next year, or cumulative survival from a base age. Specify which probability is being presented.
4.  Technology: Use Python (libraries like `pandas` recommended) to process the data and generate the table.
5.  Output Format:
    *   Provide the commented Python code used for data retrieval/calculation and table generation.
    *   Present the final data as a clean, readable table (e.g., Markdown table or structured output) with columns for 'Age' and 'Survival Probability'.
    *   Include a brief note about the source and year of the actuarial data used.

Example Output Structure:

```python
# Python code to generate mortality table
# Source: [Name of Actuarial Table Source and Year]
import pandas as pd

# ... (code for data loading/calculation) ...

# Create DataFrame
data = {'Age': [65, 66, ..., 85], 'Survival Probability': [prob_65, prob_66, ..., prob_85]}
mortality_table = pd.DataFrame(data)

print(mortality_table.to_markdown(index=False))
```

```markdown
| Age | Survival Probability |
|-----|----------------------|
| 65  | 0.XXXXX              |
| 66  | 0.XXXXX              |
| ... | ...                  |
| 85  | 0.XXXXX              |

*Data based on [Name of Actuarial Table Source and Year]. Survival probability represents [e.g., the probability of surviving to the next birthday].*
```

Constraints:

*   Ensure the data source is credible and cited.
*   Present the information in a neutral, factual manner.
