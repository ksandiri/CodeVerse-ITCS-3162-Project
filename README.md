# CodeVerse-ITCS-3162-Project
Employee Retention &amp; Job Satisfaction Analysis

Dataset
Input: JobSatisfaction.csv (14,999 employee records; 10 original features)
Topics: Satisfaction, performance evaluation, project load, working hours, tenure, workplace accidents, attrition, promotions, department, and salary category
Data Quality: No missing values in the supplied dataset

Week 1 — Team Collaboration and Infrastructure
Established a GitHub repository for project files and version control.
Collaborated with team members and assigned team roles and tasks.

Week 2 — Data Audit, Distributions, and Metadata
Used Python (pandas and matplotlib) to examine data types, missing values, distinct values, and descriptive statistics.
Created distribution bar charts for Salary, PromotionLast5years, WorkAccident, Department, and LeftCompany.
Added HR_Metadata_Dictionary.csv, defining each of the 10 HR fields, its data type, and its unique values or range.
Added HR_Unique_Value_Frequencies.csv, detailing unique values and their frequencies for each feature.
Supporting outputs include data_audit.csv and descriptive_statistics.csv.

Week 3 — Investigation: Cleaned & Binned Dataset (Deliverable 3: Processed Boolean Feature Matrix)
Script: week3_binning.py (Python version of the KNIME Auto-Binner -> Rule Engine -> One to Many workflow). Run with `python week3_binning.py`.
Cleaning: trimmed and lowercased Department and Salary (e.g. "IT" -> "it"); checked for missing values, binary flags, and valid 0–1 ranges. The 3,008 duplicate rows were kept because the data has no employee ID, so identical rows may be different employees.
Auto-Binner (equal frequency): SatisfactionLevel, LastEvaluation, and AverageMonthlyHours were each split into Low / Medium / High bins of about 33% each.
- SatisfactionLevel: Low ≤ 0.52, Medium 0.53–0.76, High ≥ 0.77
- LastEvaluation: Low ≤ 0.61, Medium 0.62–0.82, High ≥ 0.83
- AverageMonthlyHours: Low ≤ 168, Medium 169–232, High ≥ 233
Rule Engine (rule-based labeling):
- NumberProject: ≤ 2 Underloaded, 3–5 Normal, ≥ 6 Overloaded
- TimeSpendCompany: ≤ 3 Early, 4–6 Mid, ≥ 7 Veteran
- WorkAccident, LeftCompany, PromotionLast5years: 1 Yes, 0 No
One to Many (Boolean encoding): the 10 labeled features became a 14,999 × 34 matrix of 0/1 values. Each row has exactly one 1 per original feature, so every row sums to 10.
Outputs (outputs/week3/): cleaned_binned_dataset.csv, boolean_feature_matrix.csv (input for Apriori in Week 4), auto_binner_summary.csv, rule_engine_summary.csv.

Contribution Log
Member: Krishika Sandiri, 
Work Completed: Set up the GitHub repository and completed the technical distribution audit, HR metadata definitions, and unique value analysis.

Member: Emmanuel Onamade
Work Completed: Team Organization, Data Binning, Boolean Transformation

Member: Sowmya Ravuri
Work Completed: Distribution Analyzation
