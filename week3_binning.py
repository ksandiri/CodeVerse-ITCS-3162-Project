"""
Week 3 - Investigation: Cleaned & Binned Dataset (Deliverable 3)
=================================================================
ITCS 3162 - Employee Retention & Job Satisfaction Analysis

Purpose
-------
Turn the raw HR dataset (JobSatisfaction.csv) into a Boolean (0/1) feature
matrix that the Apriori algorithm can mine for association rules in Week 4.
Apriori only works with "items" that are either present (1) or absent (0),
so every column - numeric or text - has to become a set of yes/no columns.

This script is the Python version of the following KNIME workflow:

    CSV Reader -> String Manipulation (clean) -> Auto-Binner -> Rule Engine -> One to Many

Pipeline steps
--------------
  1. Clean       - Standardize text categories and validate types, ranges and
                   missing values so later steps can trust the data.
  2. Auto-Binner - Equal-frequency (quantile) bins for the CONTINUOUS metrics:
                   SatisfactionLevel, LastEvaluation, AverageMonthlyHours
                   -> Low / Medium / High
  3. Rule Engine - Hand-written, domain-based rules for the DISCRETE metrics
                   and the 0/1 flags:
                   NumberProject, TimeSpendCompany, WorkAccident,
                   LeftCompany, PromotionLast5years
  4. One to Many - Every categorical column becomes one 0/1 column per value
                   (one-hot encoding), producing the Boolean feature matrix.

How to run
----------
    python week3_binning.py

Requires: pandas  (install with  python -m pip install pandas)

Outputs (written to outputs/week3/)
-----------------------------------
    cleaned_binned_dataset.csv  - original columns + the new _Bin / _Label columns
    boolean_feature_matrix.csv  - 0/1 matrix used as Apriori input in Week 4
    auto_binner_summary.csv     - bin edges, counts and percentages per bin
    rule_engine_summary.csv     - rule text, counts and percentages per label
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Build paths relative to this file so the script works no matter which
# folder it is run from.
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "JobSatisfaction.csv"          # raw input dataset
OUT = ROOT / "outputs" / "week3"               # folder for all Week 3 outputs
OUT.mkdir(parents=True, exist_ok=True)         # create it if it doesn't exist yet

# The 10 columns we expect in the raw file, in a fixed order.
# Used both to validate the input and to keep column order consistent.
EXPECTED = [
    "SatisfactionLevel", "LastEvaluation", "NumberProject", "AverageMonthlyHours",
    "TimeSpendCompany", "WorkAccident", "LeftCompany", "PromotionLast5years",
    "Department", "Salary",
]

# Continuous (many distinct values) columns -> handled by the Auto-Binner.
CONTINUOUS = ["SatisfactionLevel", "LastEvaluation", "AverageMonthlyHours"]

# Labels for the three equal-frequency bins, lowest third to highest third.
BIN_LABELS = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# 1. Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and validate the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw data exactly as read from JobSatisfaction.csv.

    Returns
    -------
    pd.DataFrame
        A cleaned copy with columns in EXPECTED order.

    Raises
    ------
    AssertionError
        If the columns are wrong, values are missing, or any value falls
        outside its valid range. Failing loudly here is intentional - it is
        better to stop than to silently encode bad data.
    """
    # Make sure the file has exactly the 10 columns we expect (no more, no less).
    # The ^ (symmetric difference) shows which columns are extra or missing.
    assert set(df.columns) == set(EXPECTED), f"Unexpected columns: {set(df.columns) ^ set(EXPECTED)}"

    # Work on a copy (so the raw data is never modified) with a fixed column order.
    df = df[EXPECTED].copy()

    # Standardize text categories: strip stray spaces and lowercase everything.
    # Without this, 'IT' and 'it' (or 'RandD' and 'randd') would be treated as
    # different departments and become separate Boolean columns later.
    for c in ["Department", "Salary"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # No missing values allowed - a blank cell would break binning and encoding.
    assert df.notna().all().all(), "Missing values found - inspect before encoding."

    # The three flag columns must contain only 0 (no) or 1 (yes).
    for c in ["WorkAccident", "LeftCompany", "PromotionLast5years"]:
        assert set(df[c].unique()) <= {0, 1}, f"{c} has non-binary values"

    # Satisfaction and evaluation scores are proportions, so they must be 0-1.
    assert df["SatisfactionLevel"].between(0, 1).all(), "SatisfactionLevel outside 0-1"
    assert df["LastEvaluation"].between(0, 1).all(), "LastEvaluation outside 0-1"

    # Salary should only have the three known levels.
    assert set(df["Salary"].unique()) <= {"low", "medium", "high"}, "Unknown Salary level"

    # NOTE on duplicates: the file contains ~3,000 identical rows, but we KEEP them.
    # There is no employee ID column, so two identical rows can legitimately be two
    # different employees with the same profile. Dropping them would lower the
    # support counts that Apriori relies on and bias the rules.
    return df


# ---------------------------------------------------------------------------
# 2. Auto-Binner (equal-frequency / quantile binning)
# ---------------------------------------------------------------------------
def auto_bin(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discretize each continuous metric into Low / Medium / High.

    This mirrors the KNIME Auto-Binner node in "Sample quantiles" mode:
    the cut points are the 33rd and 67th percentiles, so each bin holds
    roughly one third of the employees. Equal-frequency bins are used
    instead of equal-width bins so no bin ends up nearly empty, which would
    make its items too rare for Apriori to find.

    Because the cut points are real data values, employees with the same
    value always land in the same bin (bins can therefore be slightly
    uneven, e.g. 34% / 32% / 33%).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data from clean().

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        1. The data with a new "<column>_Bin" column for each continuous metric.
        2. A summary table describing every bin (edges, counts, percentages).
    """
    rows = []  # one summary row per (feature, bin) pair
    for c in CONTINUOUS:
        # pd.qcut splits the column into q=3 quantile-based bins.
        # retbins=True also returns the bin edges so we can document them.
        binned, edges = pd.qcut(df[c], q=3, labels=BIN_LABELS, retbins=True)

        # Store the label as plain text (e.g. "Low") in a new column.
        df[f"{c}_Bin"] = binned.astype(str)

        # Record each bin's boundaries and size for the report / README.
        for i, lab in enumerate(BIN_LABELS):
            vals = df.loc[df[f"{c}_Bin"] == lab, c]  # values that fell in this bin
            rows.append({
                "Feature": c,
                "Bin": lab,
                "Lower_edge": round(edges[i], 4),      # theoretical lower cut point
                "Upper_edge": round(edges[i + 1], 4),  # theoretical upper cut point
                "Observed_min": vals.min(),            # smallest actual value in the bin
                "Observed_max": vals.max(),            # largest actual value in the bin
                "Count": len(vals),
                "Percent": round(100 * len(vals) / len(df), 2),
            })
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Rule Engine (rule-based labeling)
# ---------------------------------------------------------------------------
# For discrete columns, statistical bins don't make much sense (e.g. a
# quantile split of NumberProject would put 4 projects in its own bin),
# so we use simple business rules instead.
#
# Each feature maps to an ordered list of (condition, label) pairs.
# Rules are checked top-down and the FIRST matching condition wins, exactly
# like a KNIME Rule Engine node, e.g.:
#       $NumberProject$ <= 2 => "Underloaded"
#       $NumberProject$ <= 5 => "Normal"
#       TRUE                 => "Overloaded"
# The final `lambda v: True` acts as the catch-all default (KNIME's TRUE =>).
RULES = {
    # Workload: 2 projects is light, 3-5 is typical, 6+ is a heavy load.
    "NumberProject": [
        (lambda v: v <= 2, "Underloaded"),       # 2 projects
        (lambda v: v <= 5, "Normal"),            # 3-5 projects
        (lambda v: True,   "Overloaded"),        # 6-7 projects
    ],
    # Tenure: 2-3 years is early career here, 4-6 mid, 7+ long-serving.
    "TimeSpendCompany": [
        (lambda v: v <= 3, "Early"),             # 2-3 years
        (lambda v: v <= 6, "Mid"),               # 4-6 years
        (lambda v: True,   "Veteran"),           # 7+ years
    ],
    # The 0/1 flags are relabeled as Yes/No so the Boolean column names
    # are readable (e.g. "LeftCompany_Yes" instead of "LeftCompany_1").
    "WorkAccident":        [(lambda v: v == 1, "Yes"), (lambda v: True, "No")],
    "LeftCompany":         [(lambda v: v == 1, "Yes"), (lambda v: True, "No")],
    "PromotionLast5years": [(lambda v: v == 1, "Yes"), (lambda v: True, "No")],
}

# Human-readable version of each rule set, saved in rule_engine_summary.csv
# so the report shows exactly which thresholds were used.
RULE_TEXT = {
    "NumberProject": "<=2 => Underloaded; 3-5 => Normal; >=6 => Overloaded",
    "TimeSpendCompany": "<=3 => Early; 4-6 => Mid; >=7 => Veteran",
    "WorkAccident": "1 => Yes; 0 => No",
    "LeftCompany": "1 => Yes; 0 => No",
    "PromotionLast5years": "1 => Yes; 0 => No",
}


def apply_rules(value, rules):
    """Return the label of the first rule whose condition is true for `value`.

    Parameters
    ----------
    value : int or float
        A single cell value, e.g. 4 for NumberProject.
    rules : list of (callable, str)
        Ordered (condition, label) pairs from RULES.

    Raises
    ------
    ValueError
        If no rule matches. This should never happen because every rule list
        ends with a catch-all, but it guards against editing mistakes.
    """
    for cond, label in rules:
        if cond(value):
            return label
    raise ValueError(f"No rule matched {value!r}")


def rule_engine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply RULES to every rule-based column.

    Parameters
    ----------
    df : pd.DataFrame
        Data after auto_bin().

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        1. The data with a new "<column>_Label" column for each rule-based feature.
        2. A summary table with the rule text and how many rows got each label.
    """
    rows = []  # one summary row per (feature, label) pair
    for c, rules in RULES.items():
        out = f"{c}_Label"

        # Run every value in the column through its rule list.
        df[out] = df[c].map(lambda v: apply_rules(v, rules))

        # Count how many employees received each label (largest first).
        for lab, n in df[out].value_counts().items():
            rows.append({
                "Feature": c,
                "Rule": RULE_TEXT[c],
                "Label": lab,
                "Count": n,
                "Percent": round(100 * n / len(df), 2),
            })
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. One to Many (Boolean / one-hot encoding)
# ---------------------------------------------------------------------------
def one_to_many(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the labeled data into a Boolean (0/1) feature matrix.

    This mirrors the KNIME One to Many node. First we gather one categorical
    "item" column per original feature (10 in total). Then each distinct
    value becomes its own column named "<Feature>_<Value>" that is 1 if the
    employee has that value and 0 otherwise. Example:

        Salary = "low"   ->   Salary_high=0, Salary_low=1, Salary_medium=0

    Parameters
    ----------
    df : pd.DataFrame
        Data after rule_engine() (must contain the _Bin and _Label columns).

    Returns
    -------
    pd.DataFrame
        The 0/1 matrix: one row per employee, one column per item.
    """
    # Collect exactly one categorical column per original feature.
    items = pd.DataFrame(index=df.index)
    for c in CONTINUOUS:                 # 3 features from the Auto-Binner
        items[c] = df[f"{c}_Bin"]
    for c in RULES:                      # 5 features from the Rule Engine
        items[c] = df[f"{c}_Label"]
    items["Department"] = df["Department"]   # already categorical (cleaned text)
    items["Salary"] = df["Salary"]           # already categorical (cleaned text)

    # One-hot encode: each (feature, value) pair becomes a 0/1 column.
    # dtype=int gives 0/1 instead of True/False, which is what Apriori expects.
    matrix = pd.get_dummies(items, prefix_sep="_", dtype=int)

    # Integrity checks:
    #  - every cell must be 0 or 1
    #  - every row must have exactly one 1 per original feature, so each
    #    row sums to 10 (no employee is missing a value or has two values
    #    for the same feature)
    assert matrix.isin([0, 1]).all().all(), "Matrix contains values other than 0/1"
    assert (matrix.sum(axis=1) == len(EXPECTED)).all(), "A row does not have exactly one item per feature"
    return matrix


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the full Week 3 pipeline and save all outputs."""
    # Step 0: read the raw data.
    raw = pd.read_csv(SOURCE)

    # Steps 1-4: clean -> bin -> label -> encode.
    df = clean(raw)
    df, bin_summary = auto_bin(df)
    df, rule_summary = rule_engine(df)
    matrix = one_to_many(df)

    # Save everything. index=False keeps pandas' row numbers out of the CSVs.
    df.to_csv(OUT / "cleaned_binned_dataset.csv", index=False)
    matrix.to_csv(OUT / "boolean_feature_matrix.csv", index=False)
    bin_summary.to_csv(OUT / "auto_binner_summary.csv", index=False)
    rule_summary.to_csv(OUT / "rule_engine_summary.csv", index=False)

    # Print a short report to the console so results can be checked at a glance.
    print(f"Rows: {len(df):,}  |  Original features: {len(EXPECTED)}  |  Boolean columns: {matrix.shape[1]}")
    print("\nAuto-Binner (equal frequency):")
    print(bin_summary.to_string(index=False))
    print("\nRule Engine:")
    print(rule_summary.to_string(index=False))

    # Support = share of employees that have each item (column mean of 0/1 values).
    # Useful preview for Week 4: items with very low support (e.g. PromotionLast5years_Yes)
    # will only show up in rules if Apriori's minimum support is set low enough.
    print("\nBoolean column support (share of rows = 1):")
    print(matrix.mean().round(3).sort_values(ascending=False).to_string())
    print(f"\nSaved to {OUT}")


# Only run the pipeline when this file is executed directly
# (not when it is imported by another script, e.g. the Week 4 Apriori code).
if __name__ == "__main__":
    main()
