import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# 1. Data Segmentation
def segment_data(df, group_col, value1, value2):
    """
    Segment data into two groups based on a feature.
    """
    group_a = df[df[group_col] == value1]
    group_b = df[df[group_col] == value2]
    return group_a, group_b

# 2. Perform Statistical Test
def perform_test(group_a, group_b, metric_col, test_type="t-test"):
    """
    Perform statistical test between two groups.
    """
    if test_type == "t-test":
        stat, p_value = stats.ttest_ind(group_a[metric_col], group_b[metric_col], equal_var=False)
    elif test_type == "chi2":
        contingency = pd.crosstab(group_a[metric_col], group_b[metric_col])
        stat, p_value, _, _ = stats.chi2_contingency(contingency)
    else:
        raise ValueError("Unsupported test type. Use 't-test' or 'chi2'.")
    return p_value

# 3. Interpret Results
def interpret_results(p_value, alpha=0.05):
    """
    Interpret the p-value to accept or reject the null hypothesis.
    """
    if p_value < alpha:
        return "Reject the null hypothesis (statistically significant)."
    else:
        return "Fail to reject the null hypothesis (not statistically significant)."

# 4. A/B Hypothesis Testing Wrapper
def ab_hypothesis_testing(df, group_col, value1, value2, metric_col, test_type="t-test"):
    """
    Conduct A/B Hypothesis Testing.
    Args:
        df (DataFrame): Input dataset.
        group_col (str): Column to use for segmentation.
        value1: Value representing Group A.
        value2: Value representing Group B.
        metric_col (str): Column to test.
        test_type (str): Type of test ("t-test", "chi2").
    Returns:
        float: p-value from the statistical test.
    """
    # Segment data
    group_a, group_b = segment_data(df, group_col, value1, value2)
    
    # Perform statistical test
    p_value = perform_test(group_a, group_b, metric_col, test_type)
    
    # Interpret results
    result = interpret_results(p_value)
    print(f"Test Results for {metric_col}:")
    print(f"P-value: {p_value:.4f}")
    print(f"Conclusion: {result}")
    
    return p_value


def mannwhitney_test(group_a, group_b, metric_col):
    stat, p_value = mannwhitneyu(group_a[metric_col], group_b[metric_col])
    return p_value


