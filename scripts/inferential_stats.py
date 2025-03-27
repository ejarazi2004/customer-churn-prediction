import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


from scripts.save_utils import save_fig


def chi_square_test(data, feature, target='churn', save_plot=False):
    contigency = pd.crosstab(data[feature], data[target])
    chi2, p, dof, expected = stats.chi2_contingency(contigency)
    
    print(f"\n-- Chi-Square Test: {feature} vs {target} --")
    print("Contigency Table:")
    print(contigency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4f}")
    print("Decision:", "Reject H-0 (dependent)" if p < 0.05 else "Fail to reject H-0 (independent)")
    
    if save_plot: 
        plt.figure(figsize=(6, 4))
        sns.countplot(data=data, x=feature, hue=target)
        plt.title(f"{feature.title()} vs {target.title()}")
        plt.tight_layout
        save_fig(subfolder="inferential",filename=f"inf-stat-{feature}-vs-{target}")
        
        

def test_num_features(data, feature, target='churn', alpha=0.05, save_plot=False):
    churned = data[data[target] == 1][feature].dropna()
    not_churned = data[data[target] == 0][feature].dropna()
    
    _, p_churned = stats.shapiro(churned)
    _, p_not_churned = stats.shapiro(not_churned)
    
    normal = p_churned > alpha and p_not_churned > alpha
    
    if normal:
        stat, p = stats.ttest_ind(churned, not_churned, equal_var=False)
        test_name = "t-Test (Welch)"
    else:
        stat, p = stats.mannwhitneyu(churned, not_churned, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    print(f"\n-- {test_name} for {feature} --")
    print(f"P-value: {p:.4f}")
    print("Reject H-0: Means are signigicantly different." if p < alpha else "Fail to reject H-0: No significant difference in distribution.")
    
    if save_plot:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=data, x=target, y=feature)
        plt.title(f"{feature.title()} vs {target.title()} ({test_name})")
        plt.tight_layout()
        plt.show()
        save_fig(f"inf-{feature}-vs-{target}","inferential")