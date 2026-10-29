"""
Visualization helpers (for reference)
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
