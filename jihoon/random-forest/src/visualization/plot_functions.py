import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distribution(df: pd.DataFrame, feature: str) -> None:
    """Plot distribution of a single feature."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature].dropna(), kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap of features."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    # 테스트용 샘플
    sample_data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "feature3": [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(sample_data)
    plot_feature_distribution(df, "feature1")
    plot_correlation_heatmap(df)
