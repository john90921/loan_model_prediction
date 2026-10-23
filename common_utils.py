
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def display_importances(feature_importance_df_, top_n=40):
    # Average importance across folds for each feature
    importance = (
        feature_importance_df_
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, max(8, top_n * 0.32)))

    sns.barplot(
        data=importance,
        x="importance",
        y="feature",
        color="steelblue"
    )

    plt.title(f"Top {top_n} Features by Average Importance")
    plt.xlabel("Average importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
    plt.show()

    return importance
