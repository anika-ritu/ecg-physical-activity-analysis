"""Statistical, clustering, and classification analyses."""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mannwhitneyu, pearsonr, spearmanr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .features import SELECTED_ECG_FEATURES


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d using the pooled sample standard deviation."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    pooled_variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (
        len(a) + len(b) - 2
    )
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_variance)) if pooled_variance > 0 else np.nan


def feature_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compare selected ECG features with MVPA and activity category."""
    rows = []
    for feature in SELECTED_ECG_FEATURES:
        subset = data[[feature, "mvpa_minutes", "activity_category"]].dropna()
        active = subset.loc[subset["activity_category"] == "Active", feature].to_numpy()
        sedentary = subset.loc[subset["activity_category"] == "Sedentary", feature].to_numpy()
        if len(active) < 2 or len(sedentary) < 2:
            continue
        pearson_r, pearson_p = pearsonr(subset["mvpa_minutes"], subset[feature])
        spearman_r, spearman_p = spearmanr(subset["mvpa_minutes"], subset[feature])
        t_stat, t_p = ttest_ind(active, sedentary, equal_var=False)
        u_stat, u_p = mannwhitneyu(active, sedentary, alternative="two-sided")
        f_stat, f_p = f_oneway(active, sedentary)
        rows.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "anova_f": f_stat,
                "anova_p": f_p,
                "welch_t": t_stat,
                "welch_t_p": t_p,
                "mann_whitney_u": u_stat,
                "mann_whitney_p": u_p,
                "cohens_d": cohens_d(active, sedentary),
            }
        )
    return pd.DataFrame(rows)


def cluster_and_classify(data: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create two cardiovascular clusters and predict them from MVPA minutes."""
    columns = SELECTED_ECG_FEATURES + ["mvpa_minutes"]
    complete = data.dropna(subset=columns).copy()
    if len(complete) < 4:
        raise ValueError("At least four complete participants are required.")

    scaled = StandardScaler().fit_transform(complete[columns])
    labels = KMeans(n_clusters=2, random_state=random_state, n_init=20).fit_predict(scaled)
    complete["cluster"] = labels

    x_mvpa = complete[["mvpa_minutes"]].to_numpy()
    classifier = LogisticRegression(random_state=random_state).fit(x_mvpa, labels)
    predicted = classifier.predict(x_mvpa)
    metrics = {
        "silhouette_score": float(silhouette_score(scaled, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(scaled, labels)),
        "logistic_accuracy": float(accuracy_score(labels, predicted)),
        "logistic_precision_weighted": float(precision_score(labels, predicted, average="weighted", zero_division=0)),
        "logistic_recall_weighted": float(recall_score(labels, predicted, average="weighted", zero_division=0)),
        "logistic_f1_weighted": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
    }
    return complete, metrics
