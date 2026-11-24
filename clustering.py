from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parent
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
SEED = 42
CHOSEN_K = 4


def credit_net(extras_spend):
    return np.where(extras_spend < 20, 5, -10)


def classify_member(member, scaler, model, segment_names, credit_segments):
    features = pd.DataFrame([member], columns=FEATURES)
    cluster = int(model.predict(scaler.transform(features))[0])
    return segment_names[cluster], cluster in credit_segments


def main():
    members = pd.read_csv(PROJECT / "gym_members.csv")
    features = members[FEATURES]
    train, test = train_test_split(features, test_size=0.30, random_state=SEED)

    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    models = {}
    fit_results = []
    for k in range(1, 9):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        model.fit(train_scaled)
        models[k] = model
        fit_results.append({
            "k": k,
            "training_spread": model.inertia_,
            "test_spread": -model.score(test_scaled),
        })

    fit = pd.DataFrame(fit_results).set_index("k")
    fig, ax = plt.subplots()
    ax.plot(fit.index, fit["training_spread"], marker="o", label="Training members")
    ax.plot(fit.index, fit["test_spread"], marker="o", label="Test members")
    ax.axvline(CHOSEN_K, color="gray", linestyle="--", label="Chosen k")
    ax.set(
        title="Gym member clustering by number of segments",
        xlabel="Number of segments (k)",
        ylabel="Sum of squared distances to nearest center",
        xticks=range(1, 9),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT / "elbow_plot.png", dpi=200)
    plt.close(fig)

    model = models[CHOSEN_K]
    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEATURES,
    )
    segment_names = {
        centers["visits_per_month"].idxmin(): "Newer light users",
        centers["months_member"].idxmax(): "Longtime regulars",
        centers["visits_per_month"].idxmax(): "Frequent gym users",
        centers["classes_per_month"].idxmax(): "Group class enthusiasts",
    }

    # Choose which segments get the offer using the training members.
    train_credit = pd.DataFrame({
        "cluster": model.predict(train_scaled),
        "net_if_credited": credit_net(train["monthly_extras_spend"]),
    })
    train_net = train_credit.groupby("cluster")["net_if_credited"].sum()
    credit_segments = set(train_net[train_net > 0].index)

    # Apply the finance assumptions to each member across the full dataset.
    assignments = members.copy()
    assignments["cluster"] = model.predict(scaler.transform(features))
    assignments["segment"] = assignments["cluster"].map(segment_names)
    assignments["gets_credit"] = assignments["cluster"].isin(credit_segments)
    assignments["baseline_net"] = credit_net(assignments["monthly_extras_spend"])
    assignments["targeted_net"] = assignments["baseline_net"].where(
        assignments["gets_credit"], 0
    )
    assignments.to_csv(PROJECT / "member_segments.csv", index=False)

    summary = centers.copy()
    summary.insert(0, "segment", summary.index.map(segment_names))
    summary["member_count"] = assignments.groupby("cluster").size()
    summary["gets_credit"] = summary.index.isin(credit_segments)
    summary["net_if_all_in_segment_credited"] = (
        assignments.groupby("cluster")["baseline_net"].sum()
    )
    summary.to_csv(PROJECT / "segment_summary.csv", index_label="cluster")

    new_member = {
        "age": 30,
        "visits_per_month": 4,
        "avg_visit_minutes": 42,
        "classes_per_month": 1,
        "months_member": 5,
        "monthly_extras_spend": 10,
    }
    new_segment, gets_credit = classify_member(
        new_member, scaler, model, segment_names, credit_segments
    )

    baseline = assignments["baseline_net"].sum()
    targeted = assignments["targeted_net"].sum()
    reduction = 1 - fit.loc[CHOSEN_K, "test_spread"] / fit.loc[1, "test_spread"]

    print(f"Rows: {len(members)} | Columns: {len(members.columns)}")
    print(f"Training: {len(train)} | Test: {len(test)}")
    print(f"Average extras spend: ${members['monthly_extras_spend'].mean():.2f}")
    print("\nSpread by k")
    print(fit.round(2).to_string())
    print(f"Test spread reduction at k={CHOSEN_K}: {reduction:.1%}")
    print("\nSegment centers and full-dataset counts")
    print(summary.round(2).to_string())
    print(f"\nNew member: {new_segment} | Gets credit: {gets_credit}")
    print(f"Baseline monthly net: ${baseline:,.0f}")
    print(f"Targeted monthly net: ${targeted:,.0f}")
    print(f"Monthly improvement: ${targeted - baseline:,.0f}")


if __name__ == "__main__":
    main()
    