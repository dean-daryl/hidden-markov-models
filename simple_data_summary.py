import pandas as pd
import numpy as np
import os


def analyze_real_data():
    """Simple analysis of the real sensor data"""
    print("=== REAL DATA ANALYSIS SUMMARY ===\n")

    # Load HMM-ready data
    if os.path.exists("hmm_ready_features.csv"):
        hmm_data = pd.read_csv("hmm_ready_features.csv")
        print(f"✅ HMM-ready data loaded: {hmm_data.shape}")

        # Activity distribution
        activity_counts = hmm_data["activity"].value_counts()
        print(f"\nActivity Distribution:")
        for activity, count in activity_counts.items():
            print(f"  {activity}: {count} windows")

        # Data balance check
        balance_ratio = activity_counts.min() / activity_counts.max()
        print(f"\nData Balance Ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("  ⚠️  Dataset is imbalanced - consider more data collection")
        else:
            print("  ✅ Dataset balance is reasonable")

        # Feature info
        feature_cols = [
            col
            for col in hmm_data.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]
        print(f"\nFeatures: {len(feature_cols)} selected for HMM")

        # Data quality
        missing = hmm_data[feature_cols].isnull().sum().sum()
        infinite = np.isinf(hmm_data[feature_cols]).sum().sum()
        print(f"Missing values: {missing}")
        print(f"Infinite values: {infinite}")

    else:
        print("❌ HMM data not found")

    # Load feature importance
    if os.path.exists("feature_importance.csv"):
        importance = pd.read_csv("feature_importance.csv")
        print(f"\n✅ Feature importance loaded: {importance.shape}")
        print(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(importance.head(5).iterrows()):
            print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")
    else:
        print("❌ Feature importance not found")

    # Load raw data info
    if os.path.exists("processed_data/raw_sensor_data.csv"):
        raw_data = pd.read_csv("processed_data/raw_sensor_data.csv")
        print(f"\n✅ Raw sensor data: {raw_data.shape}")

        # Sampling rate estimation
        time_diffs = raw_data["time"].diff().dropna()
        time_diffs_seconds = time_diffs / 1e9
        estimated_rate = 1.0 / time_diffs_seconds.median()
        print(f"Estimated sampling rate: {estimated_rate:.1f} Hz")

        # Raw data distribution
        raw_activity_counts = raw_data["activity"].value_counts()
        print(f"\nRaw Data Samples by Activity:")
        for activity, count in raw_activity_counts.items():
            duration = count / estimated_rate
            print(f"  {activity}: {count} samples ({duration:.1f} seconds)")
    else:
        print("❌ Raw sensor data not found")

    print(f"\n=== TRANSITION ANALYSIS ===")
    if "hmm_data" in locals():
        # Calculate empirical transition probabilities
        sorted_data = hmm_data.sort_values("start_time").reset_index(drop=True)
        activities = sorted_data["activity"].values
        unique_activities = sorted(hmm_data["activity"].unique())

        # Count transitions
        transitions = {}
        for i in range(len(activities) - 1):
            current = activities[i]
            next_act = activities[i + 1]
            key = f"{current} -> {next_act}"
            transitions[key] = transitions.get(key, 0) + 1

        print("Observed Transitions:")
        for transition, count in sorted(transitions.items()):
            print(f"  {transition}: {count}")

    print(f"\n=== FILES GENERATED ===")
    files_to_check = [
        "hmm_ready_features.csv",
        "feature_importance.csv",
        "processed_data/raw_sensor_data.csv",
        "processed_data/extracted_features.csv",
    ]

    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file}")

    print(f"\n=== NEXT STEPS FOR HMM ===")
    print("1. Use 'hmm_ready_features.csv' for HMM training")
    print("2. Implement Gaussian emission models for each activity")
    print("3. Initialize transition matrix with empirical probabilities")
    print("4. Train HMM using labeled windows")
    print("5. Implement Viterbi algorithm for sequence decoding")
    print("6. Evaluate on test sequences")

    if "hmm_data" in locals():
        return hmm_data, activity_counts
    return None, None


if __name__ == "__main__":
    analyze_real_data()
