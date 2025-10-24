import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os


class RealDataAnalyzer:
    def __init__(self):
        self.hmm_data = None
        self.feature_importance = None
        self.raw_data = None

    def load_processed_data(self):
        """Load all processed data files"""
        print("Loading processed data...")

        # Load HMM-ready features
        if os.path.exists("hmm_ready_features.csv"):
            self.hmm_data = pd.read_csv("hmm_ready_features.csv")
            print(f"Loaded HMM data: {self.hmm_data.shape}")

        # Load feature importance
        if os.path.exists("feature_importance.csv"):
            self.feature_importance = pd.read_csv("feature_importance.csv")
            print(f"Loaded feature importance: {self.feature_importance.shape}")

        # Load raw sensor data
        if os.path.exists("processed_data/raw_sensor_data.csv"):
            self.raw_data = pd.read_csv("processed_data/raw_sensor_data.csv")
            print(f"Loaded raw data: {self.raw_data.shape}")

    def analyze_data_distribution(self):
        """Analyze the distribution of activities and data quality"""
        if self.hmm_data is None:
            print("No HMM data loaded")
            return

        print("\n=== Data Distribution Analysis ===")

        # Activity distribution
        activity_counts = self.hmm_data["activity"].value_counts()
        print(f"\nActivity distribution:")
        for activity, count in activity_counts.items():
            print(f"  {activity}: {count} windows")

        # Data quality checks
        feature_cols = [
            col
            for col in self.hmm_data.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]

        # Check for missing values
        missing_values = self.hmm_data[feature_cols].isnull().sum().sum()
        print(f"\nMissing values: {missing_values}")

        # Check for infinite values
        infinite_values = np.isinf(self.hmm_data[feature_cols]).sum().sum()
        print(f"Infinite values: {infinite_values}")

        # Feature statistics
        print(f"\nFeature statistics:")
        print(f"Number of features: {len(feature_cols)}")
        print(f"Mean feature std: {self.hmm_data[feature_cols].std().mean():.4f}")
        print(
            f"Mean feature range: {(self.hmm_data[feature_cols].max() - self.hmm_data[feature_cols].min()).mean():.4f}"
        )

        return activity_counts, feature_cols

    def create_visualizations(self):
        """Create comprehensive visualizations of the real data"""
        if self.hmm_data is None:
            print("No data to visualize")
            return

        # Create plots directory
        os.makedirs("plots", exist_ok=True)

        feature_cols = [
            col
            for col in self.hmm_data.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]

        # 1. Activity distribution
        plt.figure(figsize=(10, 6))
        activity_counts = self.hmm_data["activity"].value_counts()
        bars = plt.bar(
            activity_counts.index,
            activity_counts.values,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        plt.title("Activity Distribution in Real Data", fontsize=14, fontweight="bold")
        plt.xlabel("Activity")
        plt.ylabel("Number of Windows")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            "plots/real_data_activity_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 2. Top feature importance
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            bars = plt.barh(
                range(len(top_features)), top_features["importance"], color="skyblue"
            )
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("Feature Importance")
            plt.title(
                "Top 15 Most Important Features (Random Forest)",
                fontsize=14,
                fontweight="bold",
            )
            plt.gca().invert_yaxis()

            # Add value labels
            for i, (bar, importance) in enumerate(
                zip(bars, top_features["importance"])
            ):
                plt.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{importance:.3f}",
                    va="center",
                    ha="left",
                )

            plt.tight_layout()
            plt.savefig(
                "plots/real_data_feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        # 3. Feature distributions by activity
        top_5_features = (
            self.feature_importance.head(5)["feature"].tolist()
            if self.feature_importance is not None
            else feature_cols[:5]
        )

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, feature in enumerate(top_5_features):
            if feature in self.hmm_data.columns:
                for activity in self.hmm_data["activity"].unique():
                    activity_data = self.hmm_data[
                        self.hmm_data["activity"] == activity
                    ][feature]
                    axes[i].hist(activity_data, alpha=0.7, label=activity, bins=8)

                axes[i].set_title(f"{feature}")
                axes[i].legend()
                axes[i].set_xlabel("Feature Value")
                axes[i].set_ylabel("Frequency")

        # Remove empty subplot
        if len(top_5_features) < 6:
            fig.delaxes(axes[5])

        plt.suptitle(
            "Feature Distributions by Activity (Real Data)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            "plots/real_data_feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 4. Correlation heatmap for top features
        top_10_features = (
            self.feature_importance.head(10)["feature"].tolist()
            if self.feature_importance is not None
            else feature_cols[:10]
        )
        available_features = [f for f in top_10_features if f in self.hmm_data.columns]

        if len(available_features) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.hmm_data[available_features].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": 0.8},
            )
            plt.title(
                "Feature Correlation Matrix (Top 10 Features)",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(
                "plots/real_data_correlation_matrix.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        # 5. Raw sensor data visualization (if available)
        if self.raw_data is not None:
            self.visualize_raw_sensor_data()

    def visualize_raw_sensor_data(self):
        """Visualize raw sensor data patterns"""
        if self.raw_data is None:
            return

        # Sample data for visualization (to avoid overcrowding)
        sample_size = min(1000, len(self.raw_data))
        sample_data = self.raw_data.sample(n=sample_size, random_state=42).sort_values(
            "time"
        )

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # Plot each axis
        for i, axis in enumerate(["x", "y", "z"]):
            for activity in sample_data["activity"].unique():
                activity_data = sample_data[sample_data["activity"] == activity]
                axes[i].scatter(
                    activity_data["seconds_elapsed"],
                    activity_data[axis],
                    alpha=0.6,
                    label=activity,
                    s=10,
                )

            axes[i].set_ylabel(f"{axis.upper()} Acceleration")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Combined magnitude
        sample_data["magnitude"] = np.sqrt(
            sample_data["x"] ** 2 + sample_data["y"] ** 2 + sample_data["z"] ** 2
        )
        for activity in sample_data["activity"].unique():
            activity_data = sample_data[sample_data["activity"] == activity]
            axes[3].scatter(
                activity_data["seconds_elapsed"],
                activity_data["magnitude"],
                alpha=0.6,
                label=activity,
                s=10,
            )

        axes[3].set_ylabel("Magnitude")
        axes[3].set_xlabel("Time (seconds)")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(
            "Raw Sensor Data Patterns by Activity", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("plots/real_raw_sensor_patterns.png", dpi=300, bbox_inches="tight")
        plt.show()

    def baseline_classification_performance(self):
        """Test baseline classification performance on real data"""
        if self.hmm_data is None:
            print("No data for classification test")
            return None

        print("\n=== Baseline Classification Performance ===")

        feature_cols = [
            col
            for col in self.hmm_data.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]

        X = self.hmm_data[feature_cols].values
        y = self.hmm_data["activity"].values

        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Split data (if we have enough samples)
        if len(self.hmm_data) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            # Use all data for training and testing (small dataset)
            X_train = X_test = X
            y_train = y_test = y
            print("Warning: Small dataset - using same data for train/test")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = rf.predict(X_test_scaled)

        # Performance metrics
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features used: {len(feature_cols)}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=rf.classes_,
            yticklabels=rf.classes_,
        )
        plt.title("Confusion Matrix - Real Data Classification")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            "plots/real_data_confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        return rf, scaler

    def calculate_transition_probabilities(self):
        """Calculate empirical transition probabilities from real data"""
        if self.hmm_data is None:
            print("No data for transition analysis")
            return None

        print("\n=== Transition Probability Analysis ===")

        # Sort by time to get sequence
        sorted_data = self.hmm_data.sort_values("start_time").reset_index(drop=True)
        activities = sorted_data["activity"].values

        # Get unique activities
        unique_activities = sorted(self.hmm_data["activity"].unique())
        n_states = len(unique_activities)

        # Initialize transition matrix
        transition_matrix = np.zeros((n_states, n_states))
        activity_to_idx = {activity: i for i, activity in enumerate(unique_activities)}

        # Count transitions
        for i in range(len(activities) - 1):
            current_state = activity_to_idx[activities[i]]
            next_state = activity_to_idx[activities[i + 1]]
            transition_matrix[current_state, next_state] += 1

        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(
            transition_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )

        # Create DataFrame for better visualization
        transition_df = pd.DataFrame(
            transition_matrix, index=unique_activities, columns=unique_activities
        )

        print("Empirical Transition Probabilities:")
        print(transition_df.round(3))

        # Visualize transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            transition_df,
            annot=True,
            cmap="Blues",
            fmt=".3f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Empirical Activity Transition Probabilities (Real Data)")
        plt.xlabel("To Activity")
        plt.ylabel("From Activity")
        plt.tight_layout()
        plt.savefig(
            "plots/real_data_transition_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        return transition_df, unique_activities

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("REAL DATA ANALYSIS SUMMARY REPORT")
        print("=" * 60)

        if self.hmm_data is not None:
            activity_counts, feature_cols = self.analyze_data_distribution()

            print(f"\nDataset Overview:")
            print(f"  Total windows: {len(self.hmm_data)}")
            print(f"  Total features: {len(feature_cols)}")
            print(f"  Activities: {list(activity_counts.keys())}")

            # Data balance
            max_count = activity_counts.max()
            min_count = activity_counts.min()
            balance_ratio = min_count / max_count
            print(
                f"  Data balance ratio: {balance_ratio:.2f} (1.0 = perfectly balanced)"
            )

            if balance_ratio < 0.5:
                print(
                    "  ⚠️  Warning: Imbalanced dataset - consider collecting more data for underrepresented activities"
                )

            # Feature quality
            mean_std = self.hmm_data[feature_cols].std().mean()
            if mean_std < 0.01:
                print(
                    "  ⚠️  Warning: Low feature variance - may indicate poor discriminability"
                )
            elif mean_std > 100:
                print("  ⚠️  Warning: High feature variance - consider feature scaling")
            else:
                print("  ✅ Feature variance looks good")

        else:
            print("No processed data found!")

        print("\nNext Steps for HMM Implementation:")
        print("1. Use the generated 'hmm_ready_features.csv' for HMM training")
        print("2. Initialize HMM with empirical transition probabilities")
        print("3. Estimate emission parameters from labeled data")
        print("4. Implement Viterbi algorithm for sequence decoding")
        print("5. Evaluate on held-out sequences")

        print("\nFiles Generated:")
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
                print(f"  ❌ {file} (missing)")


def main():
    """Main analysis function"""
    analyzer = RealDataAnalyzer()

    # Load data
    analyzer.load_processed_data()

    if analyzer.hmm_data is None:
        print(
            "No processed data found. Please run real_data_feature_extraction.py first."
        )
        return

    # Perform comprehensive analysis
    print("Performing comprehensive analysis of real data...")

    # 1. Data distribution analysis
    analyzer.analyze_data_distribution()

    # 2. Create visualizations
    analyzer.create_visualizations()

    # 3. Baseline classification performance
    analyzer.baseline_classification_performance()

    # 4. Transition probabilities
    analyzer.calculate_transition_probabilities()

    # 5. Summary report
    analyzer.generate_summary_report()

    print("\n✅ Real data analysis complete!")
    print("Check the 'plots/' directory for visualizations.")


if __name__ == "__main__":
    main()
