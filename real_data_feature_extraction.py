import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RealDataFeatureExtractor:
    def __init__(
        self, data_dir="data", window_size=2.0, overlap=0.5, sampling_rate=None
    ):
        """
        Initialize feature extractor for real sensor data

        Args:
            data_dir: Directory containing the CSV files
            window_size: Window size in seconds
            overlap: Overlap ratio (0.0 to 1.0)
            sampling_rate: If None, will be estimated from data
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.raw_data = None
        self.features = None

    def load_real_data(self):
        """Load and combine all real sensor data files"""
        print("Loading real sensor data...")

        # Activity mapping
        activity_files = {
            "standing": "Standing_TotalAcceleration.csv",
            "walking": "Walking_TotalAcceleration.csv",
            "jumping": "Jumping_TotalAcceleration.csv",
            "still": "Still_TotalAcceleration.csv",
        }

        all_data = []

        for activity, filename in activity_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {activity} data from {filename}")
                df = pd.read_csv(filepath)

                # Add activity label
                df["activity"] = activity

                # Ensure we have the expected columns
                required_cols = ["time", "seconds_elapsed", "x", "y", "z"]
                if all(col in df.columns for col in required_cols):
                    all_data.append(df)
                    print(f"  - Loaded {len(df)} samples for {activity}")
                else:
                    print(f"  - Warning: Missing required columns in {filename}")
                    print(f"  - Available columns: {list(df.columns)}")
            else:
                print(f"Warning: File {filepath} not found")

        if not all_data:
            raise ValueError("No valid data files found!")

        # Combine all data
        self.raw_data = pd.concat(all_data, ignore_index=True)

        # Sort by timestamp to ensure proper ordering
        self.raw_data = self.raw_data.sort_values("time").reset_index(drop=True)

        # Estimate sampling rate if not provided
        if self.sampling_rate is None:
            self.sampling_rate = self._estimate_sampling_rate()

        print(f"Total samples loaded: {len(self.raw_data)}")
        print(f"Estimated sampling rate: {self.sampling_rate:.2f} Hz")
        print(f"Activities: {self.raw_data['activity'].value_counts().to_dict()}")

        return self.raw_data

    def _estimate_sampling_rate(self):
        """Estimate sampling rate from timestamp differences"""
        if len(self.raw_data) < 2:
            return 50.0  # Default fallback

        # Calculate time differences in nanoseconds and convert to seconds
        time_diffs = self.raw_data["time"].diff().dropna()

        # Convert nanoseconds to seconds
        time_diffs_seconds = time_diffs / 1e9

        # Calculate sampling rate as 1/median_time_diff
        median_diff = time_diffs_seconds.median()
        estimated_rate = 1.0 / median_diff if median_diff > 0 else 50.0

        return estimated_rate

    def extract_time_domain_features(self, window_data):
        """Extract time domain features from a window of sensor data"""
        features = {}

        # For each axis
        for axis in ["x", "y", "z"]:
            if axis in window_data.columns:
                data = window_data[axis].values
                prefix = f"{axis}_"

                # Basic statistics
                features[f"{prefix}mean"] = np.mean(data)
                features[f"{prefix}std"] = np.std(data)
                features[f"{prefix}var"] = np.var(data)
                features[f"{prefix}min"] = np.min(data)
                features[f"{prefix}max"] = np.max(data)
                features[f"{prefix}range"] = np.max(data) - np.min(data)
                features[f"{prefix}rms"] = np.sqrt(np.mean(data**2))

                # Percentiles
                features[f"{prefix}q25"] = np.percentile(data, 25)
                features[f"{prefix}q75"] = np.percentile(data, 75)
                features[f"{prefix}median"] = np.median(data)
                features[f"{prefix}iqr"] = np.percentile(data, 75) - np.percentile(
                    data, 25
                )

                # Higher order statistics
                if len(data) > 1:
                    features[f"{prefix}skew"] = stats.skew(data)
                    features[f"{prefix}kurtosis"] = stats.kurtosis(data)
                else:
                    features[f"{prefix}skew"] = 0
                    features[f"{prefix}kurtosis"] = 0

                # Energy
                features[f"{prefix}energy"] = np.sum(data**2)

                # Zero crossing rate
                zero_crossings = np.sum(np.diff(np.signbit(data)))
                features[f"{prefix}zcr"] = zero_crossings / len(data)

        # Cross-axis features
        if all(axis in window_data.columns for axis in ["x", "y", "z"]):
            x, y, z = (
                window_data["x"].values,
                window_data["y"].values,
                window_data["z"].values,
            )

            # Signal Magnitude Area (SMA)
            features["sma"] = np.mean(np.abs(x) + np.abs(y) + np.abs(z))

            # Signal Magnitude Vector (SMV)
            features["smv"] = np.mean(np.sqrt(x**2 + y**2 + z**2))

            # Correlations between axes
            if len(x) > 1:
                features["corr_xy"] = (
                    np.corrcoef(x, y)[0, 1]
                    if not np.isnan(np.corrcoef(x, y)[0, 1])
                    else 0
                )
                features["corr_xz"] = (
                    np.corrcoef(x, z)[0, 1]
                    if not np.isnan(np.corrcoef(x, z)[0, 1])
                    else 0
                )
                features["corr_yz"] = (
                    np.corrcoef(y, z)[0, 1]
                    if not np.isnan(np.corrcoef(y, z)[0, 1])
                    else 0
                )
            else:
                features["corr_xy"] = features["corr_xz"] = features["corr_yz"] = 0

        return features

    def extract_frequency_domain_features(self, window_data):
        """Extract frequency domain features from a window of sensor data"""
        features = {}

        for axis in ["x", "y", "z"]:
            if axis in window_data.columns:
                data = window_data[axis].values
                prefix = f"{axis}_"

                if len(data) > 1:
                    # FFT
                    fft_vals = fft(data)
                    fft_magnitude = np.abs(fft_vals[: len(fft_vals) // 2])
                    freqs = np.fft.fftfreq(len(data), 1 / self.sampling_rate)[
                        : len(fft_vals) // 2
                    ]

                    if len(fft_magnitude) > 0:
                        # Dominant frequency
                        dom_freq_idx = np.argmax(fft_magnitude)
                        features[f"{prefix}dom_freq"] = (
                            freqs[dom_freq_idx] if dom_freq_idx < len(freqs) else 0
                        )

                        # Spectral energy
                        features[f"{prefix}spectral_energy"] = np.sum(fft_magnitude**2)

                        # Spectral centroid
                        if np.sum(fft_magnitude) > 0:
                            features[f"{prefix}spectral_centroid"] = np.sum(
                                freqs * fft_magnitude
                            ) / np.sum(fft_magnitude)
                        else:
                            features[f"{prefix}spectral_centroid"] = 0

                        # Spectral rolloff (95th percentile)
                        cumsum = np.cumsum(fft_magnitude)
                        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                        features[f"{prefix}spectral_rolloff"] = (
                            freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                        )

                        # Frequency band energies
                        # Low frequency (0-5 Hz)
                        low_freq_mask = (freqs >= 0) & (freqs <= 5)
                        features[f"{prefix}low_freq_energy"] = np.sum(
                            fft_magnitude[low_freq_mask] ** 2
                        )

                        # Mid frequency (5-15 Hz)
                        mid_freq_mask = (freqs > 5) & (freqs <= 15)
                        features[f"{prefix}mid_freq_energy"] = np.sum(
                            fft_magnitude[mid_freq_mask] ** 2
                        )

                        # High frequency (>15 Hz)
                        high_freq_mask = freqs > 15
                        features[f"{prefix}high_freq_energy"] = np.sum(
                            fft_magnitude[high_freq_mask] ** 2
                        )

                        # Spectral entropy
                        if np.sum(fft_magnitude) > 0:
                            psd_norm = fft_magnitude / np.sum(fft_magnitude)
                            psd_norm = psd_norm[
                                psd_norm > 0
                            ]  # Remove zeros for log calculation
                            features[f"{prefix}spectral_entropy"] = -np.sum(
                                psd_norm * np.log2(psd_norm)
                            )
                        else:
                            features[f"{prefix}spectral_entropy"] = 0
                    else:
                        # Set default values if FFT failed
                        for feat in [
                            "dom_freq",
                            "spectral_energy",
                            "spectral_centroid",
                            "spectral_rolloff",
                            "low_freq_energy",
                            "mid_freq_energy",
                            "high_freq_energy",
                            "spectral_entropy",
                        ]:
                            features[f"{prefix}{feat}"] = 0
                else:
                    # Set default values for single sample windows
                    for feat in [
                        "dom_freq",
                        "spectral_energy",
                        "spectral_centroid",
                        "spectral_rolloff",
                        "low_freq_energy",
                        "mid_freq_energy",
                        "high_freq_energy",
                        "spectral_entropy",
                    ]:
                        features[f"{prefix}{feat}"] = 0

        return features

    def create_sliding_windows(self):
        """Create sliding windows from the sensor data"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_real_data() first.")

        print(
            f"Creating sliding windows (window_size={self.window_size}s, overlap={self.overlap})..."
        )

        # Calculate window parameters
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.overlap))

        print(f"Window size: {window_samples} samples")
        print(f"Step size: {step_samples} samples")

        windows = []

        # Group by activity to maintain activity labels
        for activity in self.raw_data["activity"].unique():
            activity_data = self.raw_data[self.raw_data["activity"] == activity].copy()
            activity_data = activity_data.sort_values("time").reset_index(drop=True)

            print(f"\nProcessing {activity} data ({len(activity_data)} samples)...")

            # Create windows for this activity
            start_idx = 0
            activity_windows = 0

            while start_idx + window_samples <= len(activity_data):
                end_idx = start_idx + window_samples
                window_data = activity_data.iloc[start_idx:end_idx]

                # Add window info
                window_info = {
                    "window_id": len(windows),
                    "activity": activity,
                    "start_time": window_data["time"].iloc[0],
                    "end_time": window_data["time"].iloc[-1],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "data": window_data,
                }

                windows.append(window_info)
                activity_windows += 1
                start_idx += step_samples

            print(f"  Created {activity_windows} windows for {activity}")

        print(f"\nTotal windows created: {len(windows)}")
        return windows

    def extract_features_from_windows(self, windows):
        """Extract features from all windows"""
        print("Extracting features from windows...")

        all_features = []

        for i, window_info in enumerate(windows):
            if i % 20 == 0:
                print(f"Processing window {i + 1}/{len(windows)}")

            window_data = window_info["data"]

            # Extract features
            time_features = self.extract_time_domain_features(window_data)
            freq_features = self.extract_frequency_domain_features(window_data)

            # Combine all features
            feature_vector = {
                "window_id": window_info["window_id"],
                "activity": window_info["activity"],
                "start_time": window_info["start_time"],
                "end_time": window_info["end_time"],
                **time_features,
                **freq_features,
            }

            all_features.append(feature_vector)

        self.features = pd.DataFrame(all_features)
        print(f"Feature extraction complete. Shape: {self.features.shape}")

        return self.features

    def analyze_features(self, save_plots=True):
        """Analyze extracted features"""
        if self.features is None:
            raise ValueError(
                "No features extracted. Call extract_features_from_windows() first."
            )

        print("Analyzing extracted features...")

        # Get feature columns (exclude metadata)
        feature_cols = [
            col
            for col in self.features.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]

        print(f"Total features: {len(feature_cols)}")
        print(f"Activities: {self.features['activity'].value_counts().to_dict()}")

        # Feature statistics
        feature_stats = self.features[feature_cols].describe()
        print("\nFeature statistics:")
        print(feature_stats)

        if save_plots:
            # Create plots directory
            os.makedirs("plots", exist_ok=True)

            # Plot 1: Activity distribution
            plt.figure(figsize=(10, 6))
            self.features["activity"].value_counts().plot(kind="bar")
            plt.title("Activity Distribution in Windows")
            plt.xlabel("Activity")
            plt.ylabel("Number of Windows")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    "plots/activity_distribution.png", dpi=300, bbox_inches="tight"
                )
            plt.show()

            # Plot 2: Feature distributions for key features
            key_features = ["x_mean", "y_mean", "z_mean", "sma", "smv"]
            available_key_features = [f for f in key_features if f in feature_cols]

            if available_key_features:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()

                for i, feature in enumerate(available_key_features[:6]):
                    for activity in self.features["activity"].unique():
                        activity_data = self.features[
                            self.features["activity"] == activity
                        ][feature]
                        axes[i].hist(activity_data, alpha=0.7, label=activity, bins=20)
                    axes[i].set_title(f"{feature} Distribution")
                    axes[i].legend()

                plt.tight_layout()
                if save_plots:
                    plt.savefig(
                        "plots/key_feature_distributions.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.show()

        return feature_stats

    def select_features_for_hmm(self, n_features=20, method="random_forest"):
        """Select best features for HMM using feature importance"""
        if self.features is None:
            raise ValueError(
                "No features extracted. Call extract_features_from_windows() first."
            )

        print(f"Selecting top {n_features} features using {method}...")

        # Get feature columns and target
        feature_cols = [
            col
            for col in self.features.columns
            if col not in ["window_id", "activity", "start_time", "end_time"]
        ]

        X = self.features[feature_cols].values
        y = self.features["activity"].values

        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if method == "random_forest":
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Get feature importance
            importance_scores = rf.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": importance_scores}
            ).sort_values("importance", ascending=False)

            print("Top 10 most important features:")
            print(feature_importance.head(10))

            # Select top features
            selected_features = feature_importance.head(n_features)["feature"].tolist()

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Create HMM-ready dataset
        hmm_data = self.features[
            ["window_id", "activity", "start_time", "end_time"] + selected_features
        ].copy()

        print(f"Selected {len(selected_features)} features for HMM")
        print("Selected features:", selected_features)

        return hmm_data, selected_features, feature_importance

    def save_processed_data(self, output_dir="processed_data"):
        """Save all processed data"""
        os.makedirs(output_dir, exist_ok=True)

        if self.raw_data is not None:
            print(f"Saving raw data to {output_dir}/raw_sensor_data.csv")
            self.raw_data.to_csv(f"{output_dir}/raw_sensor_data.csv", index=False)

        if self.features is not None:
            print(f"Saving features to {output_dir}/extracted_features.csv")
            self.features.to_csv(f"{output_dir}/extracted_features.csv", index=False)


def main():
    """Main function to run the complete feature extraction pipeline"""
    print("=== Real Data Feature Extraction Pipeline ===\n")

    # Initialize extractor
    extractor = RealDataFeatureExtractor(
        data_dir="data",
        window_size=2.0,  # 2 second windows
        overlap=0.5,  # 50% overlap
        sampling_rate=None,  # Will be estimated from data
    )

    try:
        # Step 1: Load real data
        raw_data = extractor.load_real_data()

        # Step 2: Create sliding windows
        windows = extractor.create_sliding_windows()

        # Step 3: Extract features
        features = extractor.extract_features_from_windows(windows)

        # Step 4: Analyze features
        feature_stats = extractor.analyze_features(save_plots=True)

        # Step 5: Select features for HMM
        hmm_data, selected_features, feature_importance = (
            extractor.select_features_for_hmm(n_features=20)
        )

        # Step 6: Save all data
        extractor.save_processed_data()

        # Save HMM-ready data
        print("Saving HMM-ready features to hmm_ready_features.csv")
        hmm_data.to_csv("hmm_ready_features.csv", index=False)

        # Save feature importance
        print("Saving feature importance to feature_importance.csv")
        feature_importance.to_csv("feature_importance.csv", index=False)

        print("\n=== Feature Extraction Complete ===")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Features shape: {features.shape}")
        print(f"HMM-ready data shape: {hmm_data.shape}")
        print(f"Selected {len(selected_features)} features for HMM")

        return extractor, hmm_data, selected_features

    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    extractor, hmm_data, selected_features = main()
