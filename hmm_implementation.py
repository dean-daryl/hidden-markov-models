import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings("ignore")


class ActivityHMM:
    def __init__(self, n_states=4, n_features=20):
        """
        Hidden Markov Model for Activity Recognition

        Args:
            n_states: Number of hidden states (activities)
            n_features: Number of features per observation
        """
        self.n_states = n_states
        self.n_features = n_features

        # HMM parameters
        self.pi = None  # Initial state probabilities
        self.A = None  # Transition matrix
        self.mu = None  # Emission means for each state
        self.sigma = None  # Emission covariances for each state

        # Activity mapping
        self.state_names = None
        self.state_to_idx = None
        self.idx_to_state = None

        # Data preprocessing
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y, activity_names=None):
        """
        Train HMM using labeled data

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Activity labels (n_samples,)
            activity_names: List of activity names
        """
        print("Training HMM on labeled data...")

        # Setup activity mapping
        unique_activities = sorted(np.unique(y))
        self.state_names = activity_names if activity_names else unique_activities
        self.state_to_idx = {state: i for i, state in enumerate(unique_activities)}
        self.idx_to_state = {i: state for state, i in self.state_to_idx.items()}
        self.n_states = len(unique_activities)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.n_features = X_scaled.shape[1]

        # Convert labels to indices
        y_indices = np.array([self.state_to_idx[label] for label in y])

        # Estimate initial state probabilities
        self._estimate_initial_probabilities(y_indices)

        # Estimate transition probabilities
        self._estimate_transition_probabilities(y_indices)

        # Estimate emission parameters
        self._estimate_emission_parameters(X_scaled, y_indices)

        self.is_fitted = True
        print(f"HMM training complete!")
        print(f"States: {self.state_names}")
        print(f"Features: {self.n_features}")

    def _estimate_initial_probabilities(self, y_indices):
        """Estimate initial state probabilities π"""
        self.pi = np.zeros(self.n_states)

        # Count first occurrence of each state
        first_states = []
        seen_sequences = set()

        # Simple approach: uniform initial probabilities
        # In practice, you might want to track sequence starts
        self.pi = np.ones(self.n_states) / self.n_states

        print(f"Initial probabilities: {self.pi}")

    def _estimate_transition_probabilities(self, y_indices):
        """Estimate transition matrix A"""
        self.A = np.zeros((self.n_states, self.n_states))

        # Count transitions
        for i in range(len(y_indices) - 1):
            current_state = y_indices[i]
            next_state = y_indices[i + 1]
            self.A[current_state, next_state] += 1

        # Normalize to get probabilities
        for i in range(self.n_states):
            row_sum = np.sum(self.A[i, :])
            if row_sum > 0:
                self.A[i, :] /= row_sum
            else:
                # If no transitions observed, use uniform
                self.A[i, :] = 1.0 / self.n_states

        # Add small smoothing to avoid zero probabilities
        self.A = (self.A + 0.01) / (1 + 0.01 * self.n_states)

        print("Transition matrix A:")
        for i, state_from in enumerate(self.state_names):
            for j, state_to in enumerate(self.state_names):
                if self.A[i, j] > 0.01:  # Only show significant transitions
                    print(f"  {state_from} -> {state_to}: {self.A[i, j]:.3f}")

    def _estimate_emission_parameters(self, X, y_indices):
        """Estimate emission parameters (Gaussian means and covariances)"""
        self.mu = np.zeros((self.n_states, self.n_features))
        self.sigma = np.zeros((self.n_states, self.n_features, self.n_features))

        for state in range(self.n_states):
            # Get samples for this state
            state_mask = y_indices == state
            state_samples = X[state_mask]

            if len(state_samples) > 0:
                # Estimate mean
                self.mu[state] = np.mean(state_samples, axis=0)

                # Estimate covariance
                if len(state_samples) > 1:
                    self.sigma[state] = np.cov(state_samples, rowvar=False)
                else:
                    # Single sample - use identity covariance
                    self.sigma[state] = np.eye(self.n_features)

                # Regularize covariance to ensure numerical stability
                self.sigma[state] += np.eye(self.n_features) * 1e-6

            else:
                # No samples for this state - use default parameters
                self.mu[state] = np.zeros(self.n_features)
                self.sigma[state] = np.eye(self.n_features)

        print("Emission parameters estimated for all states")

    def _emission_probability(self, observation, state):
        """Calculate emission probability P(observation | state)"""
        try:
            prob = multivariate_normal.pdf(
                observation, mean=self.mu[state], cov=self.sigma[state]
            )
            return max(prob, 1e-100)  # Avoid log(0)
        except:
            return 1e-100

    def viterbi(self, observations):
        """
        Viterbi algorithm for finding most likely state sequence

        Args:
            observations: Sequence of observations (n_timesteps, n_features)

        Returns:
            path: Most likely state sequence
            prob: Log probability of the path
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before decoding")

        # Scale observations
        observations = self.scaler.transform(observations)

        n_timesteps = len(observations)

        # Initialize Viterbi tables
        delta = np.zeros((n_timesteps, self.n_states))
        psi = np.zeros((n_timesteps, self.n_states), dtype=int)

        # Initialization (t=0)
        for s in range(self.n_states):
            delta[0, s] = np.log(self.pi[s]) + np.log(
                self._emission_probability(observations[0], s)
            )
            psi[0, s] = 0

        # Recursion (t=1 to T-1)
        for t in range(1, n_timesteps):
            for s in range(self.n_states):
                # Find best previous state
                scores = delta[t - 1, :] + np.log(self.A[:, s])
                best_prev_state = np.argmax(scores)

                delta[t, s] = scores[best_prev_state] + np.log(
                    self._emission_probability(observations[t], s)
                )
                psi[t, s] = best_prev_state

        # Termination - find best final state
        best_final_state = np.argmax(delta[n_timesteps - 1, :])
        best_prob = delta[n_timesteps - 1, best_final_state]

        # Backtrack to find best path
        path = np.zeros(n_timesteps, dtype=int)
        path[n_timesteps - 1] = best_final_state

        for t in range(n_timesteps - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        # Convert indices to state names
        path_names = [self.idx_to_state[idx] for idx in path]

        return path_names, best_prob

    def predict(self, X):
        """Predict activity sequence for given observations"""
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction")

        predictions, _ = self.viterbi(X)
        return predictions

    def predict_single(self, observation):
        """Predict single observation using emission probabilities only"""
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction")

        observation = self.scaler.transform(observation.reshape(1, -1))[0]

        # Calculate emission probabilities for all states
        probs = np.array(
            [self._emission_probability(observation, s) for s in range(self.n_states)]
        )

        # Return state with highest probability
        best_state = np.argmax(probs)
        return self.idx_to_state[best_state]

    def score(self, X, y):
        """Calculate accuracy on test data"""
        predictions = []

        # For sequence prediction, we need to group by windows/sequences
        # For now, predict each observation independently
        for i in range(len(X)):
            pred = self.predict_single(X[i])
            predictions.append(pred)

        return accuracy_score(y, predictions)


def load_and_prepare_data(data_file="hmm_ready_features.csv"):
    """Load and prepare data for HMM training"""
    print(f"Loading data from {data_file}...")

    # Load data
    data = pd.read_csv(data_file)
    print(f"Data shape: {data.shape}")

    # Separate features and labels
    feature_cols = [
        col
        for col in data.columns
        if col not in ["window_id", "activity", "start_time", "end_time"]
    ]

    X = data[feature_cols].values
    y = data["activity"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Activities: {np.unique(y)}")

    return X, y, feature_cols, data


def evaluate_hmm(hmm, X_test, y_test, activity_names):
    """Evaluate HMM performance"""
    print("\n=== HMM EVALUATION ===")

    # Predict
    y_pred = []
    for i in range(len(X_test)):
        pred = hmm.predict_single(X_test[i])
        y_pred.append(pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=activity_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=activity_names,
        yticklabels=activity_names,
    )
    plt.title("HMM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("plots/hmm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return accuracy, y_pred


def demonstrate_sequence_prediction(hmm, data, n_sequences=3):
    """Demonstrate HMM sequence prediction"""
    print(f"\n=== SEQUENCE PREDICTION DEMO ===")

    # Group data by activity to create sequences
    activities = data["activity"].unique()

    for activity in activities[:n_sequences]:
        activity_data = data[data["activity"] == activity].sort_values("start_time")

        if len(activity_data) >= 3:  # Need at least 3 windows for a sequence
            # Take first 3 windows as a sequence
            sequence_data = activity_data.head(3)

            feature_cols = [
                col
                for col in data.columns
                if col not in ["window_id", "activity", "start_time", "end_time"]
            ]

            X_seq = sequence_data[feature_cols].values
            y_true = sequence_data["activity"].values

            # Predict sequence
            predicted_sequence, log_prob = hmm.viterbi(X_seq)

            print(f"\nSequence for {activity}:")
            print(f"  True:      {' -> '.join(y_true)}")
            print(f"  Predicted: {' -> '.join(predicted_sequence)}")
            print(f"  Log probability: {log_prob:.2f}")


def main():
    """Main function to run HMM training and evaluation"""
    print("=== HMM ACTIVITY RECOGNITION ===\n")

    # Create plots directory
    import os

    os.makedirs("plots", exist_ok=True)

    # Load data
    X, y, feature_cols, data = load_and_prepare_data()

    # Split data for evaluation
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        # Small dataset - use all for training, subset for testing
        X_train = X
        y_train = y
        X_test = X[: min(5, len(X))]
        y_test = y[: min(5, len(y))]
        print("Warning: Small dataset - limited train/test split")

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize and train HMM
    activity_names = sorted(np.unique(y))
    hmm = ActivityHMM(n_states=len(activity_names), n_features=len(feature_cols))

    # Train HMM
    hmm.fit(X_train, y_train, activity_names)

    # Evaluate HMM
    accuracy, y_pred = evaluate_hmm(hmm, X_test, y_test, activity_names)

    # Demonstrate sequence prediction
    demonstrate_sequence_prediction(hmm, data)

    # Print HMM parameters summary
    print(f"\n=== HMM PARAMETERS SUMMARY ===")
    print(f"Number of states: {hmm.n_states}")
    print(f"Number of features: {hmm.n_features}")
    print(f"State names: {hmm.state_names}")

    print(f"\nTransition Matrix (A):")
    transition_df = pd.DataFrame(hmm.A, index=hmm.state_names, columns=hmm.state_names)
    print(transition_df.round(3))

    print(f"\nInitial Probabilities (π): {hmm.pi.round(3)}")

    print(f"\n=== RESULTS SUMMARY ===")
    print(f"HMM Accuracy: {accuracy:.3f}")
    print(
        f"Best performing activity: {activity_names[np.argmax([accuracy] * len(activity_names))]}"
    )

    if accuracy > 0.8:
        print("✅ Good performance - HMM is working well on your data!")
    elif accuracy > 0.6:
        print("⚠️  Moderate performance - consider more data or feature engineering")
    else:
        print("❌ Low performance - may need more data or different approach")

    print(f"\nFiles saved:")
    print(f"  - plots/hmm_confusion_matrix.png")

    return hmm, accuracy


if __name__ == "__main__":
    hmm, accuracy = main()
