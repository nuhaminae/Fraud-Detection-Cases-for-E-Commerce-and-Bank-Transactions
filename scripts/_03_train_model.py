# _03_train_model.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve
from xgboost import XGBClassifier


class TrainModel:
    """
    A class to perform model training on credit card fraud data.

    This class provides methods for loading, cleaning, and visualising data from
    two different sources: credit card info, and fraud data.
    It includes functionalities for handling missing values, removing duplicates,
    performing univariate and bivariate analysis, and handling outliers.
    """

    def __init__(
        self,
        credit_path,
        test_x_c_path,
        test_y_c_path,
        fraud_path,
        test_x_f_path,
        test_y_f_path,
        plot_dir=None,
        verbose=True,
    ):
        """
        Initialise TrainModel class from DataFrame paths.

        Args:
            credit_path (str): The path to the train credit card DataFrame.
            test_x_c_path (str): The path to test x credit data DataFrame.
            test_y_c_path (str): The path to test y credit data DataFrame.
            fraud_path (str): The path to the train fraud data DataFrame.
            test_x_f_path (str): The path to test x fraud data DataFrame.
            test_y_f_path (str): The path to test y fraud data DataFrame.
            plot_dir (str, optional): The directory to save plots. Defaults to None.
            verbose (bool, optional): Whether to display detailed info during loading.
                                        Defaults to True.
        """
        self.paths = {
            "Credit": (credit_path, test_x_c_path, test_y_c_path),
            "Fraud": (fraud_path, test_x_f_path, test_y_f_path),
        }
        self.plot_dir = plot_dir
        self.verbose = verbose

        # Create output directory if it does not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.datasets = {}
        self.models = {}
        self.test_sets = {}
        print("TrainModel class initialised\n")
        self.load_and_process()
        self.model_selection()

    @staticmethod
    def safe_relpath(path, start=os.getcwd()):
        """
        Return a relative path, handling cases where paths are on different drives.

        Args:
            path (str): The path to make relative.
            start (str, optional): The starting directory.
                                    Defaults to current working directory.

        Returns:
            str: The relative path if possible, otherwise the original path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            return path  # Fallback to absolute path if on different drives

    def display_info(self, df, label):
        """
        Display head, shape, columns, and info for a given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to display information for.
            label (str): A label for the DataFrame.
        """
        if not self.verbose:  # Fallback if display is not present
            return
        print(f"{label} Head:")
        display(df.head())
        print(f"\n{label} Shape: {df.shape}")
        print(f"\n{label} Columns: {list(df.columns)}")
        print(f"\n{label} Info:")
        df.info()

    def load_and_process_single(self, label, train_path, test_x_path, test_y_path):
        """
        Load and process a single dataset (train and test).

        Args:
            label (str): The label for the dataset ('Credit' or 'Fraud').
            train_path (str): Path to the training data CSV.
            test_x_path (str): Path to the test features CSV.
            test_y_path (str): Path to the test target CSV.

        Returns:
            tuple: A tuple containing X_train, y_train, X_test, y_test DataFrames
                    and/or Series.
        """
        print(f"\n--- {label} Dataset ---")
        print(f"Training data loaded from {self.safe_relpath(train_path)}")
        print(f"Training data loaded from {self.safe_relpath(test_x_path)}")
        print(f"Training data loaded from {self.safe_relpath(test_y_path)}\n")
        df_train = pd.read_csv(train_path)
        self.display_info(df_train, label)
        X_train = df_train.drop(columns=["Class"])
        y_train = df_train["Class"]
        X_test = pd.read_csv(test_x_path)
        y_test = pd.read_csv(test_y_path).squeeze()
        return X_train, y_train, X_test, y_test

    def load_and_process(self):
        """
        Load and process all datasets specified in self.paths.
        """
        for label, paths in self.paths.items():
            self.datasets[label] = self.load_and_process_single(label, *paths)

    # ------------Model Selection------------ #
    def model_selection(self):
        """
        Define the models to be used for training.
        """
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XG Boost": XGBClassifier(n_estimators=100, random_state=42),
        }

    # ------------Model Training and Evaluation------------ #
    def plot_confusion_matrix(self, cm, label, model_name):
        """
        Plot and save the confusion matrix.

        Args:
            cm (np.ndarray): The confusion matrix.
            label (str): The label for the dataset ('Credit' or 'Fraud').
            model_name (str): The name of the model.
        """
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{label} Confusion Matrix {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        if self.plot_dir:
            sanitized_name = model_name.replace(" ", "_")
            file_path = os.path.join(self.plot_dir, f"{label}_cm_{sanitized_name}.png")
            plt.savefig(file_path)
            print(f"Confusion matrix saved to {self.safe_relpath(file_path)}")
        plt.show()
        plt.close()

    def plot_auc_pr_curve(self, precision, recall, label, model_name):
        """
        Plot and save the Precision-Recall curve.

        Args:
            precision (np.ndarray): Precision values.
            recall (np.ndarray): Recall values.
            label (str): The label for the dataset ('Credit' or 'Fraud').
            model_name (str): The name of the model.
        """
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker=".", label=f"AUC-PR {model_name}")
        plt.title(f"{label} Precision-Recall Curve {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.plot_dir:
            sanitized_name = model_name.replace(" ", "_")
            file_path = os.path.join(self.plot_dir, f"{label}_pr_{sanitized_name}.png")
            plt.savefig(file_path)
            print(f"Precision-Recall curve saved to {self.safe_relpath(file_path)}")
        plt.show()
        plt.close()

    def evaluate_model(self, model, X_test, y_test, label, model_name):
        """
        Evaluate a trained model and print performance metrics.

        Args:
            model: The trained model.
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test target.
            label (str): The label for the dataset ('Credit' or 'Fraud').
            model_name (str): The name of the model.

        Returns:
            tuple: A tuple containing F1 score, AUC-PR, and confusion matrix.
        """
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(
            y_test, model.predict_proba(X_test)[:, 1]
        )
        auc_pr = auc(recall, precision)
        cm = confusion_matrix(y_test, y_pred)

        print(f"{label} F1-Score {model_name}: {f1:.4f}")
        print(f"{label} AUC-PR {model_name}: {auc_pr:.4f}")
        print(f"{label} Confusion Matrix {model_name}:\n{cm}\n")

        # Plot results
        self.plot_confusion_matrix(cm, label, model_name)
        self.plot_auc_pr_curve(precision, recall, label, model_name)

        return f1, auc_pr, cm

    def train_and_evaluate(self):
        """
        Train and evaluate all models on all datasets.
        """
        for label, (X_train, y_train, X_test, y_test) in self.datasets.items():
            print(f"\n--- {label} Dataset ---")
            self.test_sets[label] = (X_test, y_test)
            for name, model in self.models.items():
                print(f"\nTraining {name} on {label}...")
                model.fit(X_train, y_train)
                self.evaluate_model(model, X_test, y_test, label, name)
