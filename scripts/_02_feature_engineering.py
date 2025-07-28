# _02_feature_engineering.py
import calendar
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    """
    A class to perform feature engineering and data transformation on fraud data.

    This class handles loading data, creating new features based on transaction
    patterns and time, and preparing the data for model training through
    transformation steps like one-hot encoding, scaling, and handling class imbalance.
    """

    def __init__(self, credit_path, fraud_path, processed_dir, plot_dir=None):
        """
        Initiate FeatureEngineering class from DataFrame path.

        Args:
            credit_path (str): The path to the credit card DataFrame file (CSV).
            fraud_path (str): The path to the raw fraud data DataFrame file (CSV).
            processed_dir (str, optional): The directory to save processed DataFrames.
            plot_dir (str, optional): The directory to save plots.
                                    Defaults to None.
        """
        self.fraud_path = fraud_path
        self.credit_path = credit_path
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir
        self.df = None

        # Create output directory if it does not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("FeatureEngineering class is initialised\n")
        self.load_df()

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

    def load_single_df(self, path, label):
        """
        Load a single DataFrame from the specified paths.

        This method reads the CSV file, corrects data types for specific columns,
        and displays the head, shape, and info of the loaded DataFrame.

        Returns:
            pd.DataFrame: The loaded and cleaned DataFrame, or None if loading fails.
        """
        rel_path = self.safe_relpath(path)
        try:
            df = pd.read_csv(path)
            print(f"DataFrame loaded from {rel_path}")

            # Correct data types
            dict_col = {
                "User_Id": "object",
                "Signup_Time": "datetime",
                "Purchase_Time": "datetime",
            }

            for col, dtype in dict_col.items():
                if col in df.columns:
                    if dtype == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif col in ["Ip_Address"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        # Round to get clean integers before casting
                        df[col] = df[col].round().astype("int64")
                    else:
                        df[col] = df[col].astype(dtype)

            print(f"{label} Head:")
            display(df.head())
            print(f"\n{label}Shape: {df.shape}")
            print(f"\n{label}Columns: {list(df.columns)}")
            print(f"\n{label}Info:")
            df.info()
            print("\n" + "*==*" * 20 + "\n")

            return df

        except Exception as e:
            print(f"Failed to load from {rel_path}: {e}")
            return None

    def load_df(self):
        self.credit = self.load_single_df(self.credit_path, "CreditCard")
        self.fraud = self.load_single_df(self.fraud_path, "FraudData")

    def transaction_frequency_Velocity(self):
        """
        Calculate transaction frequency, time since last transaction, and velocity.

        This method adds new features to the DataFrame based on transaction patterns
        for each user and device. It calculates the transaction count per user and
        device, the time elapsed since the last transaction for each user,
        and a velocity feature based on purchase value and
        time since the last transaction. It also extracts time-based features
        like hour of the day and day of the week, and calculates the time since signup.
        The enriched DataFrame is then saved to a CSV file.

        Returns:
            pd.DataFrame: The DataFrame with added frequency, velocity,
                and time-based features.
        """
        rel_plot_path = self.safe_relpath(self.plot_dir)

        # Transaction count per device
        freq_by_user = (
            self.fraud.groupby("User_Id")["Purchase_Time"]
            .count()
            .rename("User_Transaction_Count")
        )
        freq_by_device = (
            self.fraud.groupby("Device_Id")["Purchase_Time"]
            .count()
            .rename("Device_Transaction_Count")
        )
        self.fraud = self.fraud.join(freq_by_user, on="User_Id").join(
            freq_by_device, on="Device_Id"
        )

        # Calc Velocity and Time Since Last by Device_Id col b/c user_Id is unique
        # Sort once and reuse
        self.df_sorted = self.fraud.sort_values(by=["Device_Id", "Purchase_Time"])

        # Calculate time since last transaction
        self.df_sorted["Time_Since_Last"] = (
            self.df_sorted.groupby("Device_Id")["Purchase_Time"]
            .diff()
            .dt.total_seconds()  # converts timedelta to float seconds
        )

        # Calculate velocity only for non-null time intervals
        self.df_sorted["Velocity"] = np.nan  # Initialise column
        valid_mask = self.df_sorted["Time_Since_Last"].notna()
        self.df_sorted.loc[valid_mask, "Velocity"] = (
            self.df_sorted.loc[valid_mask, "Purchase_Value"]
            / self.df_sorted.loc[valid_mask, "Time_Since_Last"]
        )

        # Assign enriched data back
        self.fraud = self.df_sorted.copy()

        # Add Missing_Geo binary column
        self.fraud["Missing_Geo"] = self.fraud["Country"].apply(
            lambda x: 1 if x == "Unknown" else 0
        )

        # Visualise Transaction Velocity Distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(
            data=(self.fraud[self.fraud["Velocity"] > 0.003]),
            x="Velocity",
            hue="Class",
            bins=50,
            kde=True,
            palette="Set1",
        )
        plt.title("Transaction Velocity by Class")
        plt.ylabel("Mean Fraud Rate")
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "velocity_hist.png")
            plt.savefig(plot_path)
            print(f"\nPlot saved to {rel_plot_path}")

        # show and close plot
        plt.show()
        plt.close()

        # Time-based features
        # Hour of day
        self.fraud["Hour_Of_Day"] = self.fraud["Purchase_Time"].dt.hour
        # Visualise Fraud Rate by Hour of Day
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=self.fraud.groupby("Hour_Of_Day")["Class"].mean().reset_index(),
            x="Hour_Of_Day",
            y="Class",
            hue="Hour_Of_Day",
            palette="Set1",
        )
        plt.title("Fraud Rate by Hour of Day")
        plt.ylabel("Mean Fraud Rate")
        plt.grid()
        plt.tight_layout()
        plt.legend().set_visible(False)

        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "hour_day_bar.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {rel_plot_path}")

        # show and close plot
        plt.show()
        plt.close()

        # Day of week
        # Map numerical day of week to name
        self.fraud["Day_Of_Week"] = self.fraud["Purchase_Time"].dt.dayofweek.map(
            lambda x: calendar.day_name[x]
        )
        # Define weekday order
        weekday_order = list(calendar.day_name)  # ['Monday', 'Tuesday', ..., 'Sunday']
        # Prepare grouped data
        grouped_df = self.fraud.groupby("Day_Of_Week")["Class"].mean().reset_index()
        # Sort using predefined order
        grouped_df["Day_Of_Week"] = pd.Categorical(
            grouped_df["Day_Of_Week"], categories=weekday_order, ordered=True
        )
        grouped_df = grouped_df.sort_values("Day_Of_Week")

        # VisualiseFraud Rate by Day of Week
        sns.barplot(
            data=grouped_df,
            x="Day_Of_Week",
            y="Class",
            hue="Day_Of_Week",
            palette="Set1",
        )
        plt.title("Fraud Rate by Day of Week")
        plt.ylabel("Mean Fraud Rate")
        plt.grid()
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "day_week_bar.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {rel_plot_path}")

        # show and close plot
        plt.show()
        plt.close()

        # Time since signup
        self.fraud["Time_Since_Signup"] = (
            self.fraud["Purchase_Time"] - self.fraud["Signup_Time"]
        ).dt.total_seconds()

        self.fraud.head()

        output_path = os.path.join(
            self.processed_dir, "FraudData_FeatureEngineered.csv"
        )
        self.fraud.to_csv(output_path, index=False)
        print(f"Enriched fraud data saved to {self.safe_relpath(output_path)}")

        return self.fraud

    def fraud_data_transformation(self):
        """
        Perform data transformation for model training.

        This method prepares the data for machine learning
            by performing the following steps:
        1. Splits the data into features (X) and target (y).
        2. Applies one-hot encoding to specified categorical columns.
        3. Splits the data into training and testing sets using stratified sampling.
        4. Removes non-numeric and datetime columns from the feature sets.
        5. Handles class imbalance in the training data using SMOTE.
        6. Scales the numeric features using StandardScaler.
        7. Saves the resampled and scaled training data,
            and the scaled testing data and their labels to CSV files.

        Returns:
            None. The processed dataframes are saved to files.
        """
        # Split features and target
        X = self.fraud.drop("Class", axis=1)
        y = self.fraud["Class"]

        # One-hot encode categoricals
        categorical = ["Source", "Browser", "Sex"]
        X = pd.get_dummies(X, columns=categorical, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Drop numeric and datetime columns
        non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns
        datetime_cols = X_train.select_dtypes(include=["datetime64[ns]"]).columns
        X_train = X_train.drop(columns=non_numeric_cols.union(datetime_cols))
        X_test = X_test.drop(columns=non_numeric_cols.union(datetime_cols))

        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train), columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Class imbalance handling
        print(f"Original Fraud Ratio: {y_train.mean():.2%}")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_imputed, y_train
        )

        print(f"\nShape before SMOTE: {X_train.shape}")
        print(f"Shape after SMOTE: {X_train_resampled.shape}")
        print(f"\nResampled Fraud Ratio: {y_train_resampled.mean():.2%}")

        # Normalisation and scaling
        numeric_cols = X_train_resampled.select_dtypes(
            include=["float64", "int64"]
        ).columns
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled[numeric_cols])
        X_test_scaled = scaler.transform(X_test_imputed[numeric_cols])

        # Save
        X_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
        y_resampled_df = pd.DataFrame(y_train_resampled, columns=["Class"])
        fraud_merged_df = pd.concat([X_scaled_df, y_resampled_df], axis=1)

        save_path = os.path.join(self.processed_dir, "Train_Resampled_Scaled_Fraud.csv")
        fraud_merged_df.to_csv(save_path, index=False)
        print(
            f"Resampled and scaled training data 'Train_Resampled_Scaled_Fraud' saved \
                to {self.safe_relpath(save_path)}.\n"
        )

        # Class Distribution Bar Plot After Class Balancing
        if "Class" in fraud_merged_df.columns:
            print("Class Distribution Bar Plot After Class Balancing...\n")
            plt.figure(figsize=(10, 5))
            sns.countplot(data=fraud_merged_df, x="Class", hue="Class", palette="Set1")
            plt.title("Class Distribution after SMOTE (Fraud)")
            plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
            plt.ylabel("Transaction Count")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, "fraud_new_class_count.png")
                plt.savefig(plot_path)
                print(f"Plot saved to {self.safe_relpath(plot_path)}")

            plt.show()
            plt.close()

        # Save scaled test set for model evaluation
        # Centralised paths for test set
        test_x_path = os.path.join(self.processed_dir, "X_test_scaled_fraud.csv")
        test_y_path = os.path.join(self.processed_dir, "y_test_fraud.csv")

        pd.DataFrame(X_test_scaled, columns=numeric_cols).to_csv(
            test_x_path, index=False
        )
        pd.DataFrame(y_test).to_csv(test_y_path, index=False)

        # Save raw (unscaled) for SHAP interpretation
        pd.DataFrame(X_train_resampled, columns=X_train_imputed.columns).to_csv(
            "X_train_resampled_raw_fraud.csv", index=False
        )

        return fraud_merged_df

    def credit_data_transformation(self):
        """
        Perform data transformation for model training.

        This method prepares the data for machine learning
            by performing the following steps:
        1. Splits the data into features (X) and target (y).
        2. Applies one-hot encoding to specified categorical columns.
        3. Splits the data into training and testing sets using stratified sampling.
        4. Handles class imbalance in the training data using SMOTE.
        5. Scales the numeric features using StandardScaler.
        6. Saves the resampled and scaled training data,
            and the scaled testing data and their labels to CSV files.

        Returns:
            None. The processed dataframes are saved to files.
        """
        # Split features and target
        X = self.credit.drop("Class", axis=1)
        y = self.credit["Class"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train), columns=X_train.columns
        )
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Class imbalance handling
        print(f"Original Fraud Ratio: {y_train.mean():.2%}")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_imputed, y_train
        )

        print(f"\nShape before SMOTE: {X_train.shape}")
        print(f"Shape after SMOTE: {X_train_resampled.shape}")
        print(f"\nResampled Fraud Ratio: {y_train_resampled.mean():.2%}")

        # Normalisation and scaling
        numeric_cols = X_train_resampled.select_dtypes(
            include=["float64", "int64"]
        ).columns
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled[numeric_cols])
        X_test_scaled = scaler.transform(X_test_imputed[numeric_cols])

        # Save
        X_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
        y_resampled_df = pd.DataFrame(y_train_resampled, columns=["Class"])
        credit_merged_df = pd.concat([X_scaled_df, y_resampled_df], axis=1)

        save_path = os.path.join(
            self.processed_dir, "Train_Resampled_Scaled_Credit.csv"
        )
        credit_merged_df.to_csv(save_path, index=False)
        print(
            f"Resampled and scaled training data 'Train_Resampled_Scaled_Credit'saved \
                to {self.safe_relpath(save_path)}.\n"
        )

        # Class Distribution Bar Plot After Class Balancing
        if "Class" in credit_merged_df.columns:
            print("Class Distribution Bar Plot After Class Balancing...\n")
            plt.figure(figsize=(10, 5))
            sns.countplot(data=credit_merged_df, x="Class", hue="Class", palette="Set1")
            plt.title("Class Distribution after SMOTE (Credit)")
            plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
            plt.ylabel("Transaction Count")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, "credit_new_class_count.png")
                plt.savefig(plot_path)
                print(f"Plot saved to {self.safe_relpath(plot_path)}")

            plt.show()
            plt.close()

        # Save scaled test set for model evaluation
        # Centralised paths for test set
        test_x_path = os.path.join(self.processed_dir, "X_test_scaled_credit.csv")
        test_y_path = os.path.join(self.processed_dir, "y_test_credit.csv")

        pd.DataFrame(X_test_scaled, columns=numeric_cols).to_csv(
            test_x_path, index=False
        )
        pd.DataFrame(y_test).to_csv(test_y_path, index=False)

        # Save raw (unscaled) for SHAP interpretation
        pd.DataFrame(X_train_resampled, columns=X_train_imputed.columns).to_csv(
            "X_train_resampled_raw_credit.csv", index=False
        )

        return credit_merged_df

    def run_all(self):
        """
        Execute the full feature engineering and data transformation pipeline.

        This method calls `transaction_frequency_Velocity` to generate features
        and then `fraud_data_transformation` and `credit_data_transformation`
        to prepare the data for model training.
        """
        self.transaction_frequency_Velocity()
        self.fraud_data_transformation()
        self.credit_data_transformation()
