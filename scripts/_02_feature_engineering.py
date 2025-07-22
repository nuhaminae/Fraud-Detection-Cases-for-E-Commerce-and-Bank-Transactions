# _01_data_preprocessing.py
import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    """
    A class to perform feature engineering and data transformation on fraud data.

    This class handles loading data, creating new features based on transaction
    patterns and time, and preparing the data for model training through
    transformation steps like one-hot encoding, scaling, and handling class imbalance.
    """

    def __init__(
        self,
        fraud_path,
        processed_dir,
    ):
        """
        Initiate FeatureEngineering class from DataFrame path.

        Args:
            fraud_path (str): The path to the raw fraud data DataFrame file (CSV).
            processed_dir (str, optional): The directory to save processed DataFrames.
        """
        self.fraud_path = fraud_path
        self.processed_dir = processed_dir
        self.df = None

        # Create output directory if it does not exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("FeatureEngineering class is Initialised\n")
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

    def load_df(self):
        """
        Load a single DataFrame from the specified fraud_path.

        This method reads the CSV file, corrects data types for specific columns,
        and displays the head, shape, and info of the loaded DataFrame.

        Returns:
            pd.DataFrame: The loaded and cleaned DataFrame, or None if loading fails.
        """
        rel_path = self.safe_relpath(self.fraud_path)
        try:
            df = pd.read_csv(self.fraud_path)
            print(f"DataFrame loaded from {rel_path}")

            # Correct data types
            dict_col = {
                "User_Id": "object",
                "Signup_Time": "datetime",
                "Purchase_Time": "datetime",
                "Purchase_Value": "float",
                "Device_Id": "object",
                "Source": "object",
                "Browser": "object",
                "Sex": "object",
                "Age": "int",
                "Ip_Address": "int",
                "Class": "int",
            }

            for col, dtype in dict_col.items():
                if col in df.columns:
                    if dtype == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif col in [
                        "Ip_Address",
                        "Lower_Bound_Ip_Address",
                        "Upper_Bound_Ip_Address",
                    ]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        # Round to get clean integers before casting
                        df[col] = df[col].round().astype("int64")
                    else:
                        df[col] = df[col].astype(dtype)

            print("DataFrame Head:")
            display(df.head())
            print(f"\nDataFrame Shape: {df.shape}")
            print(f"\nDataFrame Columns: {list(df.columns)}")
            print("DataFrame Info:")
            df.info()

            self.df = df  # Assign loaded and cleaned DataFrame

            return self.df

        except Exception as e:
            print(f"Failed to load from {rel_path}: {e}")
            return None

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
        # Transaction count per device
        freq_by_user = (
            self.df.groupby("User_Id")["Purchase_Time"]
            .count()
            .rename("User_Transaction_Count")
        )
        freq_by_device = (
            self.df.groupby("Device_Id")["Purchase_Time"]
            .count()
            .rename("Device_Transaction_Count")
        )
        self.df = self.df.join(freq_by_user, on="User_Id").join(
            freq_by_device, on="Device_Id"
        )

        # Time since last transaction per user
        self.df.sort_values(["User_Id", "Purchase_Time"], inplace=True)
        self.df["Time_Since_Last"] = (
            self.df.groupby("User_Id")["Purchase_Time"].diff().dt.total_seconds()
        )
        self.df["Time_Since_Last"] = self.df["Time_Since_Last"].fillna(-1)

        self.df["Velocity"] = self.df["Purchase_Value"] / self.df[
            "Time_Since_Last"
        ].replace(0, pd.NA)

        # Time-based features
        # Hour of day
        self.df["Hour_Of_Day"] = self.df["Purchase_Time"].dt.hour

        # Day of week
        self.df["Day_Of_Week"] = self.df["Purchase_Time"].dt.dayofweek

        # Time since signup
        self.df["Time_Since_Signup"] = (
            self.df["Purchase_Time"] - self.df["Signup_Time"]
        ).dt.total_seconds()

        self.df.head()

        output_path = os.path.join(
            self.processed_dir, "FraudData_FeatureEngineered.csv"
        )
        self.df.to_csv(output_path, index=False)
        print(f"Enriched fraud data saved to {self.safe_relpath(output_path)}")

        return self.df

    def data_transformation(self):
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
        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        # One-hot encode categoricals
        categorical = ["Source", "Browser", "Sex"]
        X = pd.get_dummies(X, columns=categorical, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Drop non-numeric columns (like IDs and datetime leftovers)
        # Drop datetime columns before SMOTE
        non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns
        datetime_cols = X_train.select_dtypes(include=["datetime64[ns]"]).columns
        X_train = X_train.drop(columns=non_numeric_cols.union(datetime_cols))
        X_test = X_test.drop(columns=non_numeric_cols.union(datetime_cols))

        # Class imbalance handling
        # fraud_rate = self.df["Class"].mean()
        # print(f"Original Fraud Ratio: {fraud_rate:.2%}")
        print(f"Original Fraud Ratio: {y_train.mean():.2%}")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print(
            f"Shape before SMOTE: {X_train.shape}, \
            Shape after SMOTE: {X_train_resampled.shape}"
        )
        print(f"Resampled Fraud Ratio: {y_train_resampled.mean():.2%}")

        # Normalisation and scaling
        numeric_cols = X_train.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        # Scaling
        numeric_cols = X_train_resampled.select_dtypes(
            include=["float64", "int64"]
        ).columns
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled[numeric_cols])
        X_test_scaled = scaler.transform(X_test[numeric_cols])

        # Save
        X_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
        y_resampled_df = pd.DataFrame(y_train_resampled, columns=["Class"])

        merged_df = pd.concat([X_scaled_df, y_resampled_df], axis=1)

        save_path = os.path.join(self.processed_dir, "Train_Resampled_Scaled.csv")
        merged_df.to_csv(save_path, index=False)
        print(
            f"Resampled and scaled training data \
            'Train_Resampled_Scaled'saved to {self.safe_relpath(save_path)}"
        )

        pd.DataFrame(X_test_scaled, columns=numeric_cols).to_csv(
            "X_test_scaled.csv", index=False
        )
        pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

    def run_all(self):
        """
        Execute the full feature engineering and data transformation pipeline.

        This method calls `transaction_frequency_Velocity` to generate features
        and then `data_transformation` to prepare the data for model training.
        """
        self.transaction_frequency_Velocity()
        self.data_transformation()
