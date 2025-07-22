# _01_data_preprocessing.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display


class EDA:
    def __init__(
        self,
        credit_path,
        fraud_path,
        ip_path,
        plot_dir=None,
        processed_dir=None,
        verbose=True,
    ):
        """
        Initiate EDA class from DataFrame path.

        Args:
            credit_path (str): The path to the credit card DataFrame file in CSV format.
            fraud_path (str): The path to the fraud data DataFrame file in CSV format.
            ip_path (str): The path to the IP address DataFrame file in CSV format.
            plot_dir (str, optional): The directory to save plots.
            processed_dir (str, optional): The directory to save processed DataFrames.
        """
        self.credit_path = credit_path
        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.plot_dir = plot_dir
        self.processed_dir = processed_dir
        self.credit_raw = None
        self.fraud_raw = None
        self.ip_raw = None
        self.verbose = verbose

        # Create output directory if it does not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("EDA class is Initialised\n")
        self.load_df()

    @staticmethod
    def safe_relpath(path, start=os.getcwd()):
        try:
            return os.path.relpath(path, start)
        except ValueError:
            return path  # Fallback to absolute path if on different drives

    def load_single_df(self, path, label):
        """
        Load single DataFrame
        """
        rel_path = self.safe_relpath(path)
        try:
            df = pd.read_csv(path)
            # Capitalise column names for consistency
            df.columns = [col.title() for col in df.columns]
            print(f"{label} loaded from {rel_path}")
            self.display_info(df, label)
            return df
        except Exception as e:
            print(f"Failed to load {label} from {rel_path}: {e}")
            return None

    def load_df(self):
        """
        Load multiple DataFrames
        """
        self.credit_raw = self.load_single_df(self.credit_path, "CreditCard")
        self.fraud_raw = self.load_single_df(self.fraud_path, "FraudData")
        self.ip_raw = self.load_single_df(self.ip_path, "IPAddressMap")

    def display_info(self, df, label):
        if not self.verbose:  # Fallback if display is not present
            return
        print(f"{label} Head:")
        display(df.head())
        print(f"\n{label} Shape: {df.shape}")
        print(f"\n{label} Columns: {list(df.columns)}")
        print(f"\n{label} Info:")
        df.info()
        print("\n" + "*==*" * 30 + "\n")

    # ------------Data Cleaning------------#
    @staticmethod
    def data_cleaning_single(df, label):
        """
        Remove duplicates and correct data types.
        """
        if df is None:
            print(f"DataFrame {label} not loaded. Please check initialisation.")
            return None

        # Remove duplicate columns while preserving order
        before_cols = df.columns.tolist()
        df = df.loc[:, ~df.columns.duplicated()]
        after_cols = df.columns.tolist()

        if before_cols != after_cols:
            print(f"{label}: Removed {set(before_cols) - set(after_cols)} duplicates.")

        # Remove duplicated rows
        initial_shape = df.shape
        df.drop_duplicates(inplace=True)
        if df.shape != initial_shape:
            print(f"{label}: Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

        # Correct data types
        dict_col = {
            "User_Id": "object",
            "Device_Id": "object",
            "Source": "object",
            "Browser": "object",
            "Sex": "object",
            "Country": "object",
            "Signup_Time": "datetime",
            "Purchase_Time": "datetime",
            "Class": "int",
            "Age": "int",
            "Ip_Address": "int",
            "Lower_Bound_Ip_Address": "int",
            "Upper_Bound_Ip_Address": "int",
            "purchase_value": "float",
            "Time": "float",
            "V1": "float",
            "V2": "float",
            "V3": "float",
            "V4": "float",
            "V5": "float",
            "V6": "float",
            "V7": "float",
            "V8": "float",
            "V9": "float",
            "V10": "float",
            "V11": "float",
            "V12": "float",
            "V13": "float",
            "V14": "float",
            "V15": "float",
            "V16": "float",
            "V17": "float",
            "V18": "float",
            "V19": "float",
            "V20": "float",
            "V21": "float",
            "V22": "float",
            "V23": "float",
            "V24": "float",
            "V25": "float",
            "V26": "float",
            "V27": "float",
            "V28": "float",
            "Amount": "float",
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

        print(f"{label}: Datatypes changed.")
        print(f"\nNew {label} Info:")
        df.info()
        print("\n")
        return df

    def data_cleaning(self):
        self.credit = self.data_cleaning_single(self.credit_raw, "CreditCard")
        self.fraud = self.data_cleaning_single(self.fraud_raw, "FraudData")
        self.ip = self.data_cleaning_single(self.ip_raw, "IPAddress Map")

    # ------------Handle Missing Values------------#

    @staticmethod
    def missing_values_single(df, label):
        """
        Evaluate missing values column-wise and decide whether to impute or drop.
        """
        if df is None:
            print(f"DataFrame {label} not loaded. Please check initialisation.")
            return None

        # Drop all-zero IP columns
        ip_cols = ["Ip_Address", "Lower_Bound_Ip_Address", "Upper_Bound_Ip_Address"]
        for ip_col in ip_cols:
            if ip_col in df.columns and (df[ip_col] == 0).mean() == 1:
                df.drop(columns=[ip_col], inplace=True)
                print(f"{label}: Dropped {ip_col} with all-zero values.")

        # Report missing values
        missing_per_col = df.isna().mean().sort_values(ascending=False)
        if missing_per_col.sum() == 0:
            print(f"{label}: No missing values found.")
        else:
            # Drop columns with more than 30% missing
            high_mis_cols = missing_per_col[missing_per_col > 0.3].index.tolist()
            if high_mis_cols:
                df.drop(columns=high_mis_cols, inplace=True)
                print(
                    f"{label}: Dropped columns with >30% missing: {list(high_mis_cols)}"
                )

            # Impute numeric column
            numeric_miss = df.select_dtypes(include="number").columns[
                df.select_dtypes(include="number").isna().any()
            ]
            df.fillna(df.median(numeric_only=True), inplace=True)
            print(
                f"{label}: Missing values imputed using medians:{list(numeric_miss)}."
            )

            # Drop object columns with missing values
            obj_missing = df.select_dtypes(include="object").columns[
                df.select_dtypes(include="object").isna().any()
            ]
            if obj_missing:
                df.drop(columns=obj_missing, inplace=True)
                print(
                    f"{label}: Dropped object-type missing values: {list(obj_missing)}"
                )

        return df

    def missing_values(self):
        """
        Handle missing values for multiple DataFrames
        """
        self.credit = self.missing_values_single(self.credit, "CreditCard")
        self.fraud = self.missing_values_single(self.fraud, "FraudData")
        self.ip = self.missing_values_single(self.ip, "IPAddress Map")

    # ------------Univariate EDA------------#
    def univariate_single(self, df, label):
        """
        Apply univariate EDA.
        """
        if df is None:
            print(f"DataFrame {label} not loaded. Please check initialisation.")
            return None

        rel_plot_path = self.safe_relpath(self.plot_dir)

        # Univariate analysis
        # Visualise Distribution
        numeric_cols = df.select_dtypes(include="number").columns
        exclude_cols = {
            "Ip_Address",
            "Lower_Bound_Ip_Address",
            "Upper_Bound_Ip_Address",
        }
        plot_cols = [col for col in numeric_cols if col not in exclude_cols]
        if plot_cols:
            print(f"\n{label}: Visualising numerical values ...\n")

        for col in plot_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], bins=50, kde=True, color="green")
            plt.title(f"{label} - Distribution of {col}")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, f"{label}_{col}_hist.png")
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"Plot saved to {rel_plot_path}")

            # show and close plot
            plt.show()
            plt.close()

        if plot_cols:
            skewness = df[plot_cols].skew()
            kurtosis = df[plot_cols].kurt()

            # Combine into a DataFrame
            summary_stats = pd.DataFrame({"Skewness": skewness, "Kurtosis": kurtosis})
            summary_stats = summary_stats.reset_index().rename(
                columns={"index": "Column"}
            )

            print(summary_stats)

        # Categorical Frequency
        cat_cols = df.select_dtypes(include="object").columns
        exclude_cols = {"User_Id", "Device_Id", "Country"}
        plot_cols = [col for col in cat_cols if col not in exclude_cols]

        if plot_cols:
            print(f"\n{label}: Visualising categorical values ...\n")

        for col in plot_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=col)
            plt.title(f"{label} - Frequency of {col}")
            plt.xticks(rotation=0)
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, f"{label}_{col}_count.png")
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"Plot saved to {rel_plot_path}")

            # show and close plot
            plt.show()
            plt.close()

        if "Country" in df.columns:
            # Get top 20 countries
            top_countries = df["Country"].value_counts().nlargest(20).index
            filtered_df = df[df["Country"].isin(top_countries)]

            plt.figure(figsize=(10, 5))
            sns.countplot(
                data=filtered_df, x="Country", order=top_countries, hue="Country"
            )
            plt.title(f"{label} - Frequency of Top 20 Countries")
            plt.xticks(rotation=45, ha="right")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(
                    self.plot_dir, f"{label}_Top20Country_count.png"
                )
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"Plot saved to {rel_plot_path}")

            # show and close plot
            plt.show()
            plt.close()

        print("\n" + "*==*" * 30 + "\n")

    def univariate(self):
        self.univariate_single(self.credit, "CreditCard")
        self.univariate_single(self.fraud, "FraudData")
        self.univariate_single(self.ip, "IPAddress Map")

    # ------------Bivariate EDA------------#

    def bivariate_single(self, df, label):
        """
        Apply Bivariate EDA.
        """
        if df is None:
            print(f"DataFrame {label} not loaded. Please check initialisation.")
            return None

        rel_plot_path = self.safe_relpath(self.plot_dir)

        # Bivariate analysis
        # Boxplots for Numeric vs Target
        target_col = "Class"
        if target_col not in df.columns:
            print(f"\n'{target_col}' column not found in {label}. Skipping.\n")
            return df

        numeric_cols = df.select_dtypes(include="number").columns.drop(
            target_col, errors="ignore"
        )
        exclude_cols = {
            "Ip_Address",
            "Lower_Bound_Ip_Address",
            "Upper_Bound_Ip_Address",
        }
        plot_cols = [col for col in numeric_cols if col not in exclude_cols]
        if plot_cols:
            print(f"\n{label}: Visualising outliers ...\n")

        for col in plot_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=target_col, y=col)
            plt.title(f"{label} - {col} by {target_col}")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, f"{label}_{col}_box.png")
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"Plot saved to {rel_plot_path}")

            # show and close plot
            plt.show()
            plt.close()

        # Countplots for Categorical vs Target
        target_col = "Class"
        if target_col not in df.columns:
            print(f"\n'{target_col}' column not found in {label}. Skipping.\n")
            return df

        cat_cols = df.select_dtypes(include="object").columns
        exclude_cols = {"User_Id", "Device_Id", "Country"}
        plot_cols = [col for col in cat_cols if col not in exclude_cols]

        if plot_cols:
            print(f"\n{label}: Visualising categorical values vs Target ...\n")

        for col in plot_cols:
            plt.figure(figsize=(10, 5))

            top_levels = df[col].value_counts().nlargest(10).index
            filtered_df = df[df[col].isin(top_levels)]

            sns.countplot(x=col, hue=target_col, data=filtered_df, order=top_levels)
            # sns.countplot(x=col, hue=target_col, data=df)
            plt.title(f"{label} - {col} vs {target_col}")
            plt.xticks(rotation=0)
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(
                    self.plot_dir, f"{label}_{col}_target_count.png"
                )
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"Plot saved to {rel_plot_path}")

            # show and close plot
            plt.show()
            plt.close()

        # Correlation Heatmap (CreditCard)
        corr = df.select_dtypes(include="number").corr()
        # Plot important relations only
        # important_corr = corr[(corr.abs() > 0.1) & (corr != 1.0)]
        plt.figure(figsize=(20, 10))
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.title(f"{label} Dataset - Feature Correlation")
        plt.tight_layout()

        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, f"{label}_corr.png")
            rel_plot_path = self.safe_relpath(plot_path)
            plt.savefig(plot_path)
            print(f"Plot saved to {rel_plot_path}")

        # show and close plot
        plt.show()
        plt.close()

        print("\n" + "*==*" * 30 + "\n")

    def bivariate(self):
        self.bivariate_single(self.credit, "CreditCard")
        self.bivariate_single(self.fraud, "FraudData")
        self.bivariate_single(self.ip, "IPAddress Map")

    # ------------Handle Outliers------------#
    def batch_impute_outliers(self, df, label, eligible_cols=None, threshold=3):
        """
        Class-aware batch imputation using IQR and Z-score.
        Handles outliers separately for Class 0 and Class 1.
        Args:
            df (pd.DataFrame): The input DataFrame.
            eligible_cols (list, optional): List of columns to apply imputation.
                                            If None, detects based on data type.
            threshold (float): Z-score threshold for outlier detection.(95%)

        Returns:
            pd.DataFrame: The DataFrame with imputed values.
        """

        df = df.copy()

        if eligible_cols is None:
            exclude = {
                "Ip_Address",
                "Lower_Bound_Ip_Address",
                "Upper_Bound_Ip_Address",
                "User_Id",
                "Device_Id",
            }
            numeric_cols = df.select_dtypes(include="number").columns
            eligible_cols = [
                col for col in numeric_cols if col not in exclude and col != "Class"
            ]

        if "Class" not in df.columns:
            print(
                f"{label}: 'Class' column not found. Skipping class-aware imputation."
            )
            rel_processed_dir = self.safe_relpath(self.processed_dir)

            # Save processed data to CSV
            df_name = os.path.join(self.processed_dir, f"{label}_processed.csv")
            df.to_csv(df_name, index=False)
            print(f"{label}: Processed DataFrame saved to {rel_processed_dir}.\n")
            return df

        for col in eligible_cols:
            for class_value in [0, 1]:
                class_df = df[df["Class"] == class_value]

                # IQR
                Q1 = class_df[col].quantile(0.25)
                Q3 = class_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_iqr = Q1 - 1.5 * IQR
                upper_iqr = Q3 + 1.5 * IQR

                # Z-score
                z_scores = (class_df[col] - class_df[col].mean()) / class_df[col].std()
                outliers = (
                    (class_df[col] < lower_iqr)
                    | (class_df[col] > upper_iqr)
                    | (z_scores.abs() > threshold)
                )

                outlier_indices = class_df[outliers].index

                # Impute
                median_value = class_df[col].median()
                # Avoids re-imputing identical values
                if not df.loc[outlier_indices, col].eq(median_value).all():
                    df.loc[outlier_indices, col] = median_value

        print(f"{label}: Outliers handled.")

        rel_processed_dir = self.safe_relpath(self.processed_dir)

        # Save processed data to CSV
        df_name = os.path.join(self.processed_dir, f"{label}_processed.csv")
        df.to_csv(df_name, index=False)
        print(f"{label}: Processed DataFrame saved to {rel_processed_dir}.\n")

        return df

    def impute_outliers_iqr_zscore(self):
        self.batch_impute_outliers(self.credit, "CreditCard")
        self.batch_impute_outliers(self.fraud, "FraudData")
        self.batch_impute_outliers(self.ip, "IPAddressMap")

    # ------------Convert IP adress to Integer and Merge Datasets------------#
    @staticmethod
    def map_ip_to_country(df, ip_values):
        match = df[
            (df["Lower_Bound_Ip_Address"] <= ip_values)
            & (df["Upper_Bound_Ip_Address"] >= ip_values)
        ]
        return match["Country"].values[0] if not match.empty else "Unknown"

    def apply_map_ip_to_country(self):
        self.fraud_by_country = self.fraud.copy()
        # Apply country mapping
        self.fraud_by_country["Country"] = self.fraud_by_country["Ip_Address"].apply(
            lambda ip: self.map_ip_to_country(self.ip, ip)
        )

        unknowns = self.fraud_by_country[self.fraud_by_country["Country"] == "Unknown"]
        if not unknowns.empty:
            print(f"{len(unknowns)} IPs could not be mapped to any country.")

        # Save processed data to CSV
        rel_processed_dir = self.safe_relpath(self.processed_dir)
        df_name = os.path.join(self.processed_dir, "MapIPtoCountry.csv")
        self.fraud_by_country.to_csv(df_name, index=False)
        print(f"Merged DataFrame 'MapIPtoCountry' saved to {rel_processed_dir}.\n")
