import os

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ModelExplainability:
    """
    A class to perform model explainability using SHAP
    for Random Forest and XGBoost models.
    """

    def __init__(self, test_x_c_path, test_x_f_path, rf_model, xgb_model, plot_dir):
        """
        Initialises the ModelExplainability class.

        Args:
            test_x_c_path (str): Path to the test data CSV for the Random Forest model.
            test_x_f_path (str): Path to the test data CSV for the XGBoost model.
            rf_model: Trained Random Forest model.
            xgb_model: Trained XGBoost model.
            plot_dir (str): Directory to save the SHAP plots.
        """

        self.test_x_c_path = test_x_c_path
        self.test_x_f_path = test_x_f_path
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.plot_dir = plot_dir

        # Create output directory if it does not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

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

    def random_forest_SHAP(self, i=0):
        """
        Generates and displays SHAP explanations for the Random Forest model.

        Args:
            i (int, optional): Index of the sample to explain locally. Defaults to 0.

        Returns:
            shap.force_plot: A force plot object for the local explanation.
        """
        test_x_c = pd.read_csv(self.test_x_c_path)
        print("Test shape:", test_x_c.shape)

        explainer_rf = shap.TreeExplainer(self.rf_model, test_x_c)
        shap_values_rf = explainer_rf.shap_values(test_x_c)

        print("SHAP values shape:", shap_values_rf.shape)
        print("Feature row shape:", test_x_c.iloc[i].shape)

        # Global Importance
        shap.summary_plot(shap_values_rf[..., 1], test_x_c, show=False, max_display=30)
        file_path = os.path.join(self.plot_dir, "summary_rf.png")
        plt.savefig(file_path)
        print(f"\nSHAP summary saved to {self.safe_relpath(file_path)}")

        # Local Explanation
        shap.initjs()
        return shap.force_plot(
            explainer_rf.expected_value[1],  # Explicit fraud class base value
            shap_values_rf[i, :, 1],  # SHAP values for sample i, class 1
            test_x_c.iloc[i],  # Raw feature values
        )

    def xgboost_SHAP(self, i=0):
        """
        Generates and displays SHAP explanations for the XGBoost model.

        Args:
            i (int, optional): Index of the sample to explain locally. Defaults to 0.

        Returns:
            shap.force_plot: A force plot object for the local explanation.
        """
        test_x_f = pd.read_csv(self.test_x_f_path)
        print("Test shape:", test_x_f.shape)

        explainer_xgb = shap.Explainer(self.xgb_model, test_x_f)
        shap_values_xgb = explainer_xgb(test_x_f)

        print("SHAP values shape:", shap_values_xgb.shape)
        print("Feature row shape:", test_x_f.iloc[i].shape)

        # Global Importance
        shap.summary_plot(shap_values_xgb, test_x_f, show=False)

        file_path = os.path.join(self.plot_dir, "summary_xgb.png")
        plt.savefig(file_path)
        print(f"\nSHAP summary saved to {self.safe_relpath(file_path)}")

        # Local Explanation
        single_explanation = explainer_xgb(test_x_f.iloc[i])
        shap.initjs()
        return shap.force_plot(
            explainer_xgb.expected_value,
            single_explanation.values,
            single_explanation.data,
        )
