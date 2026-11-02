"""
Model Training and Evaluation Module
Trains baseline Linear/Ridge, Random Forest, Gradient Boosting, and LightGBM models.
Computes RMSE, MAE, R2, and NRMSE.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics including NRMSE as defined in Section 5.3.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    target_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / target_range) if target_range > 0 else 0.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse": nrmse,
        "nrmse_pct": nrmse * 100.0
    }


class FmaxModelTrainer:
    """
    Manages model initialization, training, evaluation, and comparisons.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            ),
            "Gradient Boosting (GBM)": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state
            ),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        }
        self.trained_models: Dict[str, Any] = {}
        self.metrics_summary: Dict[str, Dict[str, float]] = {}

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Trains all models on X_train and evaluates on X_test.
        Returns a comparative metrics DataFrame.
        """
        results = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            y_pred = model.predict(X_test)
            metrics = compute_regression_metrics(y_test.values, y_pred)
            self.metrics_summary[name] = metrics
            
            results.append({
                "Model": name,
                "RMSE (MHz)": round(metrics["rmse"], 2),
                "MAE (MHz)": round(metrics["mae"], 2),
                "R² Score": round(metrics["r2"], 4),
                "NRMSE (%)": round(metrics["nrmse_pct"], 2)
            })
            
        return pd.DataFrame(results).sort_values(by="RMSE (MHz)")

    def get_best_model(self, metric: str = "rmse") -> Tuple[str, Any]:
        """
        Returns the top performing model based on the selected metric.
        """
        if not self.metrics_summary:
            raise ValueError("No models have been trained yet.")
        best_name = min(self.metrics_summary.keys(), key=lambda k: self.metrics_summary[k][metric])
        return best_name, self.trained_models[best_name]

    def get_feature_importances(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """
        Extracts feature importances for tree-based models or coefficients for linear models.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained.")
        
        model = self.trained_models[model_name]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
            
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        return df_imp

