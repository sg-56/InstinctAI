import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from prophet import Prophet

class TimeSeriesAnalyzer:
    def report_missing_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        abs_limit_days: int = 30,
        rel_limit: float = 0.35
    ) -> str:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        start, end = df[date_col].min(), df[date_col].max()
        total_days = (end - start).days + 1
        present = df[date_col].nunique()
        missing = total_days - present
        frac = missing / total_days

        msg = f"Total = {total_days} days, Missing = {missing} days ({frac:.1%})."
        if missing > abs_limit_days or frac > rel_limit:
            msg += " Exceeds thresholds: segment not suitable for time series analysis."
        else:
            msg += " Missing within acceptable limits."
        return msg

    def bucket_dummies_clean(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        kpi_col: str,
        keep_medium: bool = True,
        low_pct: float = 0.25,
        high_pct: float = 0.75
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        df_out = df.copy()
        perf_maps: Dict[str, pd.DataFrame] = {}

        for col in cat_cols:
            totals = (
                df_out.groupby(col, as_index=False)[kpi_col]
                      .sum()
                      .rename(columns={kpi_col: 'total_kpi'})
            )

            low_th = totals['total_kpi'].quantile(low_pct)
            high_th = totals['total_kpi'].quantile(high_pct)

            totals['bucket'] = totals['total_kpi'].apply(
                lambda x: 'low' if x <= low_th else ('high' if x >= high_th else 'medium')
            )

            perf_maps[col] = totals.copy()

            df_out = df_out.merge(totals[[col, 'bucket']], on=col, how='left')

            dummies = pd.get_dummies(df_out['bucket'], prefix=col, dtype=int)
            req_cols = [f"{col}_{b}" for b in ['low', 'medium', 'high']]
            for c in req_cols:
                if c not in dummies:
                    dummies[c] = 0

            df_out = pd.concat([df_out, dummies[req_cols]], axis=1)
            df_out.drop(columns=['bucket', col], inplace=True)

            if not keep_medium:
                df_out.drop(columns=f"{col}_medium", inplace=True)

        return df_out, perf_maps

    def extract_bucket_values(self, perf_maps: Dict[str, pd.DataFrame], col: str) -> Dict[str, List[str]]:
        return {
            'high': perf_maps[col].loc[perf_maps[col]['bucket'] == 'high', col].tolist(),
            'medium': perf_maps[col].loc[perf_maps[col]['bucket'] == 'medium', col].tolist(),
            'low': perf_maps[col].loc[perf_maps[col]['bucket'] == 'low', col].tolist(),
        }

    def aggregate_columns_by_date(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != date_column]
        agg_dict = {col: 'sum' for col in numeric_columns}
        aggregated_df = df.groupby(date_column).agg(agg_dict).reset_index()
        return aggregated_df

    def ensemble_feature_importance_auto(
        self,
        df: pd.DataFrame, 
        target_col: str,
        features: list = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        if features is None:
            features = [c for c in df.columns if c != target_col]

        X = df[features].copy()
        y = df[target_col].copy()

        y_unique = y.nunique(dropna=False)
        task_type = "classification" if not pd.api.types.is_numeric_dtype(y) or y_unique <= 10 else "regression"

        print(f"Auto-detected task_type = {task_type} (unique target values = {y_unique})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        if task_type == "classification":
            models = [
                RandomForestClassifier(n_estimators=100, random_state=random_state),
                LGBMClassifier(random_state=random_state),
                XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
                LogisticRegression(penalty="l1", solver="saga", max_iter=1000, random_state=random_state)
            ]
        else:
            models = [
                RandomForestRegressor(n_estimators=100, random_state=random_state),
                LGBMRegressor(random_state=random_state),
                XGBRegressor(random_state=random_state),
                Lasso(alpha=0.01, max_iter=10000, random_state=random_state)
            ]

        importances = []
        for model in models:
            model.fit(X_train, y_train)
            imp = self._extract_importance(model, features)
            importances.append(self._normalize_importances(imp))

        df_imp = pd.DataFrame({
            "feature": features,
            "rf_importance": importances[0],
            "lgbm_importance": importances[1],
            "xgb_importance": importances[2],
            "linear_importance": importances[3]
        })
        df_imp["final_importance"] = df_imp[[
            "rf_importance", "lgbm_importance", "xgb_importance", "linear_importance"
        ]].mean(axis=1)

        df_imp.sort_values("final_importance", ascending=False, inplace=True)
        df_imp.reset_index(drop=True, inplace=True)

        return df_imp

    def _extract_importance(self, model, feature_names: list) -> np.ndarray:
        n = len(feature_names)
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return imp if len(imp) == n else np.zeros(n)
        elif hasattr(model, "coef_"):
            coefs = getattr(model, "coef_", None)
            if coefs is not None:
                abs_coef = np.abs(coefs).sum(axis=0) if len(coefs.shape) > 1 else np.abs(coefs).flatten()
                return abs_coef if len(abs_coef) == n else np.zeros(n)
        return np.zeros(n)

    def _normalize_importances(self, imp_array: np.ndarray) -> np.ndarray:
        s = imp_array.sum()
        return imp_array if s == 0 else imp_array / s
    


    def prepare_prophet_data(self, df, date_column, target_column, regressors):

        df[date_column] = pd.to_datetime(df[date_column])

        df_prophet = df[[date_column, target_column]].rename(
            columns={date_column: "ds", target_column: "y"}
        )

        for regressor in regressors:
            if regressor in df.columns:
                df_prophet[regressor] = df[regressor]
            else:
                raise ValueError(f"Regressor '{regressor}' not found in the dataframe.")

        return df_prophet


    def forecast_regressors_for_date_range(self,
        start_date, end_date, training_data, regressors, date_column, forecast_periods=180
    ):
        future_regressor_values = {}

        for regressor in regressors:
            df_regressor = training_data[[date_column, regressor]].rename(
                columns={date_column: "ds", regressor: "y"}
            )

            regressor_model = Prophet(changepoint_prior_scale=0.7)
            regressor_model.fit(df_regressor)

            last_date = df_regressor["ds"].max()
            future_regressor = regressor_model.make_future_dataframe(periods=forecast_periods)
            forecast_regressor = regressor_model.predict(future_regressor)

            future_values = forecast_regressor[
                (forecast_regressor["ds"] > last_date)
                & (forecast_regressor["ds"] >= start_date)
                & (forecast_regressor["ds"] <= end_date)
            ][["ds", "yhat"]]

            future_values["yhat"] = future_values["yhat"].astype(int)
            future_regressor_values[regressor] = future_values.set_index("ds")

        future_df = pd.DataFrame({"ds": pd.date_range(start=start_date, end=end_date)})

        for regressor in regressors:
            regressor_values = future_regressor_values[regressor]

            future_df = future_df.merge(
                regressor_values["yhat"], left_on="ds", right_index=True, how="left"
            )

            future_df[regressor] = future_df["yhat"].fillna(future_df["yhat"].mean())
            future_df = future_df.drop(columns="yhat").rename(columns={regressor: regressor})

        return future_df


    def plot_actual_vs_forecast(
            self,
        historical_data,
        forecasted_values_df,
        modified_forecast_df,
        model,
        kpi,
        date_column,
        past_months=12,
        forecast_months=12,
    ):
        historical_data[date_column] = pd.to_datetime(historical_data[date_column])
        last_actual_data = (
            historical_data.set_index(date_column).last(f"{past_months}M").resample("M").sum()
        )
        forecast_original = model.predict(forecasted_values_df.rename(columns={"ds": "ds"}))
        forecast_original["ds"] = pd.to_datetime(forecast_original["ds"])
        forecast_original_monthly = forecast_original.set_index("ds").resample("M").sum()
        forecast_modified = model.predict(modified_forecast_df.rename(columns={"ds": "ds"}))
        forecast_modified["ds"] = pd.to_datetime(forecast_modified["ds"])
        forecast_modified_monthly = forecast_modified.set_index("ds").resample("M").sum()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=last_actual_data.index,
                y=last_actual_data[kpi],
                mode="lines+markers",
                name=f"Actual Revenue (Last {past_months} Months)",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_original_monthly.index[:forecast_months],
                y=forecast_original_monthly["yhat"][:forecast_months],
                mode="lines+markers",
                name=f"Forecasted Revenue (Original, Next {forecast_months} Months)",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_modified_monthly.index[:forecast_months],
                y=forecast_modified_monthly["yhat"][:forecast_months],
                mode="lines+markers",
                name=f"Forecasted Revenue (Modified, Next {forecast_months} Months)",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title="Revenue Comparison: Actual vs. Forecast (Original & Modified)",
            xaxis_title="Date",
            yaxis_title="Monthly Revenue",
            template="plotly_white",
        )
        return fig


    def adjust_columns(self,df, forecasted_regressors_df, adjustments):
        df = forecasted_regressors_df.copy()

        for col, adjustment in adjustments.items():
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame, skipping.")
                continue

            if isinstance(adjustment, str):
                if adjustment.endswith("%"):
                
                    percent_value = float(adjustment.strip("%").replace("+", ""))
                
                    zeros = df[col] == 0
                    df.loc[zeros, col] = percent_value
                    df.loc[~zeros, col] = df.loc[~zeros, col].astype(float) * (1 + percent_value / 100.0)
                else:
                    # Absolute change
                    abs_value = float(adjustment)
                    df[col] = df[col] + abs_value
        return df
    
    def time_series_analysis(
        self,
        input_file_path_raw_data_csv: str,
        kpi: str,
        no_of_months,
        date_column: str,
        adjustments: dict,
    ):
        df = pd.read_csv(input_file_path_raw_data_csv)
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=False, infer_datetime_format=True)

        start_date = df[date_column].max()
        end_date = start_date + relativedelta(months=no_of_months)
        forecast_periods = (end_date - start_date).days + 30
        df_ts = df
        
        df = df_ts.drop(date_column, axis=1, inplace=False)

        regressors = list(df.columns)
        df_prophet = self.prepare_prophet_data(df_ts, date_column, kpi, regressors)
        model = Prophet(
            n_changepoints=100,
            changepoint_prior_scale=0.2,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )

        for regressor in regressors:
            model.add_regressor(regressor)

        model.fit(df_prophet)

        forecasted_regressors_df = self.forecast_regressors_for_date_range(
            start_date, end_date, df_ts, regressors, date_column, forecast_periods
        )

        df_new = self.adjust_columns(df_ts, forecasted_regressors_df, adjustments)

        fig = self.plot_actual_vs_forecast(
            historical_data=df_ts,
            forecasted_values_df=forecasted_regressors_df,
            modified_forecast_df=df_new, 
            model=model,
            kpi=kpi,
            date_column=date_column,
            past_months=no_of_months,
            forecast_months=no_of_months,
        )

        return fig