import pandas as pd
import numpy as np
import json
import uuid
import pickle
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import silhouette_score

from auto_shap.auto_shap import generate_shap_values

class ClusterNode:
    def __init__(self, level, data_indices, path, kpi_column=None):
        self.id = str(uuid.uuid4())
        self.level = level
        self.indices = data_indices
        self.size = len(data_indices)
        self.path = path
        self.children = []
        self.score = None
        self.analysis = None
        self.kpi_column = kpi_column
        self.cluster_definitions = None

    def to_dict(self, include_indices=True, include_analysis=True):
        return {
            "id": self.id,
            "level": self.level,
            "size": self.size,
            "indices": self.indices if include_indices else None,
            "path": self.path,
            "score": self.score,
            "analysis": self.analysis if include_analysis else None,
            "kpi_column": self.kpi_column,
            "cluster_definitions":self.cluster_definitions,
            "children": [child.to_dict(include_indices, include_analysis) for child in self.children]
        }


class ClusteringEngine:
    def __init__(self, max_depth=5, min_cluster_size=10, max_k=6, discrete_numeric_threshold=25):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.max_k = max_k
        self.discrete_numeric_threshold = discrete_numeric_threshold
        self.original_df = None
        self.preprocessed_df = None
        self.collinear_features = {}
        self.selected_features = {}
        self.kpi_trees = {}

    def to_dict(self):
        """
        Generate a Dictionary for the clusters.
        
        Args:
            df: DataFrame to convert
            filepath: Path to save the JSON file
        """
        if self.kpi_trees is None: 
            raise ValueError("No cluster trees stored. Call build_cluster_trees first.")
        
        # Convert each tree to a dictionary 
        return {kpi: tree.to_dict() for kpi, tree in self.kpi_trees.items()}

    def get_collinear_features(self, x, threshold):
        corr_matrix = x.corr()
        drop_cols = []
        for i in range(len(corr_matrix.columns) - 1):
            for j in range(i + 1):
                val = abs(corr_matrix.iloc[j, i + 1])
                if val >= threshold:
                    drop_cols.append(corr_matrix.columns[i + 1])
        return list(set(drop_cols))

    def feature_selector(self, df, target_column, user_selected_features=None, regression=None, random_state=42, shap_sample_size=1000):
        print(f"Selecting features for target: {target_column}")
        X = df.drop(columns=[target_column])
        collinear_features = self.get_collinear_features(X.select_dtypes(include="number"), 0.8)
        X.drop(columns=collinear_features, inplace=True)
        self.collinear_features[target_column] = collinear_features
        y = df[target_column]

        if regression is None:
            regression = pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10

        model = DecisionTreeRegressor(random_state=random_state) if regression else DecisionTreeClassifier(random_state=random_state)
        sampled_X = X.sample(n=min(shap_sample_size, len(X)), random_state=random_state)
        sampled_y = y.loc[sampled_X.index]
        model.fit(sampled_X, sampled_y)

        print("Generating SHAP values...")
        _, _, importance_df = generate_shap_values(model, sampled_X, regression_model=regression, tree_model=True, boosting_model=True)
        print("SHAP computation complete.")

        mean_importance = importance_df['shap_value'].mean()
        selected_features = importance_df[importance_df['shap_value'] >= mean_importance]['feature'].tolist()
        if user_selected_features:
            selected_features = list(set(selected_features + user_selected_features))

        print(f"Selected {len(selected_features)} features for {target_column}")
        return selected_features

    def _best_k_by_silhouette(self, df):
        # print("Selecting best K using silhouette score...")
        best_k = 2
        best_score = -1
        X = df.to_numpy()
        for k in range(2, min(5, self.max_k + 1)):
            try:
                labels = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=0).fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        return best_k

    def _build_tree(self, df_features, indices, level, path_prefix, perform_analysis=True, columns_to_analyze=None, kpi_column=None):
        # print(f"Building cluster at: {path_prefix}")
        node = ClusterNode(level, indices, path=[path_prefix], kpi_column=kpi_column)
        if perform_analysis and columns_to_analyze and self.original_df is not None:
            full_df = self.original_df
            segment_df = self.original_df.loc[indices]
            analyze_cols = full_df.columns
            node.analysis = self.compare_datasets(
                full_df=full_df,
                segment_df=segment_df,
                columns_to_analyze=analyze_cols
            )
            del full_df,segment_df
            
        if level >= self.max_depth or len(indices) < self.min_cluster_size:
            return node
        segment_df = self.preprocessed_df.loc[indices]
        k = self._best_k_by_silhouette(segment_df)
        cluster_labels = MiniBatchKMeans(n_clusters=k, batch_size=1000).fit_predict(segment_df)
        node.score = silhouette_score(segment_df, cluster_labels)
        position_to_index = {i: idx for i, idx in enumerate(segment_df.index)}

        for cluster_id in range(k):
            cluster_positions = np.where(cluster_labels == cluster_id)[0]
            cluster_indices = [position_to_index[pos] for pos in cluster_positions]
            if len(cluster_indices) >= self.min_cluster_size:
                child_path = f"{path_prefix}_{cluster_id}"
                child_node = self._build_tree(df_features, cluster_indices, level + 1, child_path, perform_analysis, columns_to_analyze, kpi_column)
                child_node.path = node.path + [child_path]
                node.children.append(child_node)
        node.cluster_definitions = self.extract_cluster_definitions(node)
        return node

    def build_cluster_trees(self, raw_df, df, columns_to_analyze=None, kpi_columns=None):
        self.original_df = raw_df
        self.preprocessed_df = df
        if not kpi_columns:
            raise ValueError("At least one KPI column must be provided")
        self.kpi_trees = {}

        def process_kpi(kpi_column):
            print(f"Processing KPI: {kpi_column}")
            features = self.feature_selector(df, kpi_column, user_selected_features=columns_to_analyze)
            self.selected_features[kpi_column] = features
            indices = df.index.tolist()
            df_features = df[features + [kpi_column]] if kpi_column in df.columns else df[features]
            root_node = self._build_tree(df_features, indices, 0, f"root_{kpi_column}", True, columns_to_analyze, kpi_column)
            return (kpi_column, root_node)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_kpi, kpi) for kpi in kpi_columns]
            for future in futures:
                kpi_column, root_node = future.result()
                self.kpi_trees[kpi_column] = root_node

        return self.kpi_trees

    def generate_full_stats_cache(self, full_df):
        self.full_stats = {
            col: {
                "mean": full_df[col].mean() if pd.api.types.is_numeric_dtype(full_df[col]) else None,
                "sum": full_df[col].sum() if pd.api.types.is_numeric_dtype(full_df[col]) else None,
                "value_counts": full_df[col].value_counts(dropna=True) if not pd.api.types.is_numeric_dtype(full_df[col]) else None
            }
            for col in full_df.columns
        }

    def _calculate_percent(self,part, total):
        return round((part / total * 100), 2) if total else 0

    def compare_datasets(
        self, 
        full_df: pd.DataFrame, 
        segment_df: pd.DataFrame, 
        columns_to_analyze: List[str]
    ) -> Dict[str, Any]:
        if segment_df is None:
            return {"error": "Segmented dataset is empty"}

        def analyze_column(col):
            if col not in full_df.columns:
                return (col, {"error": f"Column '{col}' not found in dataset"})
            is_numeric = pd.api.types.is_numeric_dtype(full_df[col])
            unique_vals = len(full_df[col].dropna().unique())
            if is_numeric:
                return (col, self.analyze_discrete_numeric_column(full_df, segment_df, col)
                        if unique_vals <= self.discrete_numeric_threshold
                        else self.analyze_numeric_column(full_df, segment_df, col))
            return (col, self.analyze_categorical_column(full_df, segment_df, col))

        with ThreadPoolExecutor() as executor:
            results = dict(executor.map(analyze_column, columns_to_analyze))
        return results

    def analyze_categorical_column(
        self, full_df: pd.DataFrame, segment_df: pd.DataFrame, col: str
    ) -> Dict[str, Any]:
        full_values = set(full_df[col].dropna().unique())
        segment_values = set(segment_df[col].dropna().unique())
        segment_counts = segment_df[col].value_counts()
        total_segment = segment_df[col].dropna().shape[0]

        def top_bottom(n, reverse=False):
            return [
                {
                    "category": str(idx),
                    "percentage": self._calculate_percent(cnt, total_segment)
                }
                for idx, cnt in (segment_counts.head(n) if not reverse else segment_counts.tail(n)).items()
            ]

        return {
            "unique_categories": {
                "segment": len(segment_values),
                "full": len(full_values),
                "coverage_percentage": self._calculate_percent(len(full_values & segment_values), len(full_values))
            },
            "mode": {
                "category": str(segment_counts.index[0]) if not segment_counts.empty else None,
                "count": int(segment_counts.iloc[0]) if not segment_counts.empty else 0
            },
            "top_categories": top_bottom(3),
            "bottom_categories": top_bottom(3, reverse=True)
        }

    def analyze_discrete_numeric_column(
        self, full_df: pd.DataFrame, segment_df: pd.DataFrame, col: str
    ) -> Dict[str, Any]:
        base = self.analyze_categorical_column(full_df, segment_df, col)

        full_sum, segment_sum = full_df[col].sum(), segment_df[col].sum()
        full_mean, segment_mean = full_df[col].mean(), segment_df[col].mean()

        value_dist = {
            str(val): {
                "full_count": int(full_count),
                "segment_count": int(seg_count),
                "percentage": self._calculate_percent(seg_count, full_count)
            }
            for val in sorted(full_df[col].dropna().unique())
            if (seg_count := segment_df[segment_df[col] == val].shape[0]) / (full_count := full_df[full_df[col] == val].shape[0] if full_df[full_df[col] == val].shape[0] > 0 else 1) > 0.5
        }

        base.update({
            "numeric_stats": {
                "full_dataset": {"mean": float(full_mean), "sum": float(full_sum)},
                "segment": {"mean": float(segment_mean), "sum": float(segment_sum)},
                "contributions": {
                    "sum_contribution_percentage": self._calculate_percent(segment_sum, full_sum),
                    "mean_contribution_percentage": self._calculate_percent(segment_mean, full_mean)
                }
            },
            "value_distribution": value_dist
        })

        return base

    def analyze_numeric_column(
        self, full_df: pd.DataFrame, segment_df: pd.DataFrame, col: str
    ) -> Dict[str, Any]:
        full_sum, segment_sum = full_df[col].sum(), segment_df[col].sum()
        full_mean, segment_mean = full_df[col].mean(), segment_df[col].mean()
        return {
            "full_dataset": {"mean": float(full_mean), "sum": float(full_sum)},
            "segment": {"mean": float(segment_mean), "sum": float(segment_sum)},
            "contributions": {
                "sum_contribution_percentage": self._calculate_percent(segment_sum, full_sum),
                "mean_contribution_percentage": self._calculate_percent(segment_mean, full_mean)
            }
        }
    def extract_cluster_definitions(self,node:ClusterNode):
        """
            Helper function:
              For each hierarchical cluster, compute:
              - cluster_mean for each feature
              - overall_mean for each feature
              - z_score = (cluster_mean - overall_mean) / overall_std
              - abs_z_score for sorting
            Returns dict {cluster_label -> DataFrame}.
        """
        
        numerical_cols = self.original_df.select_dtypes(include='number').columns
        overall_mean = self.original_df[numerical_cols].mean()
        overall_std = self.original_df[numerical_cols].std()  # avoid /0

        definitions = {}
        for idx,cluster in enumerate(node.children):
            cluster_data = self.original_df[numerical_cols].iloc[node.indices,:]
            cluster_mean = cluster_data.mean()
            # print(f'Cluster Mean : {cluster_mean},Overall_mean : {overall_mean} , overall_std : {overall_std}')
            z_scores = (cluster_mean - overall_mean) / overall_std
    
            df_def = pd.DataFrame(
                {
                    "feature": numerical_cols,
                    "cluster_mean": cluster_mean.values,
                    "overall_mean": overall_mean.values,
                    "z_score": z_scores.values,
                }
            )
            # print(z_scores)
            df_def["abs_z_score"] = df_def["z_score"].abs()
            # Sort by largest absolute z-score
            df_def.sort_values("abs_z_score", ascending=False, inplace=True)
            definitions[f'cluster_{idx+1}'] = df_def[['feature','abs_z_score']].set_index('feature').to_dict()

        return definitions