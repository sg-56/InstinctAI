import pandas as pd
import numpy as np
import os
import json
import uuid
from typing import List, Dict, Any, Union, Optional, Tuple

# Machine learning imports
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import silhouette_score
import shap

class ClusterNode:
    def __init__(self, level, data_indices, df, path):
        self.id = str(uuid.uuid4())
        self.level = level
        # Store the actual indices from the original dataframe
        self.indices = data_indices
        # Calculate centroid using the actual rows
        self.centroid = df.loc[data_indices].mean(axis=0).tolist()
        self.size = len(data_indices)
        self.path = path
        self.children = []
        self.score = None
        self.analysis = None

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            "centroid": self.centroid,
            "size": self.size,
            "indices": self.indices,
            "path": self.path,
            "score": self.score,
            "analysis": self.analysis,
            "children": [child.to_dict() for child in self.children]
        }


class ClusteringEngine:
    def __init__(self, max_depth=3, min_cluster_size=10, max_k=5, discrete_numeric_threshold=25):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.max_k = max_k
        self.discrete_numeric_threshold = discrete_numeric_threshold
        self.original_df = None

    def getFeatureImportance(self, data: pd.DataFrame, target=None) -> pd.DataFrame:
        try: 
            if target is None:
                raise ValueError("Target is None")
            if target not in data.columns:
                raise ValueError("Target is not in data columns")
            if data[target].dtype == 'object':
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = DecisionTreeRegressor(random_state=42)
            
            model.fit(data.drop(target, axis=1), data[target])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data.drop(target, axis=1))
            
            # Handle different shap_values formats based on model type
            if isinstance(shap_values, list):
                # For multi-class classification
                mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            else:
                # For regression or binary classification
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
            feature_importance_df = pd.DataFrame({
                "Feature": data.drop(target, axis=1).columns,
                "Importance": mean_abs_shap
            }).sort_values(by="Importance", ascending=False).head(10).reset_index(drop=True)
    
            return feature_importance_df
        except Exception as e:
            print(f"Error in getFeatureImportance: {e}")
            return pd.DataFrame({"Feature": [], "Importance": []})

    def _best_k_by_silhouette(self, df):
        best_k = 2
        best_score = -1
        X = df.to_numpy()

        for k in range(2, min(self.max_k + 1, len(df))):
            try:
                labels = GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"Error during clustering with k={k}: {e}")
                continue

        return best_k

    def _build_tree(self, df, indices, level, path_prefix, perform_analysis=True, columns_to_analyze=None):
        """
        Build the cluster tree recursively.
        
        Args:
            df: The full DataFrame
            indices: List of indices from the original DataFrame for this cluster
            level: Current depth level
            path_prefix: Path prefix for this node
            perform_analysis: Whether to perform analysis
            columns_to_analyze: Columns to analyze
            
        Returns:
            ClusterNode: The created node
        """
        # Create node with the provided indices (which are the actual DataFrame indices)
        node = ClusterNode(level, indices, df, path=[path_prefix])

        # Perform dataset comparison analysis
        if perform_analysis and columns_to_analyze:
            full_df = df.copy()
            segment_df = df.loc[indices].copy()
            node.analysis = self.compare_datasets(
                full_df=full_df,
                segment_df=segment_df,
                columns_to_analyze=columns_to_analyze
            )

        # Terminal case: reached max depth or min cluster size
        if level >= self.max_depth or len(indices) < self.min_cluster_size:
            return node

        # Get the subset of data for this cluster
        sub_df = df.loc[indices]
        
        # Find optimal number of clusters
        k = self._best_k_by_silhouette(sub_df)
        
        # Perform clustering on the subset
        agg_clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = agg_clustering.fit_predict(sub_df)

        # Calculate silhouette score
        node.score = silhouette_score(sub_df, cluster_labels)

        # Create a mapping from position in sub_df to actual index in original df
        position_to_index = {i: idx for i, idx in enumerate(sub_df.index)}

        # Process each cluster
        for cluster_id in range(k):
            # Find positions where the cluster label matches
            cluster_positions = np.where(cluster_labels == cluster_id)[0]
            
            # Map these positions back to actual DataFrame indices
            cluster_indices = [position_to_index[pos] for pos in cluster_positions]
            
            if len(cluster_indices) >= self.min_cluster_size:
                child_path = f"{path_prefix}_{cluster_id}"
                child_node = self._build_tree(
                    df, 
                    cluster_indices, 
                    level + 1, 
                    child_path, 
                    perform_analysis, 
                    columns_to_analyze
                )
                child_node.path = node.path + [child_path]
                node.children.append(child_node)

        return node

    def build_cluster_tree(self, df, columns_to_analyze=None):
        """
        Build a cluster tree and analyze each segment.
        
        Args:
            df: DataFrame to cluster
            columns_to_analyze: List of columns to analyze in each cluster
            
        Returns:
            The root node of the cluster tree
        """
        # Store the original dataframe for later retrieval
        self.original_df = df.copy()
        
        if columns_to_analyze is None:
            columns_to_analyze = df.columns.tolist()
            
        # Use all indices from the original DataFrame
        indices = df.index.tolist()
        
        return self._build_tree(
            df, 
            indices, 
            level=0, 
            path_prefix="root", 
            perform_analysis=True, 
            columns_to_analyze=columns_to_analyze
        )

    def refine_cluster(self, df, indices, n_neighbors=50, columns_to_analyze=None):
        """
        Refine a cluster using nearest neighbors.
        
        Args:
            df: DataFrame containing the data
            indices: Indices of the data points to refine
            n_neighbors: Number of neighbors to consider
            columns_to_analyze: List of columns to analyze in the refined cluster
            
        Returns:
            The root node of the refined cluster tree
        """
        # Store the original dataframe for later retrieval if not already stored
        if self.original_df is None:
            self.original_df = df.copy()
            
        if columns_to_analyze is None:
            columns_to_analyze = df.columns.tolist()
            
        sub_df = df.loc[indices]
        X = sub_df.to_numpy()
        
        # Use at most the number of points we have
        real_n_neighbors = min(n_neighbors, len(sub_df))
        nbrs = NearestNeighbors(n_neighbors=real_n_neighbors).fit(X)
        _, neighbor_indices = nbrs.kneighbors(X)
        
        # Map neighbor indices (which are positions in X) back to indices in the sub_df
        neighbor_positions = list(set(neighbor_indices.flatten()))
        
        # Map from positions in sub_df to actual indices in original df
        position_to_orig_index = {i: idx for i, idx in enumerate(sub_df.index)}
        
        # Get the actual DataFrame indices for these neighbors
        selected_global_indices = [position_to_orig_index[pos] for pos in neighbor_positions]

        return self._build_tree(
            df, 
            selected_global_indices, 
            level=0, 
            path_prefix="refined", 
            perform_analysis=True, 
            columns_to_analyze=columns_to_analyze
        )

    # Node retrieval methods
    def get_node_by_level_or_path(self, root_node: ClusterNode, level: Optional[int] = None, 
                                path: Optional[List[str]] = None) -> List[ClusterNode]:
        """
        Find cluster nodes by level and/or path.
        
        Args:
            root_node: The root node of the cluster tree
            level: The level to search for (if None, ignores level constraint)
            path: The path to search for (if None, ignores path constraint)
            
        Returns:
            List of matching cluster nodes
        """
        result = []
        
        def _traverse(node):
            # Check if node matches criteria
            level_match = level is None or node.level == level
            path_match = path is None or self._path_matches(node.path, path)
            
            if level_match and path_match:
                result.append(node)
            
            # Continue traversing children
            for child in node.children:
                _traverse(child)
        
        _traverse(root_node)
        return result
    
    def _path_matches(self, node_path: List[str], search_path: List[str]) -> bool:
        """Check if a node's path matches or contains the search path."""
        if len(search_path) > len(node_path):
            return False
        
        # Check if the beginning of node_path matches search_path
        for i in range(len(search_path)):
            if search_path[i] != node_path[i]:
                return False
        return True
    
    def get_node_dataframe(self, node: ClusterNode) -> pd.DataFrame:
        """
        Get the DataFrame corresponding to a specific cluster node.
        
        Args:
            node: The cluster node to extract data for
            
        Returns:
            DataFrame containing the data points in the cluster
        """
        if self.original_df is None:
            raise ValueError("No original DataFrame is stored. Call build_cluster_tree first.")
        
        # Use node.indices which contain actual DataFrame indices
        return self.original_df.loc[node.indices].copy()
    
    def get_dataframes_by_level_or_path(self, root_node: ClusterNode, level: Optional[int] = None, 
                                      path: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get DataFrames for nodes matching specified level and/or path.
        
        Args:
            root_node: The root node of the cluster tree
            level: The level to search for (if None, ignores level constraint)
            path: The path to search for (if None, ignores path constraint)
            
        Returns:
            Dictionary mapping node IDs to their corresponding DataFrames
        """
        matching_nodes = self.get_node_by_level_or_path(root_node, level, path)
        
        result = {}
        for node in matching_nodes:
            node_df = self.get_node_dataframe(node)
            # Use node path as the key, converted to a string for dictionary key
            key = f"level_{node.level}_{'_'.join(node.path)}"
            result[key] = node_df
            
        return result
    
    def find_node_by_exact_path(self, root_node: ClusterNode, path: List[str]) -> Optional[ClusterNode]:
        """
        Find a specific node by its exact path.
        
        Args:
            root_node: The root node of the cluster tree
            path: The exact path to find
            
        Returns:
            The matching node or None if not found
        """
        def _find(node):
            if node.path == path:
                return node
            
            for child in node.children:
                result = _find(child)
                if result:
                    return result
            
            return None
        
        return _find(root_node)
    
    def get_dataframe_by_path_string(self, root_node: ClusterNode, path_string: str) -> pd.DataFrame:
        """
        Get a DataFrame for a node specified by a path string like "root_0_1".
        
        Args:
            root_node: The root node of the cluster tree
            path_string: A string representing the path, with components separated by underscores
            
        Returns:
            DataFrame for the specified node
        """
        path = path_string.split('_')
        node = self.find_node_by_exact_path(root_node, path)
        
        if node:
            return self.get_node_dataframe(node)
        else:
            raise ValueError(f"No node found with path: {path_string}")
    
    def get_labeled_dataframe(self, root_node: ClusterNode, level: Optional[int] = None) -> pd.DataFrame:
        """
        Get the original DataFrame with cluster labels added.
        
        Args:
            root_node: The root node of the cluster tree
            level: The specific level to get labels for (if None, includes all levels)
            
        Returns:
            DataFrame with cluster labels as additional columns
        """
        if self.original_df is None:
            raise ValueError("No original DataFrame is stored. Call build_cluster_tree first.")
        
        # Start with a copy of the original dataframe
        result_df = self.original_df.copy()
        
        # Create a mapping of index to path for each level
        index_to_path_by_level = {}
        
        def _collect_paths(node):
            if level is None or node.level == level:
                node_level = node.level
                if node_level not in index_to_path_by_level:
                    index_to_path_by_level[node_level] = {}
                
                path_str = '_'.join(node.path)
                for idx in node.indices:
                    index_to_path_by_level[node_level][idx] = path_str
            
            for child in node.children:
                _collect_paths(child)
        
        _collect_paths(root_node)
        
        # Add columns for each level
        for lvl in sorted(index_to_path_by_level.keys()):
            col_name = f"cluster_level_{lvl}"
            result_df[col_name] = pd.Series(index_to_path_by_level[lvl])
            
        return result_df

    # Get all clusters at a specific level as a dictionary of DataFrames
    def get_all_clusters_at_level(self, root_node: ClusterNode, level: int) -> Dict[str, pd.DataFrame]:
        """
        Get all clusters at a specific level as a dictionary of DataFrames.
        
        Args:
            root_node: The root node of the cluster tree
            level: The level to get clusters from
            
        Returns:
            Dictionary mapping cluster paths to their DataFrames
        """
        nodes = self.get_node_by_level_or_path(root_node, level=level)
        
        result = {}
        for node in nodes:
            path_str = '_'.join(node.path)
            result[path_str] = self.get_node_dataframe(node)
            
        return result
    
    # Dataset comparison functions
    def compare_datasets(
        self, 
        full_df: pd.DataFrame, 
        segment_df: pd.DataFrame, 
        columns_to_analyze: List[str]
    ) -> Dict[str, Any]:
        """
        Compare a segment to the full dataset across specified columns.
        
        Args:
            full_df: The complete dataset
            segment_df: A segment/subset of the dataset
            columns_to_analyze: Columns to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if segment_df.empty:
            return {"error": "Segmented dataset is empty"}
        
        result = {}
        
        for col in columns_to_analyze:
            if col not in full_df.columns:
                result[col] = {"error": f"Column '{col}' not found in dataset"}
                continue
                
            is_numeric = pd.api.types.is_numeric_dtype(full_df[col])
            
            if is_numeric:
                unique_count = len(full_df[col].dropna().unique())
                
                if unique_count <= self.discrete_numeric_threshold:
                    result[col] = self.analyze_discrete_numeric_column(full_df, segment_df, col)
                else:
                    result[col] = self.analyze_numeric_column(full_df, segment_df, col)
            else:
                result[col] = self.analyze_categorical_column(full_df, segment_df, col)
        
        return result

    def analyze_categorical_column(
        self, 
        full_df: pd.DataFrame, 
        segment_df: pd.DataFrame, 
        col: str
    ) -> Dict[str, Any]:
        """
        Analyze a categorical column, comparing segment to full dataset.
        """
        full_values = set(full_df[col].dropna().unique())
        segment_values = set(segment_df[col].dropna().unique())
        categories_in_both = full_values.intersection(segment_values)
        full_unique_count = len(full_values)
        segment_unique_count = len(segment_values)
        coverage_percentage = round((len(categories_in_both) / full_unique_count) * 100, 2) if full_unique_count > 0 else 0

        segment_counts = segment_df[col].value_counts()
        total_segment_count = segment_df[col].dropna().shape[0]  

        mode = segment_counts.index[0] if not segment_counts.empty else None
        mode_count = segment_counts.iloc[0] if not segment_counts.empty else 0

        top_count = min(3, len(segment_counts))
        top_categories = []
        for i in range(top_count):
            if i < len(segment_counts):
                category = segment_counts.index[i]
                count = segment_counts.iloc[i]
                percentage = round((count / total_segment_count) * 100, 2) if total_segment_count > 0 else 0
                top_categories.append({"category": str(category), "percentage": percentage})

        bottom_count = min(3, len(segment_counts))
        bottom_categories = []
        for i in range(1, bottom_count + 1):
            if len(segment_counts) >= i:
                category = segment_counts.index[-i]
                count = segment_counts.iloc[-i]
                percentage = round((count / total_segment_count) * 100, 2) if total_segment_count > 0 else 0
                bottom_categories.append({"category": str(category), "percentage": percentage})
        
        return {
            "unique_categories": {
                "segment": segment_unique_count,
                "full": full_unique_count,
                "coverage_percentage": coverage_percentage
            },
            "mode": {
                "category": str(mode) if mode is not None else None,
                "count": int(mode_count)
            },
            "top_categories": top_categories,
            "bottom_categories": bottom_categories
        }

    def analyze_discrete_numeric_column(
        self, 
        full_df: pd.DataFrame, 
        segment_df: pd.DataFrame, 
        col: str
    ) -> Dict[str, Any]:
        """
        Analyze a discrete numeric column, combining categorical and numeric analysis.
        """
        categorical_analysis = self.analyze_categorical_column(full_df, segment_df, col)
        
        full_mean = full_df[col].mean()
        full_sum = full_df[col].sum()
        segment_mean = segment_df[col].mean()
        segment_sum = segment_df[col].sum()
        
        sum_contribution = (segment_sum / full_sum * 100) if full_sum != 0 else 0
        mean_contribution = (segment_mean / full_mean * 100) if full_mean != 0 else 0
        
        categorical_analysis["numeric_stats"] = {
            "full_dataset": {
                "mean": float(full_mean),
                "sum": float(full_sum)
            },
            "segment": {
                "mean": float(segment_mean),
                "sum": float(segment_sum)
            },
            "contributions": {
                "sum_contribution_percentage": round(sum_contribution, 2),
                "mean_contribution_percentage": round(mean_contribution, 2)
            }
        }
        
        all_values = {}
        for value in sorted(full_df[col].dropna().unique()):
            full_count = len(full_df[full_df[col] == value])
            segment_count = len(segment_df[segment_df[col] == value])
            percentage = round((segment_count / full_count * 100), 2) if full_count > 0 else 0
            
            if percentage > 50:
                all_values[str(value)] = {
                    "full_count": full_count,
                    "segment_count": segment_count,
                    "percentage": percentage
                }
        
        categorical_analysis["value_distribution"] = all_values
        
        return categorical_analysis

    def analyze_numeric_column(
        self, 
        full_df: pd.DataFrame, 
        segment_df: pd.DataFrame, 
        col: str
    ) -> Dict[str, Any]:
        """
        Analyze a continuous numeric column.
        """
        full_mean = full_df[col].mean()
        full_sum = full_df[col].sum()
        
        segment_mean = segment_df[col].mean()
        segment_sum = segment_df[col].sum()
        
        sum_contribution = (segment_sum / full_sum * 100) if full_sum != 0 else 0
        mean_contribution = (segment_mean / full_mean * 100) if full_mean != 0 else 0
        
        return {
            "full_dataset": {
                "mean": float(full_mean),
                "sum": float(full_sum)
            },
            "segment": {
                "mean": float(segment_mean),
                "sum": float(segment_sum)
            },
            "contributions": {
                "sum_contribution_percentage": round(sum_contribution, 2),
                "mean_contribution_percentage": round(mean_contribution, 2)
            }
        }