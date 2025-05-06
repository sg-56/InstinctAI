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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import silhouette_score
import pickle
from auto_shap.auto_shap import generate_shap_values


class ClusterNode:
    def __init__(self, level, data_indices, path, kpi_column=None):
        
        self.id = str(uuid.uuid4())
        self.level = level
        self.indices = data_indices # Store the actual indices from the original dataframe
        # Calculate centroid using the actual rows
        # self.centroid = df.loc[data_indices].mean(axis=0).tolist()
        self.size = len(data_indices)
        self.path = path
        self.children = []
        self.score = None
        self.analysis = None
        self.kpi_column = kpi_column  # Store the KPI column this node is built for

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            "size": self.size,
            "indices": self.indices,
            "path": self.path,
            "score": self.score,
            "analysis": self.analysis,
            "kpi_column": self.kpi_column,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data, df=None):
        """Create a ClusterNode from a dictionary representation."""
        node = cls(
            level=data["level"],
            data_indices=data["indices"],
            path=data["path"],
            kpi_column=data.get("kpi_column")  # Handle backward compatibility
        )
        node.id = data["id"]
        # node.centroid = data["centroid"]
        node.size = data["size"]
        node.score = data["score"]
        node.analysis = data["analysis"]
        
        # Recursively create children
        for child_data in data["children"]:
            child = cls.from_dict(child_data, df)
            node.children.append(child)
            
        return node
    
    def save_to_json(self, filepath):
        """Save the cluster tree to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath, df=None):
        """Load a cluster tree from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, df)
    
    def save_to_pickle(self, filepath):
        """Save the cluster tree to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_pickle(cls, filepath):
        """Load a cluster tree from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ClusteringEngine:
    def __init__(self, max_depth=5, min_cluster_size=10, max_k=6, discrete_numeric_threshold=25):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.max_k = max_k
        self.discrete_numeric_threshold = discrete_numeric_threshold
        self.original_df = None
        self.collinear_features = {}
        self.selected_features = {}  # Dictionary to store selected features for each KPI
        self.kpi_trees = {}  # Dictionary to store cluster trees for each KPI

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
        '''
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model 
            to generalize and improves the interpretability of the model.

        Inputs: 
            x: features dataframe
            threshold: features with correlations greater than this value are removed

        Output: 
            dataframe that contains only the non-highly-collinear features
        '''

        # Calculate the correlation matrix
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                    # Print the correlated features and the correlation value
                    #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = list(set(drop_cols))
        return drops

    
    def feature_selector(self, df: pd.DataFrame, 
                          target_column: str,
                          user_selected_features: Optional[List[str]] = None,
                          regression: bool = None,
                          random_state: int = 42) -> List[str]:
        """
        Perform feature selection using SHAP values.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe containing features and target variable
        target_column : str
            The name of the target variable column
        user_selected_features : list of str, optional
            List of features that user explicitly wants to include
        regression : bool, optional
            If True, use RandomForestRegressor, if False use RandomForestClassifier.
            If None, will be inferred from the target variable type.
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        list
            List of selected feature names
        """
        # Make a copy to avoid modifying the original dataframe
        print(f"Selecting features for target: {target_column}")
        X = df.drop(columns=[target_column]).copy()
        collinear_features = self.get_collinear_features(X, 0.8)
        X.drop(columns=collinear_features, inplace=True)
        self.collinear_features[target_column] = collinear_features
        y = df[target_column].copy()
        
        # Determine task type (regression or classification) if not specified
        if regression is None:
            # If target contains continuous values, assume regression
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
                regression = True
            else:
                regression = False
        
        # Initialize the model based on the task type
        if regression:
            model = RandomForestRegressor(random_state=random_state)
        else:
            model = RandomForestClassifier(random_state=random_state)
        
        # Fit the model
        model.fit(X, y)
        
        _, _, importance_df = generate_shap_values(model, X, regression_model=regression, tree_model=True, boosting_model=True)
        
        # Select features based on criteria
        mean_importance = importance_df['shap_value'].mean()
        selected_features = importance_df[importance_df['shap_value'] >= mean_importance]['feature'].tolist()
        if user_selected_features:
            selected_features = list(set([x for x in selected_features + user_selected_features]))
        
        print(f"Selected {len(selected_features)} features for {target_column}")
        return selected_features

    def _best_k_by_silhouette(self, df):
        best_k = 2
        best_score = -1
        X = df.to_numpy()

        for k in range(2, min(self.max_k + 1, len(df))):
            try:
                labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"Error during clustering with k={k}: {e}")
                continue

        return best_k

    def _build_tree(self, df_features, indices, level, path_prefix, perform_analysis=True, columns_to_analyze=None, kpi_column=None):
        """
        Build the cluster tree recursively.
        
        Args:
            df_features: DataFrame with selected features for clustering
            indices: List of indices from the original DataFrame for this cluster
            level: Current depth level
            path_prefix: Path prefix for this node
            perform_analysis: Whether to perform analysis
            columns_to_analyze: Columns to analyze
            kpi_column: The KPI column this tree is built for
            
        Returns:
            ClusterNode: The created node
        """
        # Create a subset of the feature dataset for clustering
        sub_df_features = df_features.loc[indices]
        
        # Create node with the provided indices (which are the actual DataFrame indices)
        node = ClusterNode(level, indices, path=[path_prefix], kpi_column=kpi_column)
    
        # Perform dataset comparison analysis using the ORIGINAL df
        if perform_analysis and columns_to_analyze and self.original_df is not None:
            # Use full original dataframe and segment from original dataframe
            full_df = self.original_df.copy()
            segment_df = self.original_df.loc[indices].copy()
            
            # Filter analysis to specified columns
            analyze_cols = [col for col in columns_to_analyze if col in full_df.columns]
            node.analysis = self.compare_datasets(
                full_df=full_df,
                segment_df=segment_df,
                columns_to_analyze=analyze_cols
            )

        # Terminal case: reached max depth or min cluster size
        if level >= self.max_depth or len(indices) < self.min_cluster_size:
            return node
        
        # Find optimal number of clusters
        k = self._best_k_by_silhouette(segment_df)
        
        # Perform clustering on the subset
        agg_clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = agg_clustering.fit_predict(segment_df)

        # Calculate silhouette score
        node.score = silhouette_score(segment_df, cluster_labels)

        # Create a mapping from position in sub_df to actual index in original df
        position_to_index = {i: idx for i, idx in enumerate(segment_df.index)}

        # Process each cluster
        for cluster_id in range(k):
            # Find positions where the cluster label matches
            cluster_positions = np.where(cluster_labels == cluster_id)[0]
            
            # Map these positions back to actual DataFrame indices
            cluster_indices = [position_to_index[pos] for pos in cluster_positions]
            
            if len(cluster_indices) >= self.min_cluster_size:
                child_path = f"{path_prefix}_{cluster_id}"
                child_node = self._build_tree(
                    df_features, 
                    cluster_indices, 
                    level + 1, 
                    child_path, 
                    perform_analysis, 
                    columns_to_analyze,
                    kpi_column
                )
                child_node.path = node.path + [child_path]
                node.children.append(child_node)

        return node

    def build_cluster_trees(self, df, columns_to_analyze=None, kpi_columns=None):
        """
        Build separate cluster trees for each KPI column.
        
        Args:
            df: DataFrame to cluster
            columns_to_analyze: List of columns to analyze in each cluster
            kpi_columns: List of KPI columns to build trees for
            
        Returns:
            Dictionary mapping KPI columns to their respective cluster tree root nodes
        """
        # Store the original dataframe for later retrieval
        self.original_df = df.copy()
            
        if kpi_columns is None or len(kpi_columns) == 0:
            raise ValueError("At least one KPI column must be provided")
            
        # Initialize the dictionary to store trees for each KPI
        self.kpi_trees = {}
            
        # For each KPI column, build a separate cluster tree
        for kpi_column in kpi_columns:
            print(f"Building cluster tree for KPI: {kpi_column}")
            
            # Select features specifically for this KPI
            features = self.feature_selector(
                df=df,
                target_column=kpi_column,  # Only this KPI
                user_selected_features=columns_to_analyze
            )
            
            self.selected_features[kpi_column] = features
            
            # Use all indices from the original DataFrame
            indices = df.index.tolist()
            
            # Build tree for this specific KPI
            # Use a feature-filtered dataframe for clustering, but keep original for analysis
            df_features = df[features + [kpi_column]].copy() if kpi_column in df.columns else df[features].copy()
            
            root_node = self._build_tree(
                df_features, 
                indices, 
                level=0, 
                path_prefix=f"root_{kpi_column}", 
                perform_analysis=True, 
                columns_to_analyze=columns_to_analyze,
                kpi_column=kpi_column
            )
            
            # Store the tree
            self.kpi_trees[kpi_column] = root_node
            
        return self.kpi_trees

    def refine_cluster(self, df, indices, n_neighbors=50, columns_to_analyze=None, kpi_column=None):
        """
        Refine a cluster using nearest neighbors.
        
        Args:
            df: DataFrame containing the data
            indices: Indices of the data points to refine
            n_neighbors: Number of neighbors to consider
            columns_to_analyze: List of columns to analyze in the refined cluster
            kpi_column: The KPI column this refined cluster is for
            
        Returns:
            The root node of the refined cluster tree
        """
        # Store the original dataframe for later retrieval if not already stored
        if self.original_df is None:
            self.original_df = df.copy()
            
        if columns_to_analyze is None:
            columns_to_analyze = df.columns.tolist()
        
        # If we have a KPI column, use the selected features for that KPI
        features = None
        if kpi_column and kpi_column in self.selected_features:
            features = self.selected_features[kpi_column]
        
        # If no features specifically selected or no KPI, use all columns
        if not features:
            features = df.columns.tolist()
            if kpi_column and kpi_column in features:
                features.remove(kpi_column)
                
        df_features = df[features].copy()
            
        sub_df = df_features.loc[indices]
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
            df_features, 
            selected_global_indices, 
            level=0, 
            path_prefix="refined", 
            perform_analysis=True, 
            columns_to_analyze=columns_to_analyze,
            kpi_column=kpi_column
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
            raise ValueError("No original DataFrame is stored. Call build_cluster_trees first.")
        
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
    
    def get_labeled_dataframe(self, kpi_column=None, level: Optional[int] = None) -> pd.DataFrame:
        """
        Get the original DataFrame with cluster labels added.
        
        Args:
            kpi_column: The specific KPI column's tree to get labels from (if None, includes all KPI trees)
            level: The specific level to get labels for (if None, includes all levels)
            
        Returns:
            DataFrame with cluster labels as additional columns
        """
        if self.original_df is None or not self.kpi_trees:
            raise ValueError("No cluster trees stored. Call build_cluster_trees first.")
        
        # Start with a copy of the original dataframe
        result_df = self.original_df.copy()
        
        # If specific KPI column provided, only process that tree
        kpi_trees_to_process = {kpi_column: self.kpi_trees[kpi_column]} if kpi_column in self.kpi_trees else self.kpi_trees
        
        # Process each KPI tree
        for kpi, root_node in kpi_trees_to_process.items():
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
                col_name = f"cluster_{kpi}_level_{lvl}"
                result_df[col_name] = pd.Series(index_to_path_by_level[lvl])
                
        return result_df

    # Get all clusters at a specific level for a specific KPI
    def get_all_clusters_at_level(self, kpi_column, level: int) -> Dict[str, pd.DataFrame]:
        """
        Get all clusters at a specific level for a specific KPI as a dictionary of DataFrames.
        
        Args:
            kpi_column: The KPI column to get clusters for
            level: The level to get clusters from
            
        Returns:
            Dictionary mapping cluster paths to their DataFrames
        """
        if kpi_column not in self.kpi_trees:
            raise ValueError(f"No cluster tree for KPI column: {kpi_column}")
            
        root_node = self.kpi_trees[kpi_column]
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