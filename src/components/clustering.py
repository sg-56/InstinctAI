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
# import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import silhouette_score
import pickle
from auto_shap.auto_shap import generate_shap_values

class ClusterNode:
    def __init__(self, level, data_indices, df, path):
        self.id = str(uuid.uuid4())
        self.level = level
        # Store the actual indices from the original dataframe
        self.indices = data_indices
        # Calculate centroid using the actual rows
        # self.centroid = df.loc[data_indices].mean(axis=0).tolist()
        self.size = len(data_indices)
        self.path = path
        self.children = []
        self.score = None
        self.analysis = None

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            # "centroid": self.centroid,
            "size": self.size,
            "indices": self.indices,
            "path": self.path,
            "score": self.score,
            "analysis": self.analysis,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data, df=None):
        """Create a ClusterNode from a dictionary representation."""
        node = cls(
            level=data["level"],
            data_indices=data["indices"],
            df=df,
            path=data["path"]
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
    def __init__(self, max_depth=3, min_cluster_size=10, max_k=5, discrete_numeric_threshold=25):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.max_k = max_k
        self.discrete_numeric_threshold = discrete_numeric_threshold
        self.original_df = None
        self.selected_features = None

    @staticmethod
    def feature_selector(
        data_frame: pd.DataFrame,
    target_columns: Union[str, List[str]],
    n_features: Optional[int] = None,
    threshold: Optional[float] = None,
    user_selected_features: Optional[List[str]] = None,
    plot_importance: bool = False,
    target_weights: Optional[Dict[str, float]] = None,
    random_state: int = 42
) -> List[str]:
        """
    Select the most important features using SHAP values, supporting multiple target columns
    and user-specified feature preferences. Task type (regression/classification) is automatically
    determined based on the data type of the target column(s).
    
    Parameters:
    -----------
    data_frame : pandas DataFrame
        DataFrame containing both features and target column(s)
    target_columns : str or list of str
        Name(s) of the target column(s) in the DataFrame
    n_features : int, optional
        Number of top features to select. Either n_features or threshold must be provided.
    threshold : float, optional
        Threshold for cumulative importance. Features are selected until their
        cumulative importance exceeds this threshold (0-1). Either n_features or threshold must be provided.
    user_selected_features : list of str, optional
        List of feature names that the user believes are important and should be included
        regardless of their SHAP importance
    plot_importance : bool, default=True
        Whether to plot feature importance
    target_weights : dict, optional
        Dictionary mapping target column names to their importance weights.
        If not provided, all targets are weighted equally.
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
        # Convert target_columns to list if it's a single string
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        # Input validation
        for col in target_columns:
            if col not in data_frame.columns:
                raise ValueError(f"Target column '{col}' does not exist in the data frame")
        
        if n_features is None and threshold is None:
            raise ValueError("Either n_features or threshold must be provided")
        
        if threshold is not None and (threshold <= 0 or threshold > 1):
            raise ValueError("Threshold must be between 0 and 1")
        
        if n_features is not None and n_features <= 0:
            raise ValueError("n_features must be positive")
        
        # Separate features and targets
        X = data_frame.drop(columns=target_columns)
        
        if n_features is not None and n_features > X.shape[1]:
            n_features = X.shape[1]
            print(f"Warning: n_features was greater than the number of available features. Setting to {n_features}")
        
        # Validate user_selected_features
        if user_selected_features is not None:
            # Convert to set for faster lookup
            user_features_set = set(user_selected_features)
            invalid_features = user_features_set - set(X.columns)
            if invalid_features:
                raise ValueError(f"The following user-selected features are not in the dataset: {invalid_features}")
            
            # Check if user selected all features
            if len(user_features_set) == X.shape[1]:
                print("User selected all features. No SHAP-based selection will be performed.")
                return list(X.columns)
        else:
            user_features_set = set()
        
        # Determine task type for each target column
        task_types = {}
        for col in target_columns:
            # Check if the column contains categorical data
            unique_values = data_frame[col].nunique()
            if pd.api.types.is_numeric_dtype(data_frame[col]):
                # For numeric types, if there are few unique values and they're all integers, it's likely classification
                if unique_values <= 10 and np.array_equal(data_frame[col], data_frame[col].astype(int)):
                    task_types[col] = 'classification'
                else:
                    task_types[col] = 'regression'
            else:
                # Non-numeric types are always classification
                task_types[col] = 'classification'
        
        # Print determined task types
        print("Automatically determined task types:")
        for col, task_type in task_types.items():
            print(f"- {col}: {task_type}")
        
        # Handle target weights
        if target_weights is None:
            # Equal weights for all targets
            target_weights = {name: 1.0/len(target_columns) for name in target_columns}
        else:
            # Validate target weights
            missing_targets = set(target_columns) - set(target_weights.keys())
            if missing_targets:
                warnings.warn(f"Target weights not provided for: {missing_targets}. Using default weight of 0.")
                for target in missing_targets:
                    target_weights[target] = 0.0
                    
            # Normalize weights to sum to 1
            weight_sum = sum(target_weights.values())
            if weight_sum == 0:
                raise ValueError("Sum of target weights cannot be zero")
            target_weights = {k: v/weight_sum for k, v in target_weights.items()}
        
        print(f"Processing {len(target_columns)} target{'s' if len(target_columns) > 1 else ''}")
        for target, weight in target_weights.items():
            print(f"- {target}: weight = {weight:.3f}")
        
        # Calculate feature importance for each target
        feature_importance_dict = {}
        shap_values_dict = {}
        
        for target_name in target_columns:
            if target_weights.get(target_name, 0) == 0:
                print(f"Skipping target '{target_name}' with weight 0")
                continue
                
            print(f"Computing feature importance for target: {target_name}")
            
            # Get target values
            y = data_frame[target_name]
            
            # Get task type for this target
            task_type = task_types[target_name]
            
            # Initialize model based on task type
            if task_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state=random_state)
            elif task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100,n_jobs = -1, random_state=random_state)
            else:
                raise ValueError(f"Invalid task type '{task_type}' for target '{target_name}'")
            
            # Fit the model
            model.fit(X, y)
            
            # Create explainer and compute SHAP values
            _,_,shap_values = generate_shap_values(model,X)
            print("TARGET: ",target_name,shap_values)

            # if len(shap_values.shape) == 3:
            #     shap_values = shap_values[:,:,-1]
            # shap_values = shap_values[:,:,-1]
            
            # For classification tasks with multiple classes, shap_values will be a list
            if isinstance(shap_values, list):
                # Sum across all classes
                shap_values = np.abs(np.array(shap_values)).sum(axis=0)
            
            # Store for later use
            shap_values_dict[target_name] = shap_values['shap_value'].values
            
            # Calculate feature importance as mean absolute SHAP value for each feature
            feature_importance_dict = shap_values.set_index('feature').to_dict()['shap_value']
            # feature_importance_dict[target_name] = feature_importance.set_index('feature').to_dict()['shap_value']
            
        # Combine feature importance from all targets using weights
        combined_importance = np.zeros(X.shape[1])
        # print(feature_importance_dict)
        for target_name, importance in feature_importance_dict.items():
            weight = target_weights.get(target_name, 0)
            combined_importance += importance * weight
        # print(combined_importance)
        
        # Create DataFrame with feature names and importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': combined_importance,
            'User_Selected': [feature in user_features_set for feature in X.columns]
        }).sort_values('Importance', ascending=False)
        
        # Also create target-specific importance dataframes for plotting
        target_importance_dfs = {}
        for target_name, importance in feature_importance_dict.items():
            target_importance_dfs[target_name] = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
        
        
            
           
        
        # Start with user-selected features
        selected_features = list(user_features_set)
        
        # Add features based on SHAP importance (excluding those already selected by user)
        remaining_features = feature_importance_df[~feature_importance_df['User_Selected']]
        
        # Calculate how many additional features to select (if using n_features)
        if n_features is not None:
            remaining_n = max(0, n_features - len(selected_features))
            additional_features = remaining_features['Feature'].tolist()[:remaining_n]
            selected_features.extend(additional_features)
        else:
            # For threshold-based selection, we need to recalculate importance considering already selected features
            if selected_features:
                # Calculate the importance of already selected features
                selected_importance = feature_importance_df[feature_importance_df['User_Selected']]['Importance'].sum()
                total_importance = feature_importance_df['Importance'].sum()
                
                # Calculate the remaining importance needed
                remaining_importance_needed = min(threshold * total_importance - selected_importance, 
                                                remaining_features['Importance'].sum())
                
                if remaining_importance_needed > 0:
                    # Calculate cumulative importance for remaining features
                    remaining_features['Cumulative_Importance'] = remaining_features['Importance'].cumsum()
                    
                    # Select features until we reach the needed importance
                    additional_features = remaining_features[remaining_features['Cumulative_Importance'] <= remaining_importance_needed]['Feature'].tolist()
                    selected_features.extend(additional_features)
            else:
                # If no user-selected features, proceed with normal threshold selection
                remaining_features['Cumulative_Importance'] = remaining_features['Importance'].cumsum() / feature_importance_df['Importance'].sum()
                additional_features = remaining_features[remaining_features['Cumulative_Importance'] <= threshold]['Feature'].tolist()
                selected_features.extend(additional_features)
        
        # If no features were selected, take at least the most important one
        if not selected_features:
            selected_features = [feature_importance_df['Feature'].iloc[0]]
        
        print(f"Selected {len(selected_features)}/{X.shape[1]} features")
        print(f"- User-selected features: {len(user_features_set)}")
        print(f"- SHAP-selected features: {len(selected_features) - len(user_features_set)}")
        
        return selected_features




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

    def build_cluster_tree(self, df, columns_to_analyze=None, kpi_columns=None):
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

        self.selected_features = ClusteringEngine.feature_selector(
            data_frame=df,
            target_columns=kpi_columns or [],
            user_selected_features=columns_to_analyze or df.columns.tolist(),
            threshold=0.3
        )        
        # Use all indices from the original DataFrame
        indices = df.index.tolist()
        
        return self._build_tree(
            df[self.selected_features if self.selected_features else df.columns], 
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