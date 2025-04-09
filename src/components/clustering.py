import pandas as pd 
import os 
from sklearn.cluster import AgglomerativeClustering
import shap
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import silhouette_score



import uuid
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

class ClusterNode:
    def __init__(self, level, data_indices, df, path):
        self.id = str(uuid.uuid4())
        self.level = level
        self.indices = data_indices
        self.centroid = df.loc[data_indices].mean(axis=0).tolist()
        self.size = len(data_indices)
        self.path = path
        self.children = []
        self.score = None

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            "centroid": self.centroid,
            "size": self.size,
            "indices": self.indices,
            "path": self.path,
            "score": self.score,
            "children": [child.to_dict() for child in self.children]
        }



class ClusteringEngine:
    def __init__(self, max_depth=3, min_cluster_size=10, max_k=5):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.max_k = max_k


    def getFeatureImportance(self,data:pd.DataFrame,target=None)-> pd.DataFrame:
        try : 
            if target is None:
                raise ValueError("Target is None")
            if target not in  data.columns:
                raise ValueError("Target is not in data columns")
            if data[target].dtype == 'object':
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = DecisionTreeRegressor(random_state=42)
            
            model.fit(data.drop(target, axis=1), data[target])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
    
    
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    
            feature_importance_df = pd.DataFrame({
                                        "Feature": data.columns,
                                        "Importance": mean_abs_shap
                                        }).sort_values(by="Importance", ascending=False).head(10).reset_index(drop=True)
    
            return feature_importance_df
        except Exception as e:
            print(f"Error in getFeatureImportance: {e}")

    def _best_k_by_silhouette(self, df):
        best_k = 2
        best_score = -1
        X = df.to_numpy()

        for k in range(2, min(self.max_k + 1, len(df))):
            labels = GaussianMixture(k, covariance_type='full', random_state=0).fit_predict(X)
            # kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            score = silhouette_score(data, labels)
            # score = silhouette_score(X, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _build_tree(self, df, indices, level, path_prefix):
        node = ClusterNode(level, indices, df, path=[path_prefix])

        if level >= self.max_depth or len(indices) < self.min_cluster_size:
            return node

        sub_df = df.loc[indices]
        k = self._best_k_by_silhouette(sub_df)
        agg_clustering = AgglomerativeClustering(n_clusters=k, random_state=42,linkage='ward')
        labels = agg_clustering.fit_predict(sub_df)

        node.score = silhouette_score(sub_df, labels)

        for cluster_id in range(k):
            cluster_indices = [indices[i] for i in range(len(indices)) if labels[i] == cluster_id]
            if len(cluster_indices) >= self.min_cluster_size:
                child_path = f"{path_prefix}_{cluster_id}"
                child_node = self._build_tree(df, cluster_indices, level + 1, child_path)
                child_node.path = node.path + [child_path]
                node.children.append(child_node)

        return node

    def build_cluster_tree(self, df):
        indices = list(df.index)
        return self._build_tree(df, indices, level=0, path_prefix="root")

    def refine_cluster(self, df, indices, n_neighbors=50):
        sub_df = df.loc[indices]
        X = sub_df.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(indices))).fit(X)
        _, neighbors = nbrs.kneighbors(X)
        selected = list(set(neighbors.flatten()))
        selected_global_indices = [indices[i] for i in selected]

        return self._build_tree(df, selected_global_indices, level=0, path_prefix="refined")




       

