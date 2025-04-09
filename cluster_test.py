import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import uuid
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import plotly.graph_objects as go
import io
from src.components.dataingestion import DataIngestion
from src.components.datapreprocessing import DataFramePreprocessor

from io import BytesIO


ingestor = DataIngestion()
processor = DataFramePreprocessor()

app = FastAPI()

class ClusterNode:
    def __init__(self, level, data_indices, data_points, path):
        self.id = str(uuid.uuid4())
        self.level = level
        self.indices = data_indices
        self.size = len(data_indices)
        self.children = []
        self.path = path
        self.score = None

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            "size": self.size,
            "indices": self.indices,
            "path": self.path,
            "score": self.score,
            "children": [child.to_dict() for child in self.children]
        }

class ClusterEngine:
    def __init__(self, max_k=5):
        self.max_k = max_k
        self.breadcrumbs = []

    def best_k_by_silhouette(self, data):
        best_k = 2
        best_score = -1
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
            score = silhouette_score(data, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def build_cluster_tree(self, data, indices=None, level=0, max_depth=3, min_cluster_size=10, k=2, path_prefix="root"):
        if indices is None:
            indices = list(range(len(data)))

        node = ClusterNode(level, indices, data, path=[path_prefix])

        if level >= max_depth or len(indices) < min_cluster_size:
            return node

        sub_data = data.iloc[indices,:]
        k = self.best_k_by_silhouette(sub_data)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(sub_data)

        node.score = silhouette_score(sub_data, labels)

        for cluster_id in range(k):
            cluster_indices = [indices[i] for i in range(len(indices)) if labels[i] == cluster_id]
            if len(cluster_indices) >= min_cluster_size:
                child_path = f"{path_prefix}_{cluster_id}"
                child_node = self.build_cluster_tree(data, cluster_indices, level + 1, max_depth, min_cluster_size, k, child_path)
                child_node.path = node.path + [child_path]
                node.children.append(child_node)

        return node

    def refine_cluster(self, data, indices, n_neighbors=50, k=2, level=0, max_depth=3, path_prefix="refined"):
        sub_data = data[indices]
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(indices))).fit(sub_data)
        distances, neighbors = nbrs.kneighbors(sub_data)
        selected_indices = list(set(neighbors.flatten()))
        selected_global_indices = [indices[i] for i in selected_indices]

        return self.build_cluster_tree(data, selected_global_indices, level, max_depth, min_cluster_size=5, k=k, path_prefix=path_prefix)

    def save_cluster_tree_to_mongo(self, root_node, db_name="clustering_db", collection_name="cluster_tree"):
        client = MongoClient("mongodb://localhost:27017")
        db = client[db_name]
        collection = db[collection_name]
        collection.delete_many({})
        collection.insert_one(root_node.to_dict())
        print(f"Cluster tree saved to MongoDB collection '{collection_name}' in database '{db_name}'.")

    def get_saved_cluster_tree(self, db_name="clustering_db", collection_name="cluster_tree"):
        client = MongoClient("mongodb://localhost:27017")
        db = client[db_name]
        collection = db[collection_name]
        return collection.find_one()

    def track_breadcrumb(self, path: List[str]):
        self.breadcrumbs = path

    def get_breadcrumb(self):
        return self.breadcrumbs

    def plot_tree(self, node_dict):
        labels = []
        parents = []
        values = []

        def recurse(n, parent_label):
            label = " → ".join(n["path"])
            labels.append(label)
            parents.append(parent_label)
            values.append(n["size"])
            for child in n.get("children", []):
                recurse(child, label)

        recurse(node_dict, "")

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        ))

        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        return fig.to_json()

    def find_cluster_by_label(self, node, label_path):
        if node["path"] == label_path:
            return node
        for child in node.get("children", []):
            found = self.find_cluster_by_label(child, label_path)
            if found:
                return found
        return None

    def get_children_of_path(self, node, label_path):
        target_node = self.find_cluster_by_label(node, label_path)
        if target_node:
            return target_node.get("children", [])
        return []

class DataInput(BaseModel):
    data: pd.DataFrame
    max_depth: Optional[int] = 3
    min_cluster_size: Optional[int] = 10

    class Config:
        arbitrary_types_allowed = True

class RefineInput(BaseModel):
    data: pd.DataFrame
    indices: List[int]
    n_neighbors: Optional[int] = 50
    k: Optional[int] = 2
    max_depth: Optional[int] = 3

    class Config:
        arbitrary_types_allowed = True

class BreadcrumbInput(BaseModel):
    path: List[str]

class LabelPathInput(BaseModel):
    path: List[str]

cluster_engine = ClusterEngine()

@app.post("/cluster")
def cluster_data(input_data: DataInput):
    data = np.array(input_data.data)
    root = cluster_engine.build_cluster_tree(data, max_depth=input_data.max_depth, min_cluster_size=input_data.min_cluster_size)
    cluster_engine.save_cluster_tree_to_mongo(root)
    return {"message": "Cluster tree generated and saved.", "root_id": root.id}

@app.post("/refine")
def refine_data(input_data: RefineInput):
    data = np.array(input_data.data)
    root = cluster_engine.refine_cluster(
        data,
        input_data.indices,
        n_neighbors=input_data.n_neighbors,
        k=input_data.k,
        max_depth=input_data.max_depth
    )
    cluster_engine.save_cluster_tree_to_mongo(root)
    return {"message": "Refined cluster tree saved.", "root_id": root.id}

@app.post("/cluster/upload")
def cluster_from_csv(file: UploadFile = File(...), max_depth: int = 3, min_cluster_size: int = 10):
    contents = file.file.read()
    ingestor.ingest_from_object(contents)
    # df = ingestor.get_data()
    # print(df)
    processor.fit(ingestor.get_data())
    data = processor.transform(ingestor.get_data())
    root = cluster_engine.build_cluster_tree(data, max_depth=max_depth, min_cluster_size=min_cluster_size)
    cluster_engine.save_cluster_tree_to_mongo(root)
    return {"message": "Cluster tree generated from uploaded CSV and saved.", "root_id": root.id}

@app.get("/breadcrumb")
def get_breadcrumb():
    return {"breadcrumbs": cluster_engine.get_breadcrumb()}

@app.post("/breadcrumb")
def set_breadcrumb(input_data: BreadcrumbInput):
    cluster_engine.track_breadcrumb(input_data.path)
    return {"message": "Breadcrumb updated."}

@app.get("/cluster/visualize")
def visualize_cluster():
    tree = cluster_engine.get_saved_cluster_tree()
    if not tree:
        raise HTTPException(status_code=404, detail="No cluster tree found.")
    return cluster_engine.plot_tree(tree)

@app.get("/cluster/tree")
def get_cluster_tree():
    tree = cluster_engine.get_saved_cluster_tree()
    if not tree:
        raise HTTPException(status_code=404, detail="No cluster tree found.")
    return tree

@app.post("/cluster/search")
def search_cluster_by_label(input_data: LabelPathInput):
    tree = cluster_engine.get_saved_cluster_tree()
    if not tree:
        raise HTTPException(status_code=404, detail="No cluster tree found.")
    result = cluster_engine.find_cluster_by_label(tree, input_data.path)
    if not result:
        raise HTTPException(status_code=404, detail="Cluster with specified path not found.")
    return result

@app.post("/cluster/children")
def get_cluster_children(input_data: LabelPathInput):
    tree = cluster_engine.get_saved_cluster_tree()
    if not tree:
        raise HTTPException(status_code=404, detail="No cluster tree found.")
    children = cluster_engine.get_children_of_path(tree, input_data.path)
    return {"children": children}
