import pandas as pd
import plotly.express as px
import numpy as np

def flatten_cluster_tree(node):
    """Recursively flattens the tree into a list of dicts for plotting."""
    flat = [{
        "id": node.id,
        "label": node.path[-1],
        "parent": node.path[-2] if len(node.path) > 1 else "",
        "value": node.size
    }]
    for child in node.children:
        flat.extend(flatten_cluster_tree(child))
    return flat

def plot_sunburst(root_node):
    flat_data = flatten_cluster_tree(root_node)
    df = pd.DataFrame(flat_data)

    fig = px.sunburst(
        df,
        names='label',
        parents='parent',
        values='value',
        title="Cluster Tree Sunburst",
        maxdepth=-1,
        width=800,
        height=800
    )
    fig.update_traces(insidetextorientation='radial')
    fig.show()

def reduce_memory_usage(df):   
    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_memory} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')
    
    end_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df