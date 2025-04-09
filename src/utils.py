import pandas as pd
import plotly.express as px

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