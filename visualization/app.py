import streamlit as st
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from modelDefinition import SimpleGNN
from pyvis.network import Network
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from functools import partial

st.set_page_config(page_title="Graph Visualization", layout="wide")
st.title("Graph Visualization and GNN Model Prediction")

# Load dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# Initialize session state for custom graph creation
if 'custom_graph_mode' not in st.session_state:
    st.session_state.custom_graph_mode = False
if 'custom_nodes' not in st.session_state:
    st.session_state.custom_nodes = []
if 'custom_edges' not in st.session_state:
    st.session_state.custom_edges = []
if 'next_node_id' not in st.session_state:
    st.session_state.next_node_id = 0

# Toggle between dataset and custom graph mode
st.sidebar.title("Graph Selection Mode")
graph_mode = st.sidebar.radio(
    "Choose graph source:",
    ["Dataset Graphs", "Create Your Own Graph"]
)
st.session_state.custom_graph_mode = (graph_mode == "Create Your Own Graph")

if st.session_state.custom_graph_mode:
    # Custom graph creation UI
    st.header("Create Your Own Graph")
    
    # Number of node feature classes from the dataset
    num_classes = dataset[0].x.shape[1]
    
    # Interface for adding nodes
    st.subheader("Add Node")
    node_feature_class = st.selectbox("Select Node Class", 
                                     options=list(range(num_classes)),
                                     help="Choose the class for this node")
    
    if st.button("Add Node"):
        # Create one-hot encoding for the feature
        node_features = np.zeros(num_classes)
        node_features[node_feature_class] = 1.0
        
        # Add node to the list
        st.session_state.custom_nodes.append({
            'id': st.session_state.next_node_id,
            'features': node_features
        })
        st.session_state.next_node_id += 1
        st.success(f"Added node {st.session_state.next_node_id-1} with class {node_feature_class}")
    
    # Interface for adding edges
    if len(st.session_state.custom_nodes) >= 2:
        st.subheader("Add Edge")
        col1, col2 = st.columns(2)
        with col1:
            source_node = st.selectbox("Source Node", 
                                     options=[node['id'] for node in st.session_state.custom_nodes],
                                     format_func=lambda x: f"Node {x}")
        with col2:
            target_node = st.selectbox("Target Node", 
                                      options=[node['id'] for node in st.session_state.custom_nodes],
                                      format_func=lambda x: f"Node {x}")
        
        if st.button("Add Edge"):
            # Check if edge already exists
            edge_exists = any(e['source'] == source_node and e['target'] == target_node for e in st.session_state.custom_edges)
            reverse_edge_exists = any(e['source'] == target_node and e['target'] == source_node for e in st.session_state.custom_edges)
            
            if source_node == target_node:
                st.error("Cannot create self-loop (edge to itself)")
            elif edge_exists or reverse_edge_exists:
                st.error("Edge already exists")
            else:
                st.session_state.custom_edges.append({
                    'source': source_node,
                    'target': target_node
                })
                st.success(f"Added edge from Node {source_node} to Node {target_node}")
    
    # Reset custom graph button
    if st.button("Reset Custom Graph"):
        st.session_state.custom_nodes = []
        st.session_state.custom_edges = []
        st.session_state.next_node_id = 0
        st.success("Custom graph has been reset")
    
    # Display current custom graph state
    st.subheader("Current Custom Graph")
    st.write(f"Nodes: {len(st.session_state.custom_nodes)}")
    st.write(f"Edges: {len(st.session_state.custom_edges)}")
    
    # Convert custom graph to the format needed for visualization and model prediction
    if len(st.session_state.custom_nodes) > 0:
        # Create node features tensor
        node_features = np.array([node['features'] for node in st.session_state.custom_nodes])
        
        # Create edge_index tensor
        if len(st.session_state.custom_edges) > 0:
            edge_sources = [e['source'] for e in st.session_state.custom_edges]
            edge_targets = [e['target'] for e in st.session_state.custom_edges]
            # Create bidirectional edges for undirected graph
            edge_index = torch.tensor([
                edge_sources + edge_targets,  # Source nodes (including reverse edges)
                edge_targets + edge_sources   # Target nodes (including reverse edges)
            ], dtype=torch.long)
        else:
            # Empty edge_index for a graph with no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        for node in st.session_state.custom_nodes:
            G.add_node(node['id'])
        for edge in st.session_state.custom_edges:
            G.add_edge(edge['source'], edge['target'])
        
        # For compatibility with existing code
        graph = type('CustomGraph', (), {
            'x': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': edge_index,
            'edge_attr': None,
            'y': None  # Remove actual label for custom graph
        })
    else:
        # If no nodes, show a message
        st.info("Add nodes and edges to create your graph")
        # Set placeholder graph
        graph = dataset[0]  # Use first graph from dataset as placeholder
        G = to_networkx(graph, to_undirected=True)
else:
    # Original code for dataset graph selection
    st.header("Dataset Graph Selection")
    graph_idx = st.slider('Select Graph ID', 0, len(dataset)-1, 0)
    graph = dataset[graph_idx]
    # Convert to NetworkX graph
    G = to_networkx(graph, to_undirected=True)

# Get node features (one-hot encoded)
node_features = graph.x.numpy()

# Get edge features (one-hot encoded) - assuming they exist in the dataset as edge attributes
edge_features = graph.edge_attr.numpy() if graph.edge_attr is not None else None

# Debugging: Check the shape and a few node feature values
st.write(f"Node Features Shape: {node_features.shape}")
st.write(f"Sample Node Features: {node_features[:5]}")  # Show the first 5 nodes' features

if edge_features is not None:
    st.write(f"Edge Features Shape: {edge_features.shape}")
    st.write(f"Sample Edge Features: {edge_features[:5]}")  # Show the first 5 edges' features

# Color nodes based on the class they belong to (the index of the '1' in the one-hot encoding)
num_classes = node_features.shape[1]  # Assuming the one-hot encoding matches the number of classes

# Create a Pyvis network
net = Network(height='500px', width='100%', notebook=True, bgcolor="#ffffff", font_color="black")

# Define a list of colors for each class
colors = plt.cm.get_cmap('tab10', num_classes)  # 'tab10' colormap provides distinct colors

# Add nodes with color based on the one-hot encoding
for node, features in enumerate(node_features):
    # Find the index of the '1' in the one-hot encoded vector (this is the class)
    node_class = np.argmax(features)  # The class is the index where the '1' is
    color = colors(node_class)  # Use colormap to get a distinct color for each class
    rgba_color = color[:3]  # Get RGB values (ignore alpha)
    color_rgb = f"rgb({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)})"  # Convert to RGB

    feature_text = ", ".join([f"{v:.2f}" for v in features])
    net.add_node(node, label=f"Node {node}", title=f"Features: {feature_text}", color=color_rgb)

# Add edges with color and display one-hot encoding in tooltip
if edge_features is not None:
    for edge_idx, (edge, features) in enumerate(zip(G.edges(), edge_features)):
        # Find the index of the '1' in the one-hot encoded edge feature vector
        edge_class = np.argmax(features)  # The class is the index where the '1' is
        color = colors(edge_class)  # Use colormap to get a distinct color for each class
        rgba_color = color[:3]  # Get RGB values (ignore alpha)
        color_rgb = f"rgb({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)})"  # Convert to RGB

        # Convert edge feature to text (e.g., one-hot encoding)
        edge_feature_text = ", ".join([f"Class {i}: {v:.2f}" for i, v in enumerate(features)])

        # Add the edge with the assigned color and tooltip displaying one-hot encoding
        net.add_edge(edge[0], edge[1], color=color_rgb, width=2, title=f"Edge Features: {edge_feature_text}")  # Tooltip shows edge features

# Render the network and show it in the Streamlit app
net.show('graph_colored_by_class_with_edges_and_tooltips.html')

# Embed the HTML file in the Streamlit app
st.components.v1.html(open('graph_colored_by_class_with_edges_and_tooltips.html', 'r').read(), height=600)

# Model loading
# Load the selected models
@st.cache_resource
def load_all_models():
    in_dim = node_features.shape[1]  # Number of node features
    hidden_dim = 64
    out_dim = 2  # For MUTAG (binary classification)
    num_layers = 4

    models = {}
    for model_type in ["GCN", "GAT", "SAGE"]:
        model = SimpleGNN(in_dim, hidden_dim, out_dim, num_layers, model_type)
        model.load_state_dict(torch.load(f'visualization\models\{model_type}_model.pth', map_location=torch.device('cpu')))
        model.eval()
        models[model_type] = model
    return models

models = load_all_models()

# Prepare input tensors
x = torch.tensor(graph.x, dtype=torch.float32)  # Node features
edge_index = torch.tensor(graph.edge_index, dtype=torch.long)  # Edge indices
edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32) if graph.edge_attr is not None else None  # Edge features
batch = torch.zeros(x.size(0), dtype=torch.long)  # All nodes belong to the same graph (batch 0)

# Predict button
if st.button('Predict with all Models'):
    actual_target = graph.y.item() if graph.y is not None else None  # Handle None case

    # Show the actual target if available
    if actual_target is not None:
        st.subheader(f"Actual Target: {actual_target}")
    else:
        st.subheader("Actual Target: Not available (custom graph)")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with torch.no_grad():
        for idx, (model_name, model) in enumerate(models.items()):
            output, _ = model(x, edge_index, batch=batch)  # Forward pass
            predicted_class = output.argmax(dim=1).item()

            if idx == 0:
                with col1:
                    st.markdown(f"### {model_name}")
                    if actual_target is not None and predicted_class == actual_target:
                        st.success(f"Predicted Class: {predicted_class}")
                    else:
                        st.info(f"Predicted Class: {predicted_class}")
            elif idx == 1:
                with col2:
                    st.markdown(f"### {model_name}")
                    if actual_target is not None and predicted_class == actual_target:
                        st.success(f"Predicted Class: {predicted_class}")
                    else:
                        st.info(f"Predicted Class: {predicted_class}")
            else:
                with col3:
                    st.markdown(f"### {model_name}")
                    if actual_target is not None and predicted_class == actual_target:
                        st.success(f"Predicted Class: {predicted_class}")
                    else:
                        st.info(f"Predicted Class: {predicted_class}")

# Modify your visualization section like this:

# Visualization section
st.header("GNN Message Passing Insights")
st.write("Click the button below to see insights into the message passing process of the selected GNN model.")

# Use session state to preserve the visualization state
if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "GCN"

# Create a function for visualization to avoid code duplication
def visualize_model(model_type, models, x, edge_index, batch, node_features, num_classes, G):
    model = models[model_type]
    with torch.no_grad():
        _, intermediate_outputs = model(x, edge_index, batch=batch)
        
        # Create tabs for each layer
        tab_labels = [f"Layer {i}" for i in range(len(intermediate_outputs))]
        tabs = st.tabs(tab_labels)
        
        for i, (tab, layer_output) in enumerate(zip(tabs, intermediate_outputs)):
            with tab:
                st.subheader(f"{model_type} Layer {i} Output")
                
                # Convert output to numpy
                layer_output_np = layer_output.numpy()
                
                # Show dimensionality info
                st.write(f"Output shape: {layer_output_np.shape}")
                
                # Create a smaller version of the graph
                net = Network(height='400px', width='100%', notebook=True, 
                            bgcolor="#ffffff", font_color="black")
                
                # For visualization, show subset of nodes
                max_nodes_to_show = 20
                nodes_to_show = min(max_nodes_to_show, len(node_features))
                
                # Get important features
                feature_variances = np.var(layer_output_np, axis=0)
                top_feature_indices = np.argsort(feature_variances)[-3:][::-1]
                
                for node in range(nodes_to_show):
                    features = layer_output_np[node]
                    
                    # Node coloring
                    if i == 0:  # Input layer
                        dominant_feature = np.argmax(node_features[node])
                        color = f"hsl({int(dominant_feature * 360 / num_classes)}, 70%, 50%)"
                    else:
                        dominant_value = features[top_feature_indices[0]]
                        min_val = np.min(layer_output_np[:, top_feature_indices[0]])
                        max_val = np.max(layer_output_np[:, top_feature_indices[0]])
                        normalized = (dominant_value - min_val) / (max_val - min_val + 1e-8)
                        color = f"hsl({int(240 * (1 - normalized))}, 70%, 50%)"
                    
                    # Node label
                    top_features_str = "\n".join([f"F{f_idx}: {features[f_idx]:.2f}" 
                                               for f_idx in top_feature_indices])
                    label = f"Node {node}\n{top_features_str}"
                    net.add_node(node, label=label, color=color, font_size='10px')
                
                # Add edges between visible nodes
                for edge in G.edges():
                    if edge[0] < nodes_to_show and edge[1] < nodes_to_show:
                        net.add_edge(edge[0], edge[1], width=1)
                
                # Show the network
                html_file = f'layer_{i}_{model_type}_visualization.html'
                net.show(html_file)
                st.components.v1.html(open(html_file, 'r').read(), height=500)
                
                # Feature statistics
                st.subheader("Feature Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Feature Means:")
                    st.bar_chart(np.mean(layer_output_np, axis=0))
                
                with col2:
                    st.write("Feature Variances:")
                    st.bar_chart(np.var(layer_output_np, axis=0))
                
                # Final prediction display
                if i == len(intermediate_outputs) - 1:
                    st.subheader("Final Prediction")
                    pooled_output = global_mean_pool(layer_output, batch)
                    prediction = F.softmax(pooled_output, dim=1).numpy()[0]
                    actual_target = graph.y.item() if graph.y is not None else None  # Handle None case

                    st.write(f"Class 0 probability: {prediction[0]:.4f}")
                    st.write(f"Class 1 probability: {prediction[1]:.4f}")
                    st.write(f"Predicted class: {np.argmax(prediction)}")
                    if actual_target is not None:
                        st.write(f"Actual class: {actual_target}")
                    else:
                        st.write("Actual class: Not available (custom graph)")

                    fig, ax = plt.subplots()
                    ax.bar([0, 1], prediction)
                    ax.set_xticks([0, 1])
                    ax.set_ylim(0, 1)
                    ax.set_title("Prediction Probabilities")
                    st.pyplot(fig)

# Model selection
model_options = ["GCN", "GAT", "SAGE"]
selected_model = st.selectbox(
    "Select Model to see Insights",
    model_options,
    index=model_options.index(st.session_state.selected_model),
    key="model_select"
)

# Update session state when model changes
if st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.show_visualization = False

# Visualization button
if st.button('Show Message Passing Insights'):
    st.session_state.show_visualization = not st.session_state.show_visualization

# Show visualization if enabled
if st.session_state.show_visualization:
    visualize_model(
        st.session_state.selected_model,
        models,
        x,
        edge_index,
        batch,
        node_features,
        num_classes,
        G
    )