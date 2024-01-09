import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from gensim.models import Word2Vec
import pickle

# Load the semantic graph from a .dot file
semantic_graph = read_dot('sg.dot')

# Create a list of 'sentences'
sentences = []
for node in semantic_graph.nodes():
    sentence = [node]
    sentence.extmmmmmmmmmmmmmmend(semantic_graph.neighbors(node))
    sentences.append(sentence)

# Train a word2vec model on these 'sentences'
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get the vector for each node
node_features = {node: model.wv[node] for node in model.wv.index_to_key}

# Calculate edge features
edge_features = {}
for edge in semantic_graph.edges():
    node_i, node_j = edge
    if node_i in node_features and node_j in node_features:
        edge_features[edge] = (node_features[node_i] + node_features[node_j]) / 2

# Save node features to a Pickle file
with open('node_features.pkl', 'wb') as f:
    pickle.dump(node_features, f)

# Save edge features to a Pickle file
with open('edge_features.pkl', 'wb') as f:
    pickle.dump(edge_features, f)

# Optional: Print some of the features for verification
for node, features in list(node_features.items())[:5]:
    print(f"Node: {node}, Features: {features}")

for edge, features in list(edge_features.items())[:5]:
    print(f"Edge: {edge}, Features: {features}")
