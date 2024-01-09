import os
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from gensim.models import Word2Vec
import pickle

# Đường dẫn đến thư mục nguồn và thư mục đích
source_directory = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\Smart_Contract_semantic_graph\1_reentrancy'
output_directory_node = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\node_features\vuln'
output_directory_edge = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\edge_features\vuln'

# Duyệt qua mỗi file .dot trong thư mục nguồn
for filename in os.listdir(source_directory):
    if filename.endswith('.dot'):
        # Xây dựng đường dẫn đầy đủ đến file .dot
        dot_file_path = os.path.join(source_directory, filename)

        # Lấy tên cơ bản của file .dot
        base_filename = os.path.splitext(filename)[0]

        # Load the semantic graph from the .dot file
        semantic_graph = read_dot(dot_file_path)

        # Create a list of 'sentences'
        sentences = []
        for node in semantic_graph.nodes():
            sentence = [node]
            sentence.extend(semantic_graph.neighbors(node))
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

        # Save node features to a Pickle file in the designated directory
        node_features_file = os.path.join(output_directory_node, f'{base_filename}_node_features.pkl')
        with open(node_features_file, 'wb') as f:
            pickle.dump(node_features, f)

        # Save edge features to a Pickle file in the designated directory
        edge_features_file = os.path.join(output_directory_edge, f'{base_filename}_edge_features.pkl')
        with open(edge_features_file, 'wb') as f:
            pickle.dump(edge_features, f)
