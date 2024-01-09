from solidity_parser import parser
import json
import re
from graphviz import Digraph
import os

def convert_sol_to_json(sol_file_path, json_file_path):
    with open(sol_file_path, 'r') as file:
        source_code = file.read()
    parsed_data = parser.parse(source_code)
    json_data = json.dumps(parsed_data, indent=4)
    with open(json_file_path, 'w') as file:
        file.write(json_data)

# Function to generate a unique identifier for each node
node_pattern = re.compile(r'"(\w+)" \[label="([^"]+)"\];')
edge_pattern = re.compile(r'"(\w+)" -> "(\w+)"')

# Define the node types for classification
statement_nodes = ["FunctionDefinition", "IfStatement", "FunctionCall", "ExpressionStatement"]
variable_nodes = ["Identifier"]
operator_nodes = ["BinaryOperation"]
extended_library_nodes = ["SafeMath"]

# Function to classify nodes based on the label
def classify_node(label):
    if any(node_type in label for node_type in statement_nodes):
        return 'statement'
    elif any(node_type in label for node_type in variable_nodes):
        return 'variable'
    elif any(node_type in label for node_type in operator_nodes):
        return 'operator'
    elif any(node_type in label for node_type in extended_library_nodes):
        return 'extended_library'
    else:
        return 'other'

# Get color for a node type
def node_type_color(node_type):
    return {
        'variable': 'blue',
        'statement': 'orange',
        'operator': 'brown',
        'extended_library': 'green',
        'other': 'black'
    }.get(node_type, 'black')

# Function to create a semantic graph from DOT content
def create_semantic_graph(dot_content):
    sg = Digraph(strict=True)
    sg.node('Enter')

    # Parse nodes and classify them
    for match in node_pattern.findall(dot_content):
        node_id, label = match
        node_type = classify_node(label)
        sg.node(node_id, label=label, color=node_type_color(node_type))

    # Parse edges
    for src, tgt in edge_pattern.findall(dot_content):
        sg.edge(src, tgt)

    return sg

# Function to read JSON AST from file
def read_json_ast(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
def traverse_ast(node, graph_components, existing_ids, parent=None):
    if isinstance(node, dict):
        node_id = generate_unique_id(node, existing_ids)
        existing_ids.add(node_id)
        node_label = node.get('name', node.get('type', 'Node'))
        
        # Create a node declaration
        graph_components['nodes'].add(f'"{node_id}" [label="{node_label}"]')
        if parent:
            # Add an edge from the parent to this node
            graph_components['edges'].add(f'"{parent}" -> "{node_id}"')

        # Recurse through children or properties of the node
        for key, value in node.items():
            if isinstance(value, list) or isinstance(value, dict):
                traverse_ast(value, graph_components, existing_ids, parent=node_id)
    elif isinstance(node, list):
        for item in node:
            traverse_ast(item, graph_components, existing_ids, parent)
def generate_unique_id(node, existing_ids):
    base_id = f"{node.get('type', 'Node')}_{node.get('name', 'Unnamed')}"
    unique_id = base_id
    counter = 0
    # If the id already exists, append a counter to it to make it unique
    while unique_id in existing_ids:
        counter += 1
        unique_id = f"{base_id}_{counter}"
    return unique_id
# Convert JSON AST to DOT format
def json_ast_to_dot(ast, output_file):
    graph_components = {'nodes': set(), 'edges': set()}
    existing_ids = set()
    traverse_ast(ast, graph_components, existing_ids)

    # Write the graph components to the DOT format file
    with open(output_file, 'w') as file:
        file.write('strict digraph {\n')
        for node in graph_components['nodes']:
            file.write(f'    {node};\n')
        for edge in graph_components['edges']:
            file.write(f'    {edge};\n')
        file.write('}\n')

import re
from graphviz import Digraph

# Regular expressions to identify nodes and edges
node_pattern = re.compile(r'"(\w+)" \[label="([^"]+)"\];')
edge_pattern = re.compile(r'"(\w+)" -> "(\w+)"')

# Define the node types for classification
statement_nodes = ["FunctionDefinition", "IfStatement", "FunctionCall", "ExpressionStatement"]
variable_nodes = ["Identifier"]
operator_nodes = ["BinaryOperation"]
extended_library_nodes = ["SafeMath"]

# Function to classify nodes based on the label
def classify_node(label):
    if any(node_type in label for node_type in statement_nodes):
        return 'statement'
    elif any(node_type in label for node_type in variable_nodes):
        return 'variable'
    elif any(node_type in label for node_type in operator_nodes):
        return 'operator'
    elif any(node_type in label for node_type in extended_library_nodes):
        return 'extended_library'
    else:
        return 'other'

# Function to parse the DOT file and create a semantic graph
def create_semantic_graph(dot_content):
    # Initialize graph
    sg = Digraph(strict=True)
    sg.node('Enter')

    # Parse nodes and classify them
    for match in node_pattern.findall(dot_content):
        node_id, label = match
        node_type = classify_node(label)
        sg.node(node_id, label=label, color=node_type_color(node_type))

    # Parse edges
    for src, tgt in edge_pattern.findall(dot_content):
        sg.edge(src, tgt)

    return sg

# Get color for a node type
def node_type_color(node_type):
    return {
        'variable': 'blue',
        'statement': 'orange',
        'operator': 'brown',
        'extended_library': 'green',
        'other': 'black'
    }.get(node_type, 'black')

# Main execution function
def main(input_dot_file_path, output_sg_file_path):
    with open(input_dot_file_path, 'r') as file:
        dot_content = file.read()

    sg = create_semantic_graph(dot_content)
    sg.render(output_sg_file_path, format='dot', cleanup=True)
    print(f'Semantic graph saved to {output_sg_file_path}.dot')

def process_directory(source_directory, output_directory):
    for filename in os.listdir(source_directory):
        if filename.endswith('.sol'):
            try:
                sol_file_path = os.path.join(source_directory, filename)
                json_temp_path = os.path.join(source_directory, f'{os.path.splitext(filename)[0]}.json')
                dot_temp_path = os.path.join(source_directory, f'{os.path.splitext(filename)[0]}.dot')
                sg_file_path = os.path.join(output_directory, os.path.splitext(filename)[0])

                # Convert Solidity to JSON
                convert_sol_to_json(sol_file_path, json_temp_path)

                # Convert JSON to DOT
                ast_json = read_json_ast(json_temp_path)
                json_ast_to_dot(ast_json, dot_temp_path)

                # Read DOT and create semantic graph
                main(dot_temp_path, sg_file_path)

                # Xóa các file trung gian
                os.remove(json_temp_path)
                os.remove(dot_temp_path)

            except Exception as e:
                print(f'Error processing {filename}: {e}')
                continue  # Skip to next file

if __name__ == '__main__':
    source_directory = @"C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\Smart_Contracts_Datasets\6_transaction_order'
    output_directory = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\Smart_Contract_semantic_graph\6_transaction_order'  # Cập nhật đường dẫn thư mục đầu ra
    process_directory(source_directory, output_directory)