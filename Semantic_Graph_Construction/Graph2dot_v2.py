import os
from solidity_parser import parser
import json
from graphviz import Digraph
import re


def convert_sol_to_json(sol_file_path, json_file_path):
    # Đọc và phân tích file Solidity
    with open(sol_file_path, 'r') as file:
        source_code = file.read()
    parsed_data = parser.parse(source_code)

    # Chuyển đổi sang định dạng JSON
    json_data = json.dumps(parsed_data, indent=4)

    # Lưu kết quả ra file JSON
    with open(json_file_path, 'w') as file:
        file.write(json_data)

#Đường dẫn file Solidity và file JSON đầu ra
sol_file_path = 'BID.sol'  # Thay thế với đường dẫn file .sol của bạn
json_file_path = 'a.json'  # Thay thế với đường dẫn bạn muốn lưu file .json

#Chạy hàm chuyển đổi
convert_sol_to_json(sol_file_path, json_file_path)

import json

# Function to generate a unique identifier for each node
def generate_unique_id(node, existing_ids):
    base_id = f"{node.get('type', 'Node')}_{node.get('name', 'Unnamed')}"
    unique_id = base_id
    counter = 0
    # If the id already exists, append a counter to it to make it unique
    while unique_id in existing_ids:
        counter += 1
        unique_id = f"{base_id}_{counter}"
    return unique_id

# Function to recursively traverse the AST and create DOT nodes and edges
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

# Read JSON AST from file
def read_json_ast(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

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

# File paths
input_json_file = 'a.json'  # Update this path to your JSON file
output_dot_file = 'output_graph.dot'  # Update this path to your output DOT file

# Process the JSON AST and output to DOT format
ast_json = read_json_ast(input_json_file)
json_ast_to_dot(ast_json, output_dot_file)
print(f'DOT graph saved to {output_dot_file}')


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
    # Read DOT content from file
    with open(input_dot_file_path, 'r') as file:
        dot_content = file.read()

    # Create semantic graph
    sg = create_semantic_graph(dot_content)

    # Save semantic graph to file
    sg.render(output_sg_file_path, format='dot', cleanup=True)
    print(f'Semantic graph saved to {output_sg_file_path}')

def convert_directory_sol_to_json(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.sol'):
            sol_file_path = os.path.join(source_dir, filename)
            # Create corresponding file paths
            base_name = os.path.splitext(filename)[0]
            json_file_path = os.path.join(target_dir, f"{base_name}.json")
            dot_file_path = os.path.join(target_dir, f"{base_name}.dot")
            sg_file_path = os.path.join(target_dir, f"{base_name}_sg")

            # Convert Solidity to JSON
            convert_sol_to_json(sol_file_path, json_file_path)

            # Read JSON AST and convert to DOT format
            ast_json = read_json_ast(json_file_path)
            json_ast_to_dot(ast_json, dot_file_path)

            # Read DOT content and create semantic graph
            with open(dot_file_path, 'r') as file:
                dot_content = file.read()
            sg = create_semantic_graph(dot_content)
            sg.render(sg_file_path, format='dot', cleanup=True)

            print(f"Processed {filename}.")

# Directories
source_dir = 'C:\\Users\\hao30\\Documents\\GitHub\\NT547-Blockchain-security\\Smart_Contracts\\vuln'
target_dir = 'C:\\Users\\hao30\\Documents\\GitHub\\NT547-Blockchain-security\\dataset_vuln'

# Convert all Solidity files in the source directory
convert_directory_sol_to_json(source_dir, target_dir)
# Execute the script
if __name__ == '__main__':
    input_dot_file_path = 'output_graph.dot'  # Update with the actual path
    output_sg_file_path = 'sg'        # Update with the desired path (without extension)
    main(input_dot_file_path, output_sg_file_path)
