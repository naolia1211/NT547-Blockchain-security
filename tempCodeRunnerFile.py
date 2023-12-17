import json

# Function to recursively traverse the AST and create DOT nodes and edges
def traverse_ast(node, graph_components, parent=None):
    if isinstance(node, dict):
        node_type = node.get('type')
        node_name = node.get('name')

        # Create a node declaration
        if node_name:
            graph_components['nodes'].add(f'"{node_name}" [label="{node_name}"]')
            if parent:
                graph_components['edges'].add(f'"{parent}" -> "{node_name}"')

        # Recurse through children or properties of the node
        for key, value in node.items():
            if isinstance(value, list) or isinstance(value, dict):
                traverse_ast(value, graph_components, parent=node_name if node_name else parent)
    elif isinstance(node, list):
        for item in node:
            traverse_ast(item, graph_components, parent)

# Read JSON AST from file
def read_json_ast(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Convert JSON AST to DOT format
def json_ast_to_dot(ast, output_file):
    graph_components = {'nodes': set(), 'edges': set()}
    traverse_ast(ast, graph_components)

    # Write the graph components to the DOT format file
    with open(output_file, 'w') as file:
        file.write('strict digraph {\n')
        for node in graph_components['nodes']:
            file.write(f'    {node};\n')
        for edge in graph_components['edges']:
            file.write(f'    {edge};\n')
        file.write('}\n')

# File paths
input_json_file = 'a.json'  # Path to your input JSON file
output_dot_file = 'output_graph.dot'  # Path to your output DOT file

# Process the JSON AST and output to DOT format
ast_json = read_json_ast(input_json_file)
json_ast_to_dot(ast_json, output_dot_file)
print(f'DOT graph saved to {output_dot_file}')
