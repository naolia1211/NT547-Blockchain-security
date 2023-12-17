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
