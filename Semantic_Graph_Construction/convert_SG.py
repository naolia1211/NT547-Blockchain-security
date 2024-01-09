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

# Execute the script
if __name__ == '__main__':
    input_dot_file_path = 'output_graph.dot'  # Update with the actual path
    output_sg_file_path = 'sg.dot'        # Update with the desired path (without extension)
    main(input_dot_file_path, output_sg_file_path)
