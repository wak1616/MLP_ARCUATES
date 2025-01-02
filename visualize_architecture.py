from graphviz import Digraph
import os

def create_network_visualization():
    print("Starting visualization creation...")
    
    dot = Digraph(comment='SimpleMonotonicNN Architecture')
    dot.attr(rankdir='LR')
    
    # Define color scheme
    colors = {
        'input': {'fillcolor': 'lightgrey', 'color': 'black'},
        'dense': {'fillcolor': '#E6F3FF', 'color': '#2B7CE9'},  # Light blue
        'activation': {'fillcolor': '#FFE6E6', 'color': '#D62728'},  # Light red
        'arithmetic': {'fillcolor': '#E6FFE6', 'color': '#2CA02C'},  # Light green
        'output': {'fillcolor': '#FFFDE6', 'color': '#FF7F0E'},  # Light yellow
        'special': {'fillcolor': 'lightyellow', 'color': 'red'}  # For ideal_tx_astig
    }
    
    print("Creating input layer...")
    # Input Layer
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer')
        c.attr(style='filled', color=colors['input']['color'])
        c.node('i1', 'Age\n(1 unit)', style='filled', fillcolor=colors['input']['fillcolor'])
        c.node('i2', 'Steep_axis_term\n(1 unit)', style='filled', fillcolor=colors['input']['fillcolor'])
        c.node('i3', 'Type\n(1 unit)', style='filled', fillcolor=colors['input']['fillcolor'])
        c.node('i4', 'Residual_Astig\n(1 unit)', style='filled', fillcolor=colors['input']['fillcolor'])
        c.node('i5', 'ideal_tx_astig\n(1 unit)', style='filled', 
               fillcolor=colors['special']['fillcolor'], color=colors['special']['color'])
    
    print("Creating paths...")
    # Unconstrained Path
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Regular Path\n(Unconstrained Path)')
        c.node('h1_dense', 'Dense Layer\n16 units', 
               style='filled', fillcolor=colors['dense']['fillcolor'], color=colors['dense']['color'])
        c.node('h1_relu', 'ReLU Activation\n16 units', 
               style='filled', fillcolor=colors['activation']['fillcolor'], color=colors['activation']['color'])
        c.node('h1_out', 'Dense Layer\n1 unit', 
               style='filled', fillcolor=colors['dense']['fillcolor'], color=colors['dense']['color'])
    
    # Constrained Path
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Constrained, Monotonic Path')
        c.node('weight', 'Learnable Weight\n(1 unit)', 
               style='filled', fillcolor=colors['dense']['fillcolor'], color=colors['dense']['color'])
        c.node('softplus', 'Softplus Activation\n(ensures positive weight)', 
               style='filled', fillcolor=colors['activation']['fillcolor'], color=colors['activation']['color'])
        c.node('mult', 'Multiplication\n(1 unit)', 
               style='filled', fillcolor=colors['arithmetic']['fillcolor'], color=colors['arithmetic']['color'])
    
    print("Creating output layer...")
    # Output Combination
    dot.node('plus', 'Addition\n(1 unit)', 
            style='filled', fillcolor=colors['arithmetic']['fillcolor'], color=colors['arithmetic']['color'])
    dot.node('final', 'Neural Network Output\n(1 unit)', 
            style='filled', fillcolor=colors['output']['fillcolor'], color=colors['output']['color'])
    dot.node('pred', 'Final Prediction\n(recommend arcuate incision if\nastigmatism meets minimum threshold)\n(1 unit)', 
            style='filled, dashed', fillcolor=colors['output']['fillcolor'], color=colors['output']['color'])
    
    print("Adding connections...")
    # Regular path connections
    for i in range(1, 6):
        dot.edge(f'i{i}', 'h1_dense', 'Dense (5→16)', color=colors['dense']['color'])
    dot.edge('h1_dense', 'h1_relu', 'Linear→ReLU', color=colors['activation']['color'])
    dot.edge('h1_relu', 'h1_out', 'Dense (16→1)', color=colors['dense']['color'])
    dot.edge('h1_out', 'plus', 'Linear output', color=colors['dense']['color'])
    
    # Monotonic path connections
    dot.edge('weight', 'softplus', 'Activation', color=colors['activation']['color'])
    dot.edge('softplus', 'mult', 'Weight', color=colors['arithmetic']['color'])
    dot.edge('i5', 'mult', 'Input', color=colors['special']['color'])
    dot.edge('mult', 'plus', 'Weighted input', color=colors['arithmetic']['color'])
    
    # Final connections
    dot.edge('plus', 'final', 'Sum', color=colors['output']['color'])
    dot.edge('final', 'pred', 'Apply threshold\n(0.25D)', color=colors['output']['color'])
    
    # Add legend on the right
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend')
        c.attr(style='filled')
        c.attr(rank='same')  # Keep legend items at same level
        c.attr(rankdir='TB')  # Top to bottom arrangement for legend items
        
        # Position legend to the right
        c.attr(labeljust='r')  # Right-justify the label
        c.attr(labelloc='t')   # Place label at top
        
        # Add some invisible edges to force legend to the right
        dot.node('invisible', style='invis')
        dot.edge('final', 'invisible', style='invis')
        dot.edge('invisible', 'note1', style='invis')
        
        c.node('note1', 'Color Coding:\n' +
               '• Blue: Dense/Linear layers\n' +
               '• Red: Activation functions\n' +
               '• Green: Arithmetic operations\n' +
               '• Yellow: Output layers\n' +
               '• Grey: Input features\n' +
               '• Red outline: Special input\n(ideal_tx_astig)', 
               shape='note')
        c.node('note2', 'All activations are ReLU\nexcept where noted', shape='note')
        c.node('note3', 'Numbers in parentheses\nshow tensor dimensions', shape='note')
    
    print("Saving visualization...")
    output_file = 'network_architecture'
    dot.render(output_file, format='png', cleanup=True)
    
    if os.path.exists(f'{output_file}.png'):
        print(f"Visualization saved as {output_file}.png")
        print(f"Full path: {os.path.abspath(f'{output_file}.png')}")
    else:
        print("Error: Visualization file was not created")

if __name__ == "__main__":
    print("Script started")
    create_network_visualization()
    print("Script finished") 