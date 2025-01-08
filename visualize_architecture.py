from graphviz import Digraph

def visualize_network():
    dot = Digraph(comment='Monotonic Neural Network')
    dot.attr(rankdir='LR')
    
    # Add input nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(color='lightgrey', label='Input Features', penwidth='2.0')
        c.node('x1', 'Age')
        c.node('x2', 'Steep_axis_term')
        c.node('x3', 'Type')
        c.node('x4', 'MeanK_IOLMaster')
        c.node('x5', 'Treatment_astigmatism')
        c.node('x6', 'WTW_IOLMaster')
        c.node('x7', 'treated_astig')
    
    # Add monotonic transformation nodes
    with dot.subgraph(name='cluster_1') as c:
        c.attr(color='lightblue', label='Monotonic Transformations')
        c.node('m1', 'Constant')
        c.node('m2', 'Linear')
        c.node('m3', 'Quadratic')
        c.node('m4', 'Cubic')
        c.node('m5', 'Quartic')
        c.node('m6', 'Logarithmic')
        c.node('m7', 'Exponential')
        c.node('m8', 'Logistic')
    
    # Add hidden layer as single box
    dot.node('hidden', 'Hidden Layer\n(24 units)\n[ReLU after]\n↓\n(7 units)\n[ReLU after]', 
             shape='box', style='filled', fillcolor='lightgreen', penwidth='2.0')
    
    # Add weight nodes with clarified label
    with dot.subgraph(name='cluster_3') as c:
        c.attr(color='yellow', label='Weights\n(output from hidden layer)', penwidth='2.0')
        for i in range(7):
            c.node(f'w{i}', f'w{i}')
    
    # Add output node with non-negative constraint note
    dot.node('out', 'Arcuate Sweep\n(max(0, prediction))', 
             shape='doubleoctagon', penwidth='2.0')
    
    # Add edges with colors for monotonic transformations
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'darkgreen']
    
    # Connect input features to hidden layer
    for i in range(6):
        dot.edge(f'x{i+1}', 'hidden', penwidth='1.5')
    
    # Connect treated_astig to monotonic transformations
    for i in range(8):
        dot.edge('x7', f'm{i+1}', penwidth='1.5')
    
    # Connect hidden layer to weights
    for i in range(7):
        dot.edge('hidden', f'w{i}', penwidth='1.5')
    
    # Connect monotonic transformations to weights with colors
    for i in range(7):
        dot.edge(f'm{i+1}', f'w{i}', color=colors[i], penwidth='1.5')
    
    # Connect weights to output
    for i in range(7):
        dot.edge(f'w{i}', 'out', color=colors[i], penwidth='1.5')
    
    # Add comprehensive legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', color='black', penwidth='2.0')
        c.node('l1', 'Input Features', shape='box', color='lightgrey', 
               style='filled', penwidth='2.0')
        c.node('l2', 'Monotonic Transforms', shape='box', color='lightblue', 
               style='filled', penwidth='2.0')
        c.node('l3', 'Hidden Layer', shape='box', color='lightgreen', 
               style='filled', penwidth='2.0')
        c.node('l4', 'Weights\n(from hidden layer)', shape='box', color='yellow', 
               style='filled', penwidth='2.0')
        c.node('l5', 'Output', shape='doubleoctagon', penwidth='2.0')
        
        # Updated processing notes
        c.node('note1', 'Processing Notes:\n' +
               '1. ReLU activation after:\n' +
               '   - First hidden layer (24 units)\n' +
               '   - Second hidden layer (7 units)\n' +
               '2. Features are standardized before input\n' +
               '3. Hidden layer outputs weights for monotonic terms\n' +
               '4. Monotonic terms are multiplied by their weights\n' +
               '5. Final prediction ensures non-negative output', 
               shape='note', penwidth='2.0')
        
        # Arrange legend vertically
        c.edge('l1', 'l2', style='invis')
        c.edge('l2', 'l3', style='invis')
        c.edge('l3', 'l4', style='invis')
        c.edge('l4', 'l5', style='invis')
        c.edge('l5', 'note1', style='invis')
    
    # Save the visualization
    dot.render('network_architecture', format='png', cleanup=True)

if __name__ == '__main__':
    visualize_network()

