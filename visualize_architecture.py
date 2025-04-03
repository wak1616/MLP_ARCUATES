from graphviz import Digraph

def visualize_network():
    dot = Digraph(comment='Arcuate Sweep Predictor Network')
    dot.attr(rankdir='LR')
    
    # Add input features (including treated_astig)
    with dot.subgraph(name='cluster_0') as c:
        c.attr(color='lightgrey', label='''<<B>Input Features</B>>''', penwidth='2.0')
        c.node('x1', 'Age')
        c.node('x2', 'Steep_axis_term')
        c.node('x3', 'MeanK_IOLMaster')
        c.node('x4', 'Residual_Astigmatism')
        c.node('x5', 'WTW_IOLMaster')
        c.node('x6', 'treated_astig')
    
    # Update unconstrained path node to represent the full MLP
    dot.node('mlp_path', '''<<B>MLP Path</B><BR/><BR/>[Linear(input_dim → 48)]<BR/>↓<BR/>[LeakyReLU]<BR/>↓<BR/>[Linear(48 → 10)]<BR/>↓<BR/>[ReLU]<BR/>↓<BR/>[Linear(10 → 1)]<BR/><BR/>(predicts Arcuate Sweep)>''', 
             shape='box', style='filled', fillcolor='lightgreen', penwidth='2.0')
    
    # Update output node description
    dot.node('out', '''<<B>Predicted Arcuate Sweep</B>>''', 
             shape='doubleoctagon', penwidth='2.0')
    
    # Connect ALL inputs to MLP path
    for i in range(6):
        dot.edge(f'x{i+1}', 'mlp_path', penwidth='1.5')
    
    # Connect MLP path to output
    dot.edge('mlp_path', 'out', penwidth='1.5')
    
    # Update processing notes
    dot.node('note1', '''<<B>Processing Notes:</B><BR ALIGN="LEFT"/>1. All input features feed into the MLP Path.<BR ALIGN="LEFT"/>2. The MLP Path processes the features through a series<BR ALIGN="LEFT"/>   of linear layers and activation functions.<BR ALIGN="LEFT"/>3. The final linear layer outputs the predicted Arcuate Sweep.<BR ALIGN="LEFT"/>4. Output is constrained to be non-negative in the prediction script (not shown here).>''', 
             shape='box', penwidth='2.0', margin='0.1,0.1')
    
    # Save the visualization
    dot.render('network_architecture', format='png', cleanup=True)

if __name__ == '__main__':
    visualize_network()

