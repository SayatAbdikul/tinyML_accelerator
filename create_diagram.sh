#!/bin/bash
# Interactive diagram creator for RTL modules

echo "=== RTL Diagram Generator ==="
echo ""
echo "Available diagram types:"
echo "1. Block diagram (module structure)"
echo "2. FSM diagram (state machine)"
echo "3. Datapath diagram (data flow)"
echo "4. Timing diagram placeholder"
echo ""
read -p "Select diagram type (1-4): " choice

case $choice in
  1)
    read -p "Enter module name: " module_name
    cat > "diagrams/${module_name}_custom.dot" << EOF
digraph ${module_name} {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    // Main module
    ${module_name} [label="${module_name}", fillcolor=lightgreen, shape=box3d];
    
    // Add your inputs, outputs, and connections here
    // Example:
    // input1 [shape=invhouse, fillcolor=lightyellow];
    // output1 [shape=house, fillcolor=lightcoral];
    // input1 -> ${module_name} -> output1;
}
EOF
    echo "Created diagrams/${module_name}_custom.dot"
    echo "Edit the file and run: dot -Tsvg diagrams/${module_name}_custom.dot -o diagrams/${module_name}_custom.svg"
    ;;
    
  2)
    read -p "Enter FSM name: " fsm_name
    cat > "diagrams/${fsm_name}_fsm.dot" << EOF
digraph ${fsm_name}_fsm {
    rankdir=TB;
    node [shape=circle, style=filled, fillcolor=lightblue];
    
    IDLE [fillcolor=lightgreen];
    
    // Add your states and transitions here
    // Example:
    // STATE1;
    // STATE2;
    // IDLE -> STATE1 [label="condition"];
    // STATE1 -> STATE2 [label="condition"];
    // STATE2 -> IDLE;
}
EOF
    echo "Created diagrams/${fsm_name}_fsm.dot"
    echo "Edit the file and run: dot -Tsvg diagrams/${fsm_name}_fsm.dot -o diagrams/${fsm_name}_fsm.svg"
    ;;
    
  3)
    read -p "Enter datapath name: " dp_name
    cat > "diagrams/${dp_name}_datapath.dot" << EOF
digraph ${dp_name}_datapath {
    rankdir=LR;
    node [shape=box, style="rounded,filled"];
    
    // Data sources
    node [fillcolor=lightyellow];
    source1 [label="Input Data"];
    
    // Processing blocks
    node [fillcolor=lightblue];
    proc1 [label="Processing Block"];
    
    // Outputs
    node [fillcolor=lightcoral];
    output1 [label="Output"];
    
    // Connections
    source1 -> proc1 -> output1;
}
EOF
    echo "Created diagrams/${dp_name}_datapath.dot"
    echo "Edit the file and run: dot -Tsvg diagrams/${dp_name}_datapath.dot -o diagrams/${dp_name}_datapath.svg"
    ;;
    
  4)
    echo "For timing diagrams, use WaveDrom:"
    echo "1. Install: npm install -g wavedrom-cli"
    echo "2. Create a JSON file with your timing"
    echo "3. Run: wavedrom-cli -i timing.json -s timing.svg"
    echo ""
    echo "Example JSON:"
    cat << 'EOF'
{
  "signal": [
    {"name": "clk",  "wave": "p........"},
    {"name": "data", "wave": "x.345x...", "data": ["A", "B", "C"]},
    {"name": "valid", "wave": "01..0...."}
  ]
}
EOF
    ;;
    
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac
