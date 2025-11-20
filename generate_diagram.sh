#!/bin/bash
# Simple RTL Visualization using Graphviz DOT language

mkdir -p diagrams

# Create a simple block diagram for execution_unit
cat > diagrams/execution_unit.dot << 'EOF'
digraph execution_unit {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    // Main module
    exec_unit [label="execution_unit\n(Main Controller)", fillcolor=lightgreen, shape=box3d];
    
    // Inputs
    node [shape=invhouse, fillcolor=lightyellow];
    clk [label="clk"];
    rst [label="rst"];
    start [label="start"];
    opcode [label="opcode[4:0]"];
    addr [label="addr"];
    
    // Outputs
    node [shape=house, fillcolor=lightcoral];
    result [label="result"];
    done [label="done"];
    
    // Submodules
    node [shape=box, fillcolor=lightblue];
    load_v [label="load_v\n(Vector Loader)"];
    load_m [label="load_m\n(Matrix Loader)"];
    vector_buf [label="vector_buffer_file"];
    matrix_buf [label="matrix_buffer_file"];
    gemv [label="top_gemv\n(GEMV Engine)"];
    relu_mod [label="relu\n(ReLU Activation)"];
    
    // Connect inputs to main
    clk -> exec_unit;
    rst -> exec_unit;
    start -> exec_unit;
    opcode -> exec_unit;
    addr -> exec_unit;
    
    // Connect main to outputs
    exec_unit -> result;
    exec_unit -> done;
    
    // Connect to submodules
    exec_unit -> load_v [label="control", style=dashed];
    exec_unit -> load_m [label="control", style=dashed];
    exec_unit -> vector_buf [label="control", style=dashed];
    exec_unit -> matrix_buf [label="control", style=dashed];
    exec_unit -> gemv [label="control", style=dashed];
    exec_unit -> relu_mod [label="control", style=dashed];
    
    // Data paths
    load_v -> vector_buf [label="data"];
    load_m -> matrix_buf [label="data"];
    vector_buf -> gemv [label="x, bias"];
    matrix_buf -> gemv [label="weights"];
    gemv -> relu_mod [label="y"];
    relu_mod -> exec_unit [label="activated"];
}
EOF

# Create FSM state diagram
cat > diagrams/execution_fsm.dot << 'EOF'
digraph execution_fsm {
    rankdir=TB;
    node [shape=circle, style=filled, fillcolor=lightblue];
    
    IDLE [fillcolor=lightgreen];
    LOAD_VECTOR;
    LOAD_MATRIX;
    EXECUTE_GEMV;
    GEMV_READ_X;
    GEMV_READ_X_TILES;
    GEMV_READ_BIAS;
    GEMV_READ_BIAS_TILES;
    GEMV_COMPUTE;
    EXECUTE_RELU;
    STORE_RESULT;
    COMPLETE [fillcolor=lightcoral];
    
    IDLE -> LOAD_VECTOR [label="opcode=LOAD_V"];
    IDLE -> LOAD_MATRIX [label="opcode=LOAD_M"];
    IDLE -> EXECUTE_GEMV [label="opcode=GEMV"];
    IDLE -> EXECUTE_RELU [label="opcode=RELU"];
    
    LOAD_VECTOR -> COMPLETE [label="load_done"];
    LOAD_MATRIX -> COMPLETE [label="load_done"];
    
    EXECUTE_GEMV -> GEMV_READ_X;
    GEMV_READ_X -> GEMV_READ_X_TILES;
    GEMV_READ_X_TILES -> GEMV_READ_X_TILES [label="more tiles"];
    GEMV_READ_X_TILES -> GEMV_READ_BIAS [label="tiles done"];
    
    GEMV_READ_BIAS -> GEMV_READ_BIAS_TILES;
    GEMV_READ_BIAS_TILES -> GEMV_READ_BIAS_TILES [label="more tiles"];
    GEMV_READ_BIAS_TILES -> GEMV_COMPUTE [label="tiles done"];
    
    GEMV_COMPUTE -> COMPLETE [label="gemv_done"];
    
    EXECUTE_RELU -> COMPLETE [label="relu_done"];
    
    COMPLETE -> IDLE;
}
EOF

# Create design hierarchy
cat > diagrams/design_hierarchy.dot << 'EOF'
digraph hierarchy {
    rankdir=TB;
    node [shape=box, style="rounded,filled"];
    
    exec [label="execution_unit", fillcolor=lightgreen];
    
    load_v [label="load_v", fillcolor=lightblue];
    load_m [label="load_m", fillcolor=lightblue];
    vec_buf [label="vector_buffer_file", fillcolor=lightyellow];
    mat_buf [label="matrix_buffer_file", fillcolor=lightyellow];
    gemv [label="top_gemv", fillcolor=lightcoral];
    relu [label="relu", fillcolor=lightcoral];
    
    exec -> load_v [label="load_v_inst"];
    exec -> load_m [label="load_m_inst"];
    exec -> vec_buf [label="vector_buffer_inst"];
    exec -> mat_buf [label="matrix_buffer_inst"];
    exec -> gemv [label="gemv_unit"];
    exec -> relu [label="relu_unit"];
}
EOF

echo "Generating SVG diagrams..."
dot -Tsvg diagrams/execution_unit.dot -o diagrams/execution_unit_block.svg
dot -Tpng diagrams/execution_unit.dot -o diagrams/execution_unit_block.png

dot -Tsvg diagrams/execution_fsm.dot -o diagrams/execution_unit_fsm.svg
dot -Tpng diagrams/execution_fsm.dot -o diagrams/execution_unit_fsm.png

dot -Tsvg diagrams/design_hierarchy.dot -o diagrams/design_hierarchy.svg
dot -Tpng diagrams/design_hierarchy.dot -o diagrams/design_hierarchy.png

echo "Done! Generated diagrams:"
ls -lh diagrams/*.{svg,png} 2>/dev/null | tail -n +2
