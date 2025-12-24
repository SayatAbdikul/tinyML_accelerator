#!/bin/bash
# Script to generate all diagram PNG files from DOT sources
# Usage: ./generate_diagrams.sh

DIAGRAM_DIR="$(dirname "$0")"
cd "$DIAGRAM_DIR"

echo "Generating RTL Architecture Diagrams..."
echo "========================================"

# Check if Graphviz is installed
if ! command -v dot &> /dev/null; then
    echo "Error: Graphviz 'dot' command not found!"
    echo "Please install Graphviz:"
    echo "  macOS: brew install graphviz"
    echo "  Ubuntu: sudo apt-get install graphviz"
    exit 1
fi

# Generate PNG files from DOT sources
echo "1. Generating system architecture..."
dot -Tpng system_architecture.dot -o system_architecture.png

echo "2. Generating module hierarchy..."
dot -Tpng module_hierarchy.dot -o module_hierarchy.png

echo "3. Generating execution unit diagram..."
dot -Tpng execution_unit.dot -o execution_unit.png

echo "4. Generating GEMV pipeline..."
dot -Tpng gemv_pipeline.dot -o gemv_pipeline.png

echo "5. Generating memory system..."
dot -Tpng memory_system.dot -o memory_system.png

echo "6. Generating FSM states..."
dot -Tpng fsm_states.dot -o fsm_states.png

echo ""
echo "Done! Generated diagrams:"
echo "  - system_architecture.png"
echo "  - module_hierarchy.png"
echo "  - execution_unit.png"
echo "  - gemv_pipeline.png"
echo "  - memory_system.png"
echo "  - fsm_states.png"
echo ""
echo "You can also generate SVG or PDF:"
echo "  dot -Tsvg system_architecture.dot -o system_architecture.svg"
echo "  dot -Tpdf system_architecture.dot -o system_architecture.pdf"
