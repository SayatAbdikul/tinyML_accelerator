#!/bin/bash
# Wrapper script to run tests with venv automatically activated
# This ensures the correct Python environment is used

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# Set PYTHONPATH for compiler imports
export PYTHONPATH="$SCRIPT_DIR/../../compiler:$PYTHONPATH"

# Run make with all provided arguments
make "$@"
