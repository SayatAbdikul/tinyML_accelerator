# TinyML Accelerator RTL Documentation - Quick Reference

## What You Got

I've created comprehensive documentation for all RTL modules with multiple levels of abstraction:

### 📚 Documentation Files

1. **`docs/README.md`** - Main documentation index
   - Overview of entire design
   - Quick start guide
   - Module summary
   - File organization

2. **`docs/RTL_ARCHITECTURE.md`** - Complete architecture documentation (700+ lines)
   - Detailed module descriptions
   - Signal flow diagrams (ASCII art)
   - Memory architecture
   - GEMV pipeline
   - Quantization details
   - Performance characteristics

3. **`docs/diagrams/README.md`** - Diagram documentation
   - Usage instructions
   - Diagram descriptions

### 🎨 Visual Diagrams (6 diagrams in both DOT and PNG formats)

1. **`system_architecture`** - Top-level system overview
   - Fetch → Decode → Execute flow
   - 4 memory instances
   - Control signals

2. **`module_hierarchy`** - Complete module tree
   - All 22 modules
   - Parent-child relationships
   - Module sizes (lines of code)

3. **`execution_unit`** - Execution unit details
   - FSM controller
   - Buffer management
   - Operation modules
   - Data flow

4. **`gemv_pipeline`** - GEMV computation
   - 32 PE array
   - Accumulation tree
   - 3-stage quantization
   - Control FSM

5. **`memory_system`** - Memory architecture
   - 4 separate instances
   - Memory map
   - Access patterns
   - Synchronization notes

6. **`fsm_states`** - State machine diagrams
   - Top-level FSM (7 states)
   - Execution FSM (7 states)
   - GEMV FSM (9 states)

### 📊 Abstraction Levels Covered

**Level 1: System Level**
```
Input → [ TinyML Accelerator ] → Output
```

**Level 2: Subsystem Level**
```
Fetch & Decode → Execution Unit → Memory
                      ↓
              Buffers ↔ Compute
```

**Level 3: Module Level**
```
22 modules organized in hierarchy:
- Top: tinyml_accelerator_top
- Control: fetch_unit, i_decoder
- Execution: modular_execution_unit + 5 sub-modules
- Buffers: buffer_controller + 2 buffer_files
- Compute: top_gemv + 32 PEs + quantization
- Memory: 4× simple_memory
- Arithmetic: wallace_32x32, compressor_3to2
```

**Level 4: Component Level**
```
- Processing Elements (PE): 8×8 multipliers
- Quantization: Scale calculator + saturator
- Wallace Tree: 3:2 compressors
- FSMs: State machines
- Buffers: Register files
```

## All 22 RTL Modules Documented

### Top Level (1)
- `tinyml_accelerator_top` - Main coordinator

### Control (2)
- `fetch_unit` - Instruction fetch
- `i_decoder` - Instruction decoder

### Execution (6)
- `modular_execution_unit` - Execution coordinator
- `buffer_controller` - Buffer management
- `load_execution` - Load orchestration
- `gemv_execution` - GEMV orchestration
- `relu_execution` - ReLU orchestration
- `store_execution` - Store orchestration

### Data Movement (3)
- `load_v` - Vector loading
- `load_m` - Matrix loading
- `store` - Memory writing

### Computation (2)
- `top_gemv` - GEMV computation
- `relu` - ReLU activation

### Buffers & Memory (3)
- `buffer_file` - Multi-buffer storage
- `simple_memory` - Memory array

### Processing & Quantization (5)
- `pe` - Processing element (32 instances)
- `quantization` - Quantization unit
- `quantizer_pipeline` - Pipeline stage
- `scale_calculator` - Scale computation
- `wallace_32x32` - 32-bit multiplier
- `compressor_3to2` - Full adder

## How to Use

### View Documentation
```bash
# Main index
open docs/README.md

# Architecture details
open docs/RTL_ARCHITECTURE.md

# View diagrams
open docs/diagrams/system_architecture.png
open docs/diagrams/module_hierarchy.png
open docs/diagrams/gemv_pipeline.png
```

### Generate Diagrams in Different Formats
```bash
cd docs/diagrams

# Regenerate PNG
./generate_diagrams.sh

# Generate SVG (scalable for presentations)
dot -Tsvg system_architecture.dot -o system_architecture.svg

# Generate PDF
dot -Tpdf module_hierarchy.dot -o module_hierarchy.pdf
```

### Search Documentation
```bash
# Find specific module
grep -r "load_v" docs/RTL_ARCHITECTURE.md

# Find signal names
grep -r "exec_done" docs/RTL_ARCHITECTURE.md

# Find FSM states
grep -r "WAIT_GEMV" docs/RTL_ARCHITECTURE.md
```

## Key Features Documented

✅ **Complete Module Hierarchy** - All 22 modules mapped
✅ **Signal Flows** - Data paths through the system
✅ **FSM Diagrams** - All state machines documented
✅ **Memory Architecture** - 4 separate instances explained
✅ **GEMV Pipeline** - Computation flow with 32 PEs
✅ **Quantization** - 32-bit → 8-bit conversion process
✅ **Tiling Strategy** - 32-element tiles for efficiency
✅ **Performance** - Latency and throughput characteristics
✅ **Design Patterns** - Modularity, FSMs, handshakes

## Connection Information

### Module Connections
All documented with:
- Parent-child relationships
- Signal interfaces
- Data flow directions
- Control handshakes (start/done)

### Example: GEMV Execution Chain
```
modular_execution_unit
    ↓ (start/done)
gemv_execution
    ↓ (buffer read/write)
buffer_controller
    ↓ (tile data)
top_gemv
    ↓ (w, x, bias)
32× pe (parallel)
    ↓ (pe_out)
accumulation
    ↓ (res[])
quantization
    ↓ (y[])
output
```

## Files Created

```
docs/
├── README.md                    # Main index (new)
├── RTL_ARCHITECTURE.md          # Complete doc (new)
└── diagrams/                    # New directory
    ├── README.md                # Diagram guide (new)
    ├── generate_diagrams.sh     # Generation script (new)
    ├── system_architecture.dot  # System diagram (new)
    ├── system_architecture.png  # Generated (new)
    ├── module_hierarchy.dot     # Module tree (new)
    ├── module_hierarchy.png     # Generated (new)
    ├── execution_unit.dot       # Execution detail (new)
    ├── execution_unit.png       # Generated (new)
    ├── gemv_pipeline.dot        # GEMV pipeline (new)
    ├── gemv_pipeline.png        # Generated (new)
    ├── memory_system.dot        # Memory arch (new)
    ├── memory_system.png        # Generated (new)
    ├── fsm_states.dot           # FSM diagrams (new)
    └── fsm_states.png           # Generated (new)
```

**Total: 16 new files** (3 markdown, 6 DOT sources, 6 PNG diagrams, 1 script)

## What's Included

### For Each Module:
- ✅ Purpose and functionality
- ✅ Line count
- ✅ Key features
- ✅ Inputs/outputs
- ✅ Internal structure
- ✅ Connections to other modules

### For the System:
- ✅ Overall architecture
- ✅ Instruction flow
- ✅ Data flow
- ✅ Control flow
- ✅ Memory map
- ✅ FSM states
- ✅ Performance metrics
- ✅ Design patterns

## Next Steps

1. **Review the main README**: `docs/README.md`
2. **Study the diagrams**: Open PNG files in `docs/diagrams/`
3. **Read detailed architecture**: `docs/RTL_ARCHITECTURE.md`
4. **Cross-reference with code**: Compare docs with `rtl/*.sv` files

## Benefits

✨ **Complete Coverage** - All modules documented
✨ **Multiple Views** - Text + visual diagrams
✨ **Multiple Levels** - System → Component abstraction
✨ **Searchable** - Markdown format, easy grep
✨ **Visual** - 6 professional diagrams
✨ **Modifiable** - DOT sources for customization
✨ **Scalable** - SVG/PDF generation supported

---

**Summary**: You now have professional-grade documentation covering all 22 RTL modules with:
- 3 markdown documents (~1200 lines total)
- 6 visual diagrams (DOT + PNG)
- 4 abstraction levels
- Complete module hierarchy
- Signal flows and FSM states
- Memory architecture details
- Performance characteristics

Perfect for understanding, maintaining, and presenting your TinyML accelerator design! 🚀
