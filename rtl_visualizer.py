#!/usr/bin/env python3
"""
RTL Diagram Generator using Graphviz
Analyzes SystemVerilog files and generates block diagrams
"""

import re
import graphviz
from pathlib import Path

def parse_module(sv_file):
    """Parse SystemVerilog module to extract ports and hierarchy"""
    with open(sv_file, 'r') as f:
        content = f.read()
    
    # Extract module name
    module_match = re.search(r'module\s+(\w+)', content)
    if not module_match:
        return None
    
    module_name = module_match.group(1)
    
    # Extract inputs/outputs
    inputs = re.findall(r'input\s+(?:logic\s+)?(?:\[[^\]]+\]\s+)?(\w+)', content)
    outputs = re.findall(r'output\s+(?:logic\s+)?(?:\[[^\]]+\]\s+)?(\w+)', content)
    
    # Extract instantiated submodules
    instances = re.findall(r'(\w+)\s+#?\s*\([^)]*\)\s+(\w+)\s*\(', content)
    
    return {
        'name': module_name,
        'inputs': inputs,
        'outputs': outputs,
        'instances': instances
    }

def create_block_diagram(module_info, output_file='rtl_diagram'):
    """Create a block diagram using Graphviz"""
    dot = graphviz.Digraph(comment=f"{module_info['name']} RTL Block Diagram")
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Main module
    main_label = f"{module_info['name']}"
    dot.node(module_info['name'], main_label, fillcolor='lightgreen', shape='box3d')
    
    # Input ports
    dot.attr('node', shape='circle', fillcolor='lightyellow')
    for inp in module_info['inputs'][:10]:  # Limit display
        dot.node(f"in_{inp}", inp, shape='invhouse')
        dot.edge(f"in_{inp}", module_info['name'])
    
    # Output ports
    dot.attr('node', fillcolor='lightcoral')
    for out in module_info['outputs'][:10]:  # Limit display
        dot.node(f"out_{out}", out, shape='house')
        dot.edge(module_info['name'], f"out_{out}")
    
    # Submodules
    dot.attr('node', shape='box', fillcolor='lightblue')
    for mod_type, inst_name in module_info['instances']:
        dot.node(inst_name, f"{inst_name}\n({mod_type})")
        dot.edge(module_info['name'], inst_name, style='dashed', label='contains')
    
    # Render
    dot.render(output_file, format='svg', cleanup=True)
    dot.render(output_file, format='png', cleanup=True)
    print(f"Generated {output_file}.svg and {output_file}.png")
    return dot

def create_fsm_diagram(sv_file, output_file='fsm_diagram'):
    """Extract and visualize FSM states"""
    with open(sv_file, 'r') as f:
        content = f.read()
    
    # Find FSM states
    enum_match = re.search(r'typedef\s+enum[^{]*{([^}]+)}', content)
    if not enum_match:
        print("No FSM found")
        return
    
    states = [s.strip().rstrip(',') for s in enum_match.group(1).split('\n') if s.strip()]
    
    # Find state transitions
    transitions = re.findall(r'state\s*<=\s*(\w+)', content)
    
    dot = graphviz.Digraph(comment='FSM State Diagram')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
    
    # Add states
    for state in states:
        if state:
            color = 'lightgreen' if state == 'IDLE' else 'lightblue'
            dot.node(state, state, fillcolor=color)
    
    # Add transitions (simplified)
    seen_transitions = set()
    for i, trans in enumerate(transitions):
        if i > 0 and trans:
            prev_state = transitions[i-1] if i > 0 else 'IDLE'
            edge = (prev_state, trans)
            if edge not in seen_transitions:
                dot.edge(prev_state, trans)
                seen_transitions.add(edge)
    
    dot.render(output_file, format='svg', cleanup=True)
    dot.render(output_file, format='png', cleanup=True)
    print(f"Generated {output_file}.svg and {output_file}.png")
    return dot

def create_hierarchy_diagram(rtl_dir, top_module, output_file='hierarchy'):
    """Create full design hierarchy"""
    dot = graphviz.Digraph(comment='Design Hierarchy')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled')
    
    modules = {}
    rtl_path = Path(rtl_dir)
    
    # Parse all modules
    for sv_file in rtl_path.glob('*.sv'):
        info = parse_module(sv_file)
        if info:
            modules[info['name']] = info
    
    # Build hierarchy
    def add_hierarchy(mod_name, level=0):
        if mod_name not in modules:
            return
        
        color = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral'][level % 4]
        dot.node(mod_name, mod_name, fillcolor=color)
        
        for sub_type, sub_inst in modules[mod_name]['instances']:
            if sub_type in modules:
                dot.edge(mod_name, sub_type, label=sub_inst)
                add_hierarchy(sub_type, level + 1)
    
    if top_module in modules:
        add_hierarchy(top_module)
        dot.render(output_file, format='svg', cleanup=True)
        dot.render(output_file, format='png', cleanup=True)
        print(f"Generated {output_file}.svg and {output_file}.png")
    else:
        print(f"Top module {top_module} not found")
    
    return dot

if __name__ == '__main__':
    import os
    # Example usage
    rtl_dir = 'rtl'
    
    # Create diagrams directory
    os.makedirs('diagrams', exist_ok=True)
    
    print("=== Generating RTL Diagrams ===\n")
    
    # 1. Block diagram for execution unit
    print("1. Creating execution_unit block diagram...")
    try:
        exec_info = parse_module(f'{rtl_dir}/execution_unit.sv')
        if exec_info:
            create_block_diagram(exec_info, 'diagrams/execution_unit_block')
        else:
            print("   Could not parse module")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. FSM diagram
    print("\n2. Creating FSM state diagram...")
    try:
        create_fsm_diagram(f'{rtl_dir}/execution_unit.sv', 'diagrams/execution_unit_fsm')
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Design hierarchy
    print("\n3. Creating design hierarchy...")
    try:
        create_hierarchy_diagram(rtl_dir, 'execution_unit', 'diagrams/design_hierarchy')
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== Done! ===")
    print("Check the 'diagrams/' directory for generated SVG and PNG files")
