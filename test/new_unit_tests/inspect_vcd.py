import sys
import re

def parse_vcd(filename):
    print(f"Parsing {filename}...")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("VCD file not found.")
        return

    sym_res_wre = None
    sym_res_wad = None
    sym_res_din = None
    sym_clk = None
    sym_state = None
    sym_row_idx = None
    sym_bias_idx = None

    # Parse header
    # Parse header
    idx = 0
    in_core = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('$scope module gemv_unit $end'):
            in_core = True
        elif line.startswith('$upscope $end'):
            in_core = False
        elif line.startswith('$var') and in_core:
            parts = line.split()
            size = int(parts[2])
            sym = parts[3]
            name = parts[4]
            if name == 'res_wre': sym_res_wre = sym
            elif name == 'res_wad': sym_res_wad = sym
            elif name == 'res_din': sym_res_din = sym
            elif name == 'clk': sym_clk = sym
            elif name == 'state': sym_state = sym
            elif name == 'row_idx': sym_row_idx = sym
            elif name == 'bias_store_idx': sym_bias_idx = sym
        elif line.startswith('$enddefinitions'):
            idx = i + 1
            break

    # Tracking state
    wre = 0
    wad = -1
    din = 0
    state = -1
    row_idx = -1
    bias_idx = -1
    clk = 0

    time = 0

    print("--- Searching changes ---")
    for line in lines[idx:]:
        line = line.strip()
        if not line: continue

        if line.startswith('#'):
            time = int(line[1:])
            continue

        val = None
        sym = None
        if line.startswith('b'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    val = int(parts[0][1:], 2)
                except ValueError:
                    val = -1
                sym = parts[1]
        else:
            if line[0] in '01xXzZ':
                v = line[0]
                val = int(v) if v in '01' else -1
                sym = line[1:]

        changed = ""
        if sym == sym_res_wre: wre = val; changed="wre"
        elif sym == sym_res_wad: wad = val; changed="wad"
        elif sym == sym_res_din: din = val; changed="din"
        elif sym == sym_state: state = val; changed="state"
        elif sym == sym_row_idx: row_idx = val; changed="row_idx"
        elif sym == sym_bias_idx: bias_idx = val; changed="bias_idx"
        
        if changed and 110 <= time <= 150:
            print(f"Time {time} | variable {changed}={val} | state={state}, row_idx={row_idx}, wad={wad}, wre={wre}, din={din}")

if __name__ == "__main__":
    parse_vcd("gemv_execution.vcd")


