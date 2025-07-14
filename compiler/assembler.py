import re

# === Define opcode mapping ===
OPCODES = {
    "LOAD_V": 0x01,
    "LOAD_M": 0x02,
    "STORE":  0x03,
    "GEMV":   0x04,
    "RELU":   0x05,
    "NOP":    0x00
}

def assemble_line(line):
    parts = re.split(r'[,\s]+', line.strip())
    if not parts or parts[0].startswith(";") or parts[0] == "":
        return None  # comment or empty line

    instr = parts[0]
    opcode = OPCODES.get(instr)
    if opcode is None:
        raise ValueError(f"Unknown instruction: {instr}")

    if instr == "NOP":
        return f"{opcode:016X}000000"

    # Basic format: OPCODE, ARG1, ARG2, ...
    args = [int(x, 0) for x in parts[1:]]  # auto handles hex/dec

    if instr == "LOAD_V" or instr == "STORE":
        dest, addr, length = args
        word = (addr << 40) | (length << 10)  | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "LOAD_M":
        dest, addr, rows, cols = args
        # Simplified: rows << 8 | cols (use carefully!)
        word = (addr << 40) | (rows << 20) | (cols << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "GEMV":
        dest, w, x, b, rows, cols = args
        # Example encoding: upper half = op/dest/w/x, lower = rows/cols
        word = (w << 40) | (x << 35) | (b << 30) | (rows << 20) | (cols << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "RELU":
        dest, x = args
        word = (x << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    else:
        raise NotImplementedError(f"Unsupported instruction: {instr}")

def assemble_file(asm_file, out_file="program.hex"):
    with open(asm_file) as f:
        lines = f.readlines()

    machine_code = []
    for line in lines:
        encoded = assemble_line(line)
        # print(f"Encoding line: '{line.strip()}' -> '{encoded}'")
        if encoded:
            machine_code.extend(encoded.strip().split("\n"))

    with open(out_file, "w") as f:
        for code in machine_code:
            f.write(code + "\n")

    # print(f"✅ Assembled {len(machine_code)} instructions to '{out_file}'")

if __name__ == "__main__":
    # Example usage
    asm_file = "model_assembly.asm"
    out_file = "program.hex"
    assemble_file(asm_file, out_file)
