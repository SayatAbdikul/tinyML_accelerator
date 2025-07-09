# disassembler.py
def decode_instruction(word):
    instr = int(word, 16)
    opcode = instr & 0x1F  # last 5 bits

    if opcode == 0x00:
        return "NOP"

    elif opcode == 0x01 or opcode == 0x03:  # LOAD_V or STORE
        dest = (instr >> 5) & 0x1F
        length = (instr >> 10) & 0x3FF
        addr = (instr >> 40) & 0xFFFFFF
        if opcode == 0x01:  # LOAD_V
            return f"LOAD_V {dest}, 0x{addr:X}, {length}"
        else:  # STORE
            return f"STORE {dest}, 0x{addr:X}, {length}"

    elif opcode == 0x02:  # LOAD_M
        dest = (instr >> 5) & 0x1F
        cols = (instr >> 10) & 0x3FF
        rows = (instr >> 20) & 0x3FF
        addr = (instr >> 40) & 0xFFFFFF
        return f"LOAD_M {dest}, 0x{addr:X}, {rows}, {cols}"


    elif opcode == 0x04:  # GEMV
        dest = (instr >> 5) & 0x1F
        cols = (instr >> 10) & 0x3FF
        rows = (instr >> 20) & 0x3FF
        b = (instr >> 30) & 0x1F
        x = (instr >> 35) & 0x1F
        w = (instr >> 40) & 0x1F
        return f"GEMV {dest}, {w}, {x}, {b}, {rows}, {cols}"

    elif opcode == 0x05:  # RELU
        dest = (instr >> 5) & 0x1F
        x = (instr >> 10) & 0x1F
        return f"RELU {dest}, {x}"

    else:
        return f"UNKNOWN_OPCODE_{opcode:02X}"


def disassemble_file(hex_file, out_file="disassembled.asm"):
    with open(hex_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(out_file, "w") as out:
        out.write("; Disassembled code\n\n")
        for i, line in enumerate(lines):
            decoded = decode_instruction(line)
            out.write(f"{decoded}\n")
            print(f"{i:02}: {line} -> {decoded}")

    print(f"\n✅ Disassembly complete: {out_file}")


# Example usage
if __name__ == "__main__":
    disassemble_file("program.hex")
