import subprocess

print("Hello, pick the test you want to run:")
print("1. Instruction Decoder Test")
print("2. GEMV Test")

choice = input("Enter your choice (1-2): ")
if choice not in {'1', '2'}:
    print("Invalid choice. Exiting.")
    exit(1)
# Run the selected test based on user input
commands = {
    '1': ["verilator -Wall --cc rtl/i_decoder.sv --exe test/i_decoder_tb.cpp", 
          "make -C obj_dir -f Vi_decoder.mk Vi_decoder",
          "./obj_dir/Vi_decoder"],
    '2': ["verilator -Wall --trace -cc rtl/gemv.sv rtl/pe.sv --top gemv --exe test/gemv_tb.cpp",
          "make -C obj_dir -f Vgemv.mk Vgemv",
          "./obj_dir/Vgemv"]
}

for cmd in commands[choice]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd}")
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
    print("-" * 40)