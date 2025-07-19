import subprocess

print("Hello, pick the test you want to run:")
print("1. Instruction Decoder Test")
print("2. GEMV Test")
print("3. Scale Calculator Test")
print("4. Quantizer Pipeline Test")

choice = input("Enter your choice (1-4): ")
if choice not in {'1', '2', '3', '4'}:
    print("Invalid choice. Exiting.")
    exit(1)
# Run the selected test based on user input
commands = {
    '1': ["verilator -Wall --cc rtl/i_decoder.sv --exe test/i_decoder_tb.cpp", 
          "make -C obj_dir -f Vi_decoder.mk Vi_decoder",
          "./obj_dir/Vi_decoder"],
    '2': ["verilator -Wall --trace -cc rtl/gemv.sv rtl/pe.sv --top gemv --exe test/gemv_tb.cpp",
          "make -C obj_dir -f Vgemv.mk Vgemv",
          "./obj_dir/Vgemv"],
    '3': ["verilator -Wall --cc rtl/scale_calculator.sv --exe test/scale_calculator_tb.cpp",
          "make -C obj_dir -f Vscale_calculator.mk Vscale_calculator",
          "./obj_dir/Vscale_calculator"],
    '4': ["verilator -Wall --cc rtl/quantizer_pipeline.sv --exe test/quantizer_pipeline_tb.cpp",
          "make -C obj_dir -f Vquantizer_pipeline.mk Vquantizer_pipeline",
          "./obj_dir/Vquantizer_pipeline"]
}

for cmd in commands[choice]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd}")
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
    print("-" * 40)