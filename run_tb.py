import subprocess

print("Hello, pick the test you want to run:")
print("1. Instruction Decoder Test")
print("2. GEMV Test")
print("3. Scale Calculator Test")
print("4. Quantizer Pipeline Test")
print("5. Quantizer Overall Test")
print("6. Wallace Multiplier Test")
print("7. GEMV Buffer File Test")
print("8. PE Test")
print("9. Memory Test")
print("10. Buffer File Test")
print("11. Load Matrix Test")
print("12. Top GEMV Test")
print("13. ReLU Test")
print("14. Load Vector Test")

choice = input("Enter your choice (1-14): ")
if choice not in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'}:
    print("Invalid choice. Exiting.")
    exit(1)
# Run the selected test based on user input
commands = {
    '1': ["verilator -Wall --cc rtl/i_decoder.sv --exe test/i_decoder_tb.cpp", 
          "make -C obj_dir -f Vi_decoder.mk Vi_decoder",
          "./obj_dir/Vi_decoder"],
    '2': ["verilator -Wall --trace -cc rtl/gemv.sv rtl/pe.sv rtl/scale_calculator.sv "
      "rtl/quantizer_pipeline.sv rtl/wallace_32x32.sv rtl/compressor_3to2.sv --top gemv --exe test/gemv_tb.cpp",
          "make -C obj_dir -f Vgemv.mk Vgemv",
          "./obj_dir/Vgemv"],
    '3': ["verilator -Wall --cc rtl/scale_calculator.sv --exe test/scale_calculator_tb.cpp",
          "make -C obj_dir -f Vscale_calculator.mk Vscale_calculator",
          "./obj_dir/Vscale_calculator"],
    '4': ["verilator -Wall --cc rtl/quantizer_pipeline.sv --exe test/quantizer_pipeline_tb.cpp",
          "make -C obj_dir -f Vquantizer_pipeline.mk Vquantizer_pipeline",
          "./obj_dir/Vquantizer_pipeline"],
    '5': ["verilator -Wall --cc rtl/scale_calculator.sv rtl/wallace_32x32.sv rtl/compressor_3to2.sv rtl/quantizer_pipeline.sv rtl/quantization.sv --top quantization --exe test/quantization_tb.cpp",
          "make -C obj_dir -f Vquantization.mk Vquantization",
          "./obj_dir/Vquantization"],
      '6': ["verilator -Wall --cc rtl/wallace_32x32.sv rtl/compressor_3to2.sv --top wallace_32x32 --exe test/wallace_32x32_tb.cpp --trace",
          "make -C obj_dir -f Vwallace_32x32.mk Vwallace_32x32",
          "./obj_dir/Vwallace_32x32"],
      '7': ["verilator -Wall --cc rtl/gemv_buffer_file.sv --exe test/gemv_buffer_file_tb.cpp",
          "make -C obj_dir -f Vgemv_buffer_file.mk Vgemv_buffer_file",
          "./obj_dir/Vgemv_buffer_file"],
      '8': ["verilator -Wall --cc rtl/pe.sv --exe test/pe_tb.cpp",
          "make -C obj_dir -f Vpe.mk Vpe",
          "./obj_dir/Vpe"],
      '9': ["verilator -Wall --cc rtl/simple_memory.sv --exe test/simple_memory_tb.cpp",
          "make -C obj_dir -f Vsimple_memory.mk Vsimple_memory",
          "./obj_dir/Vsimple_memory"],
    '10': ["verilator -Wall --trace --cc rtl/buffer_file.sv --exe test/buffer_file_tb.cpp",
            "make -C obj_dir -f Vbuffer_file.mk Vbuffer_file",
            "./obj_dir/Vbuffer_file"],
    '11': ["verilator -Wall --cc rtl/load_m.sv rtl/simple_memory.sv --top load_m --exe test/load_m_tb.cpp",
            "make -C obj_dir -f Vload_m.mk Vload_m",
            "./obj_dir/Vload_m"],
    '12': ["verilator -Wall --trace --cc rtl/top_gemv.sv rtl/pe.sv rtl/scale_calculator.sv rtl/quantizer_pipeline.sv "
            "rtl/wallace_32x32.sv rtl/compressor_3to2.sv --top top_gemv --exe test/top_gemv_tb.cpp",
            "make -C obj_dir -f Vtop_gemv.mk Vtop_gemv",
            "./obj_dir/Vtop_gemv"],
    '13': ["verilator -Wall --cc rtl/relu.sv --exe test/relu_tb.cpp",
            "make -C obj_dir -f Vrelu.mk Vrelu",
            "./obj_dir/Vrelu"],
    '14': ["verilator -Wall --cc rtl/load_v.sv rtl/simple_memory.sv --top load_v --exe test/load_v_tb.cpp",
            "make -C obj_dir -f Vload_v.mk Vload_v",
            "./obj_dir/Vload_v"]
    
}

for cmd in commands[choice]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd}")
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
    print("-" * 40)