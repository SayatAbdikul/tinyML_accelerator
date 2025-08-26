// Top Module - tinyML Accelerator
// Combines instruction fetch, decode, execution, and memory management
module tinyml_accelerator_top #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 128,
    parameter ADDR_WIDTH = 24,
    parameter MAX_ROWS = 128,
    parameter MAX_COLS = 128,
    parameter OUT_N = 10
)(
    input logic clk,
    input logic rst,
    input logic start,           // Start processing
    output logic signed [DATA_WIDTH-1:0] y [0:OUT_N-1],  // Results
    output logic done            // Processing complete
);

    // ===== Local Parameters =====
    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    
    // ===== Instruction Fetch and Decode =====
    logic fetch_enable, fetch_done;
    logic [63:0] current_instruction;
    logic [ADDR_WIDTH-1:0] pc;
    
    // Decoded instruction fields
    logic [4:0] opcode, dest, b_id, x_id, w_id;
    logic [9:0] length_or_cols, rows;
    logic [23:0] addr;
    
    // ===== Main FSM =====
    typedef enum logic [2:0] {
        IDLE,
        FETCH_INSTRUCTION,
        EXECUTE_INSTRUCTION,
        WAIT_EXECUTION,
        OUTPUT_RESULTS,
        COMPLETE
    } main_state_t;
    
    main_state_t state;
    logic [3:0] instruction_count;
    logic [4:0] last_executed_opcode;
    logic exec_start_delayed;
    
    // ===== Memory and Loading =====
    // NOTE: All loading is now handled by execution_unit internally
    // These signals are kept for potential future use but not connected
    logic weight_tile_valid;
    logic [TILE_WIDTH-1:0] weight_tile_data;
    
    // Unpacked weight tile data for execution unit (currently unused)
    logic signed [DATA_WIDTH-1:0] weight_tile_data_unpacked [0:TILE_ELEMS-1];
    
    // Convert packed weight data to unpacked array (currently unused)
    always_comb begin
        for (int i = 0; i < TILE_ELEMS; i++) begin
            weight_tile_data_unpacked[i] = weight_tile_data[i*DATA_WIDTH +: DATA_WIDTH];
        end
    end
    
    // ===== Buffer Management =====
    // Vector buffers for computation
    logic signed [DATA_WIDTH-1:0] x_buffer [0:MAX_COLS-1];
    logic signed [DATA_WIDTH-1:0] bias_buffer [0:MAX_ROWS-1];
    
    // Weight Buffer signals
    logic weight_buffer_we, weight_buffer_re;
    logic [TILE_WIDTH-1:0] weight_write_data;
    logic weight_write_done, weight_read_done;
    logic [DATA_WIDTH-1:0] weight_read_data [0:31];
    logic [1:0] weight_write_buffer, weight_read_buffer;
    
    // Vector Buffer signals
    logic vector_buffer_we, vector_buffer_re;
    logic [TILE_WIDTH-1:0] vector_write_data;
    logic vector_write_done, vector_read_done;
    logic [DATA_WIDTH-1:0] vector_read_data [0:31];
    logic [1:0] vector_write_buffer, vector_read_buffer;
    
    // ===== Execution Unit =====
    logic exec_start, exec_done;
    logic signed [DATA_WIDTH-1:0] exec_result [0:MAX_ROWS-1];
    
    // ===== Module Instantiations =====
    
    // Instruction Fetch Unit
    fetch_unit #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .INSTR_WIDTH(64),
        .DATA_WIDTH(8)
    ) fetch_unit_inst (
        .clk(clk),
        .rst_n(~rst),
        .fetch_en_i(fetch_enable),
        .pc_o(pc),
        .instr_o(current_instruction),
        .done(fetch_done)
    );
    
    // Instruction Decoder
    i_decoder decoder_inst (
        .instr(current_instruction),
        .opcode(opcode),
        .dest(dest),
        .length_or_cols(length_or_cols),
        .rows(rows),
        .addr(addr),
        .b(b_id),
        .x(x_id),
        .w(w_id)
    );
    
    // NOTE: load_v and load_m modules are handled internally by execution_unit
    // No need for separate loaders at top level - execution unit is self-contained
    
    // Execution Unit
    execution_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS)
    ) exec_unit (
        .clk(clk),
        .rst(rst),
        .start(exec_start_delayed), // Use delayed start signal
        .opcode(opcode),
        .dest(dest),
        .length_or_cols(length_or_cols),
        .rows(rows),
        .addr(addr),
        .b_id(b_id),
        .x_id(x_id),
        .w_id(w_id),
        
        // Data interfaces (for direct access if needed)
        .x_buffer(x_buffer),
        .bias_buffer(bias_buffer),
        
        // Memory interface (not used in this integration)
        /* verilator lint_off PINCONNECTEMPTY */
        .mem_req(),
        .mem_addr(),
        /* verilator lint_on PINCONNECTEMPTY */
        .mem_valid(1'b0),
        .mem_data(8'b0),
        
        // Weight loading interface (execution unit handles loading internally)
        /* verilator lint_off PINCONNECTEMPTY */
        .weight_load_start(),
        .weight_load_addr(),
        .weight_load_length(),
        /* verilator lint_on PINCONNECTEMPTY */
        .weight_tile_valid(1'b0),        // Not used - execution unit loads internally
        .weight_tile_data('{default:0}), // Not used - execution unit loads internally
        
        // Results
        .result(exec_result),
        .done(exec_done)
    );
    
    // Weight Buffer File for matrix storage
    buffer_file #(
        .BUFFER_WIDTH(1024),
        .BUFFER_COUNT(4),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(32)
    ) weight_buffer (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(weight_buffer_we),
        .read_enable(weight_buffer_re),
        .write_data(weight_write_data),
        .write_buffer(weight_write_buffer),
        .read_buffer(weight_read_buffer),
        .read_data(weight_read_data),
        /* verilator lint_off PINCONNECTEMPTY */
        .writing_done(),
        .reading_done()
        /* verilator lint_on PINCONNECTEMPTY */
    );
    
    // Vector Buffer File for input vector storage  
    buffer_file #(
        .BUFFER_WIDTH(512),
        .BUFFER_COUNT(4),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(32)
    ) vector_buffer (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(vector_buffer_we),
        .read_enable(vector_buffer_re),
        .write_data(vector_write_data),
        .write_buffer(vector_write_buffer),
        .read_buffer(vector_read_buffer),
        .read_data(vector_read_data),
        /* verilator lint_off PINCONNECTEMPTY */
        .writing_done(),
        .reading_done()
        /* verilator lint_on PINCONNECTEMPTY */
    );
    
    // ===== Buffer Data Management =====
    // NOTE: Execution unit handles its own data management internally
    // These buffers are legacy/unused - kept for potential future top-level data access
    always_comb begin
        // Initialize x_buffer to zero (execution unit manages its own data)
        for (int i = 0; i < MAX_COLS; i++) begin
            x_buffer[i] = 0; // Execution unit doesn't use top-level x_buffer
        end
        
        // Initialize bias buffer to zero  
        for (int i = 0; i < MAX_ROWS; i++) begin
            bias_buffer[i] = 0; // Execution unit doesn't use top-level bias_buffer
        end
    end
    
    // ===== Main Control FSM =====
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            fetch_enable <= 0;
            exec_start <= 0;
            exec_start_delayed <= 0;
            instruction_count <= 0;
            last_executed_opcode <= 0;
            
            // Initialize buffer controls (legacy - kept for future use)
            weight_buffer_we <= 0;
            weight_buffer_re <= 0;
            vector_buffer_we <= 0;
            vector_buffer_re <= 0;
            weight_write_buffer <= 0;
            weight_read_buffer <= 0;
            vector_write_buffer <= 0;
            vector_read_buffer <= 0;
            
            // Initialize output
            for (int i = 0; i < OUT_N; i++) begin
                y[i] <= 0;
            end
        end else begin
            // Default values
            done <= 0;
            fetch_enable <= 0;
            exec_start <= 0;
            exec_start_delayed <= exec_start; // Delay exec_start by one cycle
            weight_buffer_we <= 0;
            weight_buffer_re <= 0;
            vector_buffer_we <= 0;
            vector_buffer_re <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        instruction_count <= 0;
                        state <= FETCH_INSTRUCTION;
                    end
                end
                
                FETCH_INSTRUCTION: begin
                    fetch_enable <= 1;
                    if (fetch_done) begin
                        state <= EXECUTE_INSTRUCTION;
                    end
                end
                
                EXECUTE_INSTRUCTION: begin
                    last_executed_opcode <= opcode;
                    
                    // Start execution immediately - the execution unit will handle everything
                    exec_start <= 1;
                    state <= WAIT_EXECUTION;
                end
                
                WAIT_EXECUTION: begin
                    if (exec_done) begin
                        instruction_count <= instruction_count + 1;
                        
                        // For now, just execute one instruction and finish
                        // Later this can be extended to handle multiple instructions
                        state <= OUTPUT_RESULTS;
                    end
                end
                
                OUTPUT_RESULTS: begin
                    // Copy results to output
                    for (int i = 0; i < OUT_N; i++) begin
                        y[i] <= exec_result[i];
                    end
                    state <= COMPLETE;
                end
                
                COMPLETE: begin
                    done <= 1;
                    // Stay in COMPLETE state until reset or new start
                    if (start) begin
                        instruction_count <= 0;
                        state <= FETCH_INSTRUCTION;
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
