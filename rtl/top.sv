// Top Module - Wrapper for tinyML Accelerator
// This maintains compatibility with existing interface while using the new modular design
module top #(
    parameter DATA_WIDTH = 8, 
    parameter TILE_WIDTH = 128,
    parameter ADDR_WIDTH = 24,
    parameter OUT_N = 10
)(
    input logic clk,
    input logic rst,
    output logic signed [7:0] y [0:OUT_N-1],
    output logic done
);

    // Auto-start signal - can be modified to add external start control
    logic start_processing;
    
    // Start processing on the first clock after reset
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            start_processing <= 0;
        end else begin
            start_processing <= 1;  // Auto-start for now
        end
    end
    
    // Instantiate the main accelerator
    tinyml_accelerator_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_ROWS(128),
        .MAX_COLS(128),
        .OUT_N(OUT_N)
    ) accelerator (
        .clk(clk),
        .rst(rst),
        .start(start_processing),
        .y(y),
        .done(done)
    );

endmodule

// Legacy/Original Top Module (renamed for reference)
// This is the original implementation kept for comparison
module top_legacy #(
parameter DATA_WIDTH = 8, 
parameter TILE_WIDTH = 128,
parameter ADDR_WIDTH = 24,
parameter OUT_N = 10
)
(
    input logic clk,
    input logic rst,
    // Missing required inputs that should be added:
    input logic signed [DATA_WIDTH-1:0] x [0:127],      // Input vector
    input logic signed [DATA_WIDTH-1:0] bias [0:127],   // Bias vector
    output logic signed [7:0] y [0:OUT_N-1],
    output logic done
);
    // Widen state encoding (was [1:0] but had >4 states)
    typedef enum logic [2:0] {
        IDLE, GET_INSTRUCTION, GET_VALUES, EXECUTE, PROCESS_WEIGHTS, DONE
    } state_t;
    state_t state;

    // Number of elements per tile (8-bit elements)
    localparam int TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;

    // Track current exec mode and dynamic load length
    typedef enum logic [1:0] { EXEC_NONE, EXEC_GEMV, EXEC_LOADM } exec_mode_e;
    exec_mode_e exec_mode;
    logic [19:0] load_length_bits;
    logic [4:0]  last_opcode;

    // --- GET_INSTRUCTION start ---
    logic [63:0] instruction;
    logic [3:0] instr_idx, byte_cnt;
    logic [7:0] mem_out;
    logic [23:0] mem_addr;
    logic is_store;
    // Use consistent simple_memory port names (.din/.dout)
    simple_memory get_instruction(
        .clk(clk),
        .we(0),
        .addr(mem_addr),
        .din(8'b0), // No write data needed
        .dout(mem_out)
    );
    // --- GET_INSTRUCTION end ---

    // instruction values
    // Avoid shadowing top inputs x[] by renaming decoded fields
    logic [4:0]  opcode, dest, b, x_id, w_id;
    logic [9:0]  length_or_cols, rows;
    logic [23:0] addr;
    i_decoder decode(
        .instruction(instruction),
        .opcode(opcode),
        .dest(dest),
        .b(b),
        .x(x_id),
        .w(w_id),
        .length_or_cols(length_or_cols),
        .rows(rows),
        .addr(addr)
    );

    // --- LOAD_M start ---
    // Packed tile from memory (TILE_WIDTH bits = TILE_ELEMS * DATA_WIDTH)
    logic [TILE_WIDTH-1:0] data_out;
    logic tile_out, valid_out;

    // One-tile, on-demand load control
    logic [23:0] curr_addr;
    logic        load_start, load_inflight;

    load_m #(
        .TILE_WIDTH(TILE_WIDTH)
    ) load_matrix (
        .clk(clk),
        .rst(rst),
        .valid_in(load_start),       // pulse to start transfer
        .dram_addr(curr_addr),       // base address
        .length(load_length_bits),   // dynamic: one tile for GEMV, full matrix for LOAD_M
        .data_out(data_out),
        .tile_out(tile_out),
        .valid_out(valid_out)
    );

    // Optional buffer kept instantiated but not used for reads here
    weight_buffer_file #(
        .DATA_WIDTH(DATA_WIDTH),
        .BUFFER_WIDTH(1024), // in testing purposes, should be 100352
        .BUFFER_COUNT(2),
        .TILE_WIDTH(TILE_WIDTH)
    ) weight_buffer (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(tile_out),
        .read_enable(0),
        .write_data(data_out),
        .write_buffer(dest),
        .read_buffer(0),
        .read_data(0), // Not used in this example
        .writing_done(), // Not used in this example
        .reading_done() // Not used in this example
    );
    // --- LOAD_M end ---

    // --- GEMV integration start ---
    // Unpacked tile for top_gemv (array form)
    logic signed [DATA_WIDTH-1:0] w_tile_bus [0:TILE_ELEMS-1];
    // Handshake to GEMV
    logic gemv_start, gemv_done, gemv_w_ready, gemv_tile_done;
    logic w_valid;

    // Unpack packed tile MSB-first to element array
    always_comb begin
        for (int i = 0; i < TILE_ELEMS; i++) begin
            w_tile_bus[i] = data_out[TILE_WIDTH-1 - i*DATA_WIDTH -: DATA_WIDTH];
        end
    end

    logic signed [DATA_WIDTH-1:0] y_gemv [0:127];

    top_gemv #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_ROWS(128),
        .MAX_COLUMNS(128),
        .TILE_SIZE(TILE_ELEMS)
    ) gemv (
        .clk(clk),
        .rst(rst),
        .start(gemv_start),
        .w_ready(gemv_w_ready),
        .w_valid(w_valid),
        .w_tile_row_in(w_tile_bus),
        .x(x),
        .bias(bias),
        .rows(rows),
        .cols(length_or_cols),
        .y(y_gemv),
        .tile_done(gemv_tile_done),
        .done(gemv_done)
    );

    // Gate w_valid only during GEMV processing
    assign w_valid = (state == PROCESS_WEIGHTS) && (exec_mode == EXEC_GEMV) && tile_out;
    // --- GEMV integration end ---

    // --- RELU compute (combinational) ---
    logic signed [DATA_WIDTH-1:0] relu_out [0:127];
    relu #(
        .DATA_WIDTH(DATA_WIDTH),
        .LENGTH(128)
    ) relu_inst (
        .in_vec(x),
        .out_vec(relu_out)
    );

    // --- FSM ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            instr_idx <= 0;
            byte_cnt <= 0;
            done <= 0;
            gemv_start <= 0;
            load_start <= 0;
            load_inflight <= 0;
            curr_addr <= '0;
            exec_mode <= EXEC_NONE;
            load_length_bits <= '0;
            last_opcode <= 5'h00;
        end else begin
            // defaults
            gemv_start <= 0;
            load_start <= 0;
            done <= 0;

            case (state)
                IDLE: begin
                    state <= GET_INSTRUCTION;
                    if (instr_idx < 8) begin
                        mem_addr <= instr_idx * 8; // Assuming each instruction is 8 bytes
                    end else begin
                        state <= DONE;
                    end
                end

                GET_INSTRUCTION: begin
                    instruction[(8 - byte_cnt)*8 +: 8] <= mem_out;
                    byte_cnt <= byte_cnt + 1;
                    if (byte_cnt == 8) begin
                        state <= EXECUTE;
                        instr_idx <= instr_idx + 1;
                    end
                end

                EXECUTE: begin
                    last_opcode <= opcode;
                    unique case (opcode)
                        5'h00: begin // NOP
                            exec_mode <= EXEC_NONE;
                            state <= DONE;
                        end
                        5'h01: begin // LOAD_V (not implemented here)
                            exec_mode <= EXEC_NONE;
                            state <= DONE;
                        end
                        5'h02: begin // LOAD_M
                            curr_addr        <= addr; // base matrix address
                            load_length_bits <= (rows * length_or_cols) * 8; // bits
                            load_inflight    <= 0;
                            exec_mode        <= EXEC_LOADM;
                            load_start       <= 1;    // single pulse to start the transfer
                            state            <= PROCESS_WEIGHTS;
                        end
                        5'h03: begin // STORE (not implemented)
                            exec_mode <= EXEC_NONE;
                            state <= DONE;
                        end
                        5'h04: begin // GEMV
                            curr_addr        <= addr;         // base matrix address
                            load_length_bits <= TILE_WIDTH;   // one tile per request
                            load_inflight    <= 0;
                            exec_mode        <= EXEC_GEMV;
                            gemv_start       <= 1;            // pulse start
                            state            <= PROCESS_WEIGHTS;
                        end
                        5'h05: begin // RELU
                            exec_mode <= EXEC_NONE;
                            state <= DONE;
                        end
                        default: begin
                            exec_mode <= EXEC_NONE;
                            state <= DONE;
                        end
                    endcase
                end

                PROCESS_WEIGHTS: begin
                    unique case (exec_mode)
                        EXEC_GEMV: begin
                            if (gemv_w_ready && !load_inflight) begin
                                load_start    <= 1;                // one-cycle pulse
                                load_inflight <= 1;
                            end
                            if (tile_out) begin
                                load_inflight <= 0;
                                curr_addr <= curr_addr + TILE_ELEMS[23:0];
                            end
                            if (gemv_done) begin
                                state <= DONE;
                            end
                        end
                        EXEC_LOADM: begin
                            if (valid_out) begin
                                state <= DONE;
                            end
                        end
                        default: begin
                            state <= DONE;
                        end
                    endcase
                end

                DONE: begin
                    done <= 1; // Indicate completion
                    if (last_opcode == 5'h04) begin
                        for (int k = 0; k < OUT_N; k++) begin
                            y[k] <= y_gemv[k];
                        end
                    end else if (last_opcode == 5'h05) begin
                        for (int k = 0; k < OUT_N; k++) begin
                            y[k] <= relu_out[k];
                        end
                    end else begin
                        for (int k = 0; k < OUT_N; k++) begin
                            y[k] <= '0;
                        end
                    end
                end
            endcase
        end
    end

endmodule
