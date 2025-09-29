// Top Module - tinyML Accelerator
// Fetches an instruction, decodes it, then executes it using the execution_unit.
// Simple single-instruction controller (fetch -> decode -> execute -> done).
module tinyml_accelerator_top #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,              // Match execution_unit default (256) for TILE_ELEMS consistency
    parameter ADDR_WIDTH = 24,
    parameter MAX_ROWS  = 784,
    parameter MAX_COLS  = 784,
    parameter OUT_N     = 10,
    parameter INSTR_WIDTH = 64,
    parameter HEX_FILE  = "/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/dram.hex"
)(
    input  logic clk,
    input  logic rst,            // Active high
    input  logic start,          // Start a single instruction cycle
    output logic signed [DATA_WIDTH-1:0] y [0:OUT_N-1], // Truncated execution results
    output logic done            // High for 1 cycle when instruction fully executed
);

    // ------------------------------------------------------------
    // State Machine for top-level control
    // ------------------------------------------------------------
    typedef enum logic [2:0] { T_IDLE, T_FETCH, T_WAIT_FETCH, T_DECODE, T_EXECUTE_START, T_EXECUTE_WAIT, T_DONE } top_state_t;
    top_state_t t_state, t_state_n;

    // ------------------------------------------------------------
    // Fetch Unit
    // ------------------------------------------------------------
    logic fetch_en;
    logic [ADDR_WIDTH-1:0] pc;
    logic [INSTR_WIDTH-1:0] instr;
    logic fetch_done;

    fetch_unit #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .INSTR_WIDTH(INSTR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH),
        .HEX_FILE   (HEX_FILE)
    ) fetch_u (
        .clk       (clk),
        .rst_n     (~rst),          // fetch_unit uses active-low reset
        .fetch_en_i(fetch_en),
        .pc_o      (pc),
        .instr_o   (instr),
        .done      (fetch_done)
    );

    // ------------------------------------------------------------
    // Decoder
    // ------------------------------------------------------------
    logic [4:0]  d_opcode;
    logic [4:0]  d_dest;
    logic [9:0]  d_length_or_cols;
    logic [9:0]  d_rows;
    logic [23:0] d_addr;
    logic [4:0]  d_b, d_x, d_w;

    i_decoder decoder_u (
        .instr (instr),
        .opcode(d_opcode),
        .dest  (d_dest),
        .length_or_cols(d_length_or_cols),
        .rows  (d_rows),
        .addr  (d_addr),
        .b     (d_b),
        .x     (d_x),
        .w     (d_w)
    );

    // Latched copies for stable drive during execution
    logic [4:0]  ex_opcode;
    logic [4:0]  ex_dest;
    logic [9:0]  ex_length_or_cols;
    logic [9:0]  ex_rows;
    logic [ADDR_WIDTH-1:0] ex_addr;
    logic [4:0]  ex_b, ex_x, ex_w;

    // ------------------------------------------------------------
    // Execution Unit
    // ------------------------------------------------------------
    logic exec_start;
    logic exec_done;
    logic signed [DATA_WIDTH-1:0] exec_result [0:MAX_ROWS-1];

    execution_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .TILE_WIDTH (TILE_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .MAX_ROWS   (MAX_ROWS),
        .MAX_COLS   (MAX_COLS)
    ) execution_u (
        .clk          (clk),
        .rst          (rst),
        .start        (exec_start),
        .opcode       (ex_opcode),
        .dest         (ex_dest),
        .length_or_cols(ex_length_or_cols),
        .rows         (ex_rows),
        .addr         (ex_addr),
        .b_id         (ex_b),
        .x_id         (ex_x),
        .w_id         (ex_w),
        .result       (exec_result),
        .done         (exec_done)
    );

    // ------------------------------------------------------------
    // Top-level FSM next-state logic
    // ------------------------------------------------------------
    always_comb begin
        t_state_n = t_state;
        fetch_en  = 1'b0;
        exec_start = 1'b0;
        done = 1'b0;
        case (t_state)
            T_IDLE: begin
                if (start) begin
                    t_state_n = T_FETCH;
                end
            end
            T_FETCH: begin
                fetch_en   = 1'b1;          // Pulse enable to begin fetch
                t_state_n  = T_WAIT_FETCH;
            end
            T_WAIT_FETCH: begin
                if (fetch_done) begin
                    t_state_n = T_DECODE;
                end
            end
            T_DECODE: begin
                // Latch decoded fields next cycle
                t_state_n = T_EXECUTE_START;
            end
            T_EXECUTE_START: begin
                exec_start = 1'b1;          // One-cycle start pulse
                t_state_n  = T_EXECUTE_WAIT;
            end
            T_EXECUTE_WAIT: begin
                if (exec_done) begin
                    t_state_n = T_DONE;
                end
            end
            T_DONE: begin
                done      = 1'b1;           // Pulse done
                t_state_n = T_IDLE;         // Ready for next instruction
            end
            default: t_state_n = T_IDLE;
        endcase
    end

    // ------------------------------------------------------------
    // Sequential section: state & latching decoded fields
    // ------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            t_state <= T_IDLE;
            ex_opcode <= '0; ex_dest <= '0; ex_length_or_cols <= '0; ex_rows <= '0; ex_addr <= '0; ex_b <= '0; ex_x <= '0; ex_w <= '0;
        end else begin
            t_state <= t_state_n;
            if (t_state == T_DECODE) begin
                ex_opcode        <= d_opcode;
                ex_dest          <= d_dest;
                ex_length_or_cols<= d_length_or_cols;
                ex_rows          <= d_rows;
                ex_addr          <= d_addr;
                ex_b             <= d_b;
                ex_x             <= d_x;
                ex_w             <= d_w;
            end
        end
    end

    // ------------------------------------------------------------
    // Output assignment (truncate execution results to OUT_N)
    // ------------------------------------------------------------
    generate
        genvar gi;
        for (gi = 0; gi < OUT_N; gi++) begin : OUT_COPY
            always_comb begin
                if (gi < MAX_ROWS) y[gi] = exec_result[gi]; else y[gi] = '0;
            end
        end
    endgenerate

endmodule
