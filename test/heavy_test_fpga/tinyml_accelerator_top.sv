module tinyml_accelerator_top #(
    parameter DATA_WIDTH  = 8,
    parameter TILE_WIDTH  = 64,
    parameter ADDR_WIDTH  = 16,
    parameter MAX_ROWS    = 1024,
    parameter MAX_COLS    = 1024,
    parameter OUT_N       = 10,
    parameter INSTR_WIDTH = 64,
    // Buffer sizing: must be large enough for the MNIST model
    parameter VECTOR_BUFFER_WIDTH = 8192,    // bits per vector buffer (1024 int8 elements @ TILE_ELEMS=8)
    parameter VECTOR_BUFFER_COUNT = 16,      // support buffer IDs 0-15
    parameter MATRIX_BUFFER_WIDTH = 131072, // bits per matrix buffer (16384 int8 elements)

    parameter HEX_FILE   = "/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/dram.hex"
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
    // Shared Memory System (FPGA Version)
    // ------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic [DATA_WIDTH-1:0] mem_rdata;
    logic [DATA_WIDTH-1:0] mem_wdata;
    logic                  mem_we;
    logic                  mem_req;
    
    // Unified Memory Instance
    
    simple_memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .HEX_FILE  (HEX_FILE)
    ) main_memory (
        .clk(clk),
        .we(mem_we),
        .addr(mem_addr),
        .din(mem_wdata),
        .dout(mem_rdata),
        .dump(1'b0)
    );
    

    // ------------------------------------------------------------
    // Fetch Unit
    // ------------------------------------------------------------
    logic fetch_en;
    logic [ADDR_WIDTH-1:0] pc; 
    logic [INSTR_WIDTH-1:0] instr;
    logic fetch_done;
    logic store_instr;
    
    logic fetch_mem_req;
    logic [ADDR_WIDTH-1:0] fetch_mem_addr;
    
    fetch_unit #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .INSTR_WIDTH(INSTR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) fetch_u (
        .clk       (clk),
        .rst_n     (~rst),          
        .fetch_en_i(fetch_en),
        .pc_o      (pc),
        .instr_o   (instr),
        .done      (fetch_done),
        // Memory 
        .mem_req   (fetch_mem_req),
        .mem_addr  (fetch_mem_addr),
        .mem_rdata (mem_rdata),
        .mem_valid (1'b1)
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
    // Fix: d_addr is 24 bits from decoder, but ex_addr should be ADDR_WIDTH (16)
    // We truncate/cast here.
    logic [ADDR_WIDTH-1:0] ex_addr; 
    logic [4:0]  ex_b, ex_x, ex_w;

    // ------------------------------------------------------------
    // Execution Unit
    // ------------------------------------------------------------
    logic exec_start;
    logic exec_done;
    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    // Result array holds last written tile (TILE_ELEMS elements)
    logic signed [DATA_WIDTH-1:0] exec_result [0:TILE_ELEMS-1];

    // Execution unit memory interface signals
    logic exec_mem_req;
    logic exec_mem_we;
    logic [ADDR_WIDTH-1:0] exec_mem_addr;
    logic [DATA_WIDTH-1:0] exec_mem_wdata;

    modular_execution_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .TILE_WIDTH (TILE_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .VECTOR_BUFFER_WIDTH (VECTOR_BUFFER_WIDTH),
        .VECTOR_BUFFER_COUNT (VECTOR_BUFFER_COUNT),
        .MATRIX_BUFFER_WIDTH (MATRIX_BUFFER_WIDTH),
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
        .done         (exec_done),
        // Memory Interface -- using ADDR_WIDTH 16 for FPGA, but modular unit expects 24? 
        // modular_execution_unit ADDR_WIDTH param needs to be consistent. 
        // In FPGA top, ADDR_WIDTH is 16. modular_execution_unit will get parameter 16.
        .mem_req      (exec_mem_req),
        .mem_we       (exec_mem_we),
        .mem_addr     (exec_mem_addr), // This will be 16 bits if param propagated correctly
        .mem_wdata    (exec_mem_wdata),
        .mem_rdata    (mem_rdata),
        .mem_valid    (1'b1)
    );

    // ------------------------------------------------------------
    // Top-level FSM next-state logic
    // ------------------------------------------------------------
    // Track if we hit a zero instruction (program end)
    logic program_done;
    
    // Memory Arbitration Logic
    always_comb begin
        if (t_state == T_FETCH || t_state == T_WAIT_FETCH) begin
            // Fetch Unit has access
            mem_addr  = fetch_mem_addr;
            mem_req   = fetch_mem_req;
            mem_we    = 1'b0; 
            mem_wdata = '0;
        end else begin
            // Execution Unit has access
            mem_addr  = exec_mem_addr;
            mem_req   = exec_mem_req;
            mem_we    = exec_mem_we;
            mem_wdata = exec_mem_wdata;
        end
    end
    
    always_comb begin
        t_state_n = t_state;
        fetch_en  = 1'b0;
        exec_start = 1'b0;
        done = 1'b0;
        program_done = 1'b0;
        
        case (t_state)
            T_IDLE: begin
                // Wait for start signal before beginning execution
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
                    if (instr == '0) begin
                        // Zero instruction = program end, signal done and go to IDLE
                        program_done = 1'b1;
                        done = 1'b1;
                        t_state_n = T_IDLE;
                    end else begin
                        t_state_n = T_DECODE;
                    end
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
                // Instruction complete, automatically fetch next instruction
                // (don't pulse done here - only pulse when program ends)
                t_state_n = T_FETCH;
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
            store_instr <= 0;
        end else begin
            t_state <= t_state_n;
            if (t_state == T_DECODE) begin
                ex_opcode        <= d_opcode;
                ex_dest          <= d_dest;
                ex_length_or_cols<= d_length_or_cols;
                ex_rows          <= d_rows;
                ex_addr          <= d_addr[ADDR_WIDTH-1:0]; // Truncate 24->16
                ex_b             <= d_b;
                ex_x             <= d_x;
                ex_w             <= d_w;
                // Track if this is a store instruction for potential future use
                store_instr <= (d_opcode == 5'b00011);
            end
        end
    end

    // ------------------------------------------------------------
    // Output assignment (truncate execution results to OUT_N)
    // ------------------------------------------------------------
    generate
        genvar gi;
        for (gi = 0; gi < OUT_N; gi++) begin : OUT_COPY
            if (gi < TILE_ELEMS) begin : VALID_OUT
                always_comb y[gi] = store_instr ? '1 : exec_result[gi];
            end else begin : INVALID_OUT
                always_comb y[gi] = '1;
            end
        end
    endgenerate
endmodule
