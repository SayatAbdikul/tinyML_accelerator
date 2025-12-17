module fetch_unit #(
    parameter ADDR_WIDTH  = 24,  // word-addressed (each address is one byte in current design)
    parameter INSTR_WIDTH = 64,
    parameter DATA_WIDTH  = 8,
    parameter HEX_FILE    = "/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/dram.hex"
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     fetch_en_i,
    output logic [ADDR_WIDTH-1:0]    pc_o,
    output logic [INSTR_WIDTH-1:0]   instr_o,
    output logic                     done
);

    typedef enum logic [1:0] { IDLE, FETCH, DONE } state_t;
    state_t state, next_state;

    logic [$clog2(INSTR_WIDTH/8)-1:0] byte_cnt;
    logic [ADDR_WIDTH-1:0]            pc;
    logic [INSTR_WIDTH-1:0]           instruction;

    // Memory interface
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic [DATA_WIDTH-1:0] mem_dout;

    simple_memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .HEX_FILE  (HEX_FILE)
    ) memory_inst (
        .clk (clk),
        .we  (1'b0),
        .addr(mem_addr),
        .din ({DATA_WIDTH{1'b0}}),
        .dout(mem_dout),
        .dump(1'b0)
    );

    // Address always current pc (pc points to next byte request)
    assign mem_addr = pc;

    // Expose pc
    assign pc_o = pc;

    // Next State logic (accounts for initial latency cycle)
    always_comb begin
        next_state = state;
        unique case (state)
            IDLE:  if (fetch_en_i) next_state = FETCH;
            FETCH: if (byte_cnt == $clog2(INSTR_WIDTH/8)'((INSTR_WIDTH/8)-1)) next_state = DONE;
            DONE:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // Sequential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= IDLE;
            pc                <= '0;
            byte_cnt          <= '0;
            instruction       <= '0;
            instr_o           <= '0;
            done              <= 1'b0;
        end else begin
            state      <= next_state;
            done       <= 1'b0; // pulse style

            unique case (state)
                IDLE: begin
                    if (fetch_en_i) begin
                        byte_cnt          <= '0;
                        pc               <= pc + 1; // Hold current pc (next byte to fetch)
                    end
                end

                FETCH: begin
                    // Issue next address every cycle
                    instruction[((INSTR_WIDTH/8 - 1 - int'(byte_cnt))*8) +: 8] <= mem_dout;
                    $display("Fetched byte: %h for mem_addr=%0d", mem_dout, mem_addr);
                    if (byte_cnt < $clog2(INSTR_WIDTH/8)'((INSTR_WIDTH/8)-1)) begin
                        byte_cnt <= byte_cnt + 1'b1;
                        pc <= pc + 1'b1;
                    end
                end

                DONE: begin
                    instr_o <= instruction;
                    $display("Fetch done, instruction: %h", instruction);
                    done    <= 1'b1;
                end
            endcase
        end
    end

endmodule
