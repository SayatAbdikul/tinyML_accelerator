module load_v #(
    parameter TILE_WIDTH = 256,  // Must be multiple of 8
    parameter DATA_WIDTH = 8
)
(
    input logic clk,
    input logic rst,
    input logic valid_in,
    input logic [23:0] dram_addr,
    input logic [9:0] length,  // in elements - increased width to ensure no truncation
    output logic [DATA_WIDTH-1:0] data_out [0:TILE_WIDTH/DATA_WIDTH-1],
    output logic tile_out,
    output logic valid_out
);

    // Unify counts
    localparam ELEM_COUNT = TILE_WIDTH / DATA_WIDTH;

    // Enforce 8-bit elements; generalization would require packing/unpacking
    initial begin
        if (DATA_WIDTH != 8) begin
            $fatal(1, "load_v expects DATA_WIDTH=8, got %0d", DATA_WIDTH);
        end
    end

    logic [15:0] length_cnt;
    // Single shared memory instance
    logic [7:0] mem_data_out;
    // Expose for Verilator TB backdoor writes
    logic [23:0] mem_addr /*verilator public*/;
    logic        mem_we   /*verilator public*/ = 0;
    logic [7:0]  mem_din  /*verilator public*/;

    simple_memory #(
        .ADDR_WIDTH(24),
        .DATA_WIDTH(DATA_WIDTH)
    ) memory_inst (
        .clk(clk),
        .we(mem_we),
        .addr(mem_addr),
        .din(mem_din),   // allow TB writes; was 8'b0
        .dout(mem_data_out),
        .dump(1'b0)
    );

    // Internal state
    // Count 0..ELEM_COUNT-1
    logic [$clog2(ELEM_COUNT)-1:0] byte_cnt;
    logic [DATA_WIDTH-1:0] tile [0:ELEM_COUNT-1];

    enum logic [2:0] {IDLE, INIT_READING, READING, NEXT_TILE, DONE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            byte_cnt <= '0;
            mem_addr <= '0;
            for (int i = 0; i < ELEM_COUNT; i++) begin
                tile[i] <= '0;
            end
            valid_out <= 0;
            tile_out  <= 0;
            length_cnt <= 0;
            // $display("Resetting load_v module");
        end else begin
            valid_out <= 0;
            tile_out  <= 0;
            case (state)
                IDLE: begin
                    byte_cnt <= '0;
                    if (valid_in) begin
                        mem_addr   <= dram_addr;  // Present first address
                        length_cnt <= 0;          // New transfer
                        state      <= INIT_READING; // Prime one cycle for sync read
                    end
                end

                // New priming state for 1-cycle read latency
                INIT_READING: begin
                    // After this cycle, mem_data_out matches mem_addr
                    state <= READING;
                    mem_addr <= mem_addr + 1;
                end

                READING: begin
                    // Capture data for the address presented in the previous cycle
                    if(length_cnt + byte_cnt*8 < length * DATA_WIDTH) begin
                        tile[byte_cnt] <= mem_data_out;
                    end else begin
                        tile[byte_cnt] <= '0; // Fill with zeros if out of range
                    end

                    // End-of-tile after ELEM_COUNT captures (0..ELEM_COUNT-1)
                    if (int'(byte_cnt) == ELEM_COUNT - 1) begin
                        state <= NEXT_TILE;
                    end else begin
                        byte_cnt <= byte_cnt + 1;
                         // Prepare next read
                        mem_addr <= mem_addr + 1;
                    end
                end

                NEXT_TILE: begin
                    //$display("Tile loaded. Last address: %0h. Length is %0h elements. Remaining: %0h bits", mem_addr - 1, length, (length * DATA_WIDTH) - (length_cnt + TILE_WIDTH));
                    // for (int i = 0; i < ELEM_COUNT; i++) begin
                    //     if (tile[i] != 0) begin
                    //         $display(" non-zero load_v data_out[%0d] = %0h", i, tile[i]);
                    //     end
                    // end
                    tile_out <= 1;
                    // Account for this tile worth of bits
                    length_cnt <= length_cnt + TILE_WIDTH;

                    // If more full tiles remain, re-prime before next capture
                    if (length_cnt + TILE_WIDTH < length * DATA_WIDTH) begin
                        state    <= INIT_READING; // prime for next tile
                        byte_cnt <= '0;           // reset for new tile
                        // mem_addr already points to the next byte to read
                    end else begin
                        state     <= DONE;
                        valid_out <= 1;
                    end
                end

                DONE: begin
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    always_comb begin
        // Assign output data
        for (int i = 0; i < ELEM_COUNT; i++) begin
            data_out[i] = tile[i];
        end
    end
endmodule
