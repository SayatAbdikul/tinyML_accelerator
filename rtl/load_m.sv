module load_m #(
    parameter TILE_WIDTH = 256  // Must be multiple of 8
)
(
    input logic clk,
    input logic rst,
    input logic valid_in,
    input logic [23:0] dram_addr,
    input logic [19:0] length,  // in bits
    output logic [TILE_WIDTH-1:0] data_out,
    output logic tile_out,
    output logic valid_out
);

    localparam NUM_BYTES = TILE_WIDTH / 8;
    logic [19:0] length_cnt;
    logic [7:0] mem_data_out;
    logic [23:0] mem_addr;

    simple_memory #(
        .ADDR_WIDTH(24),
        .DATA_WIDTH(8)
    ) memory_inst (
        .clk(clk),
        .we(0),
        .addr(mem_addr),
        .din(8'b0),
        .dout(mem_data_out)
    );

    logic [$clog2(NUM_BYTES)-1:0] byte_cnt;
    logic [TILE_WIDTH-1:0] tile;

    enum logic [2:0] {IDLE, INIT_READING, READING, NEXT_TILE, DONE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            byte_cnt <= '0;
            mem_addr <= '0;
            tile <= '0;
            valid_out <= 0;
            tile_out <= 0;
            length_cnt <= 0;
            $display("Resetting load_m module");
        end else begin
            tile_out <= 0;
            // Only clear valid_out when starting a new operation
            case (state)
                IDLE: begin
                    byte_cnt <= '0;
                    if (valid_in) begin
                        mem_addr   <= dram_addr;  // Present first address
                        length_cnt <= 0;          // Start of transfer
                        state      <= INIT_READING; // Prime 1 cycle for sync read
                        valid_out <= 0; // Clear valid_out when starting new operation
                        // $display(" length is %0d bits", length);
                    end
                end

                // New priming state to account for 1-cycle read latency
                INIT_READING: begin
                    // After this cycle, mem_data_out matches current mem_addr
                    state <= READING;
                    mem_addr <= mem_addr + 1;
                end

                READING: begin
                    // Capture the data for the address presented in the previous cycle
                    if(length_cnt + byte_cnt*8 < length) begin
                        tile[((NUM_BYTES-1) - int'(byte_cnt))*8 +: 8] <= mem_data_out;
                        //$display("length_cnt + byte_cnt*8 = %0d, length = %0d", length_cnt + byte_cnt*8, length);
                        //$display("Captured %h from addr = %h", mem_data_out, mem_addr - 1);
                    end
                    else begin
                        tile[((NUM_BYTES-1) - int'(byte_cnt))*8 +: 8] <= '0; // Fill with zeros if out of range
                        //$display("Out of range read at addr = %h, filling with zeros", mem_addr - 1);
                    end
                    //$display("Captured %h from addr = %h", mem_data_out, mem_addr - 1);


                    // End-of-tile check (NUM_BYTES captures: 0..NUM_BYTES-1)
                    if (byte_cnt == NUM_BYTES-1) begin
                        state <= NEXT_TILE;
                    end else begin
                        byte_cnt <= byte_cnt + 1;
                        mem_addr <= mem_addr + 1;
                    end
                end

                NEXT_TILE: begin
                    //$display("Tile data: %0h", tile);
                    //$display("Last address read (decimal): %0d", mem_addr - 1);
                    tile_out <= 1;
                    length_cnt <= length_cnt + TILE_WIDTH;

                    // If more full tiles remain, re-prime before next capture
                    if (length_cnt + TILE_WIDTH < length) begin
                        state    <= INIT_READING; // prime for the next tile
                        byte_cnt <= '0;           // reset for new tile
                        // mem_addr already points to the next byte to read
                    end else begin
                        state <= DONE;
                        valid_out <= 1; // Assert completion signal
                    end
                end

                DONE: begin
                    // Keep valid_out high until acknowledged
                    // This ensures the execution unit can see the completion
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    assign data_out = tile;
endmodule
