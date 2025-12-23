module load_m #(
    parameter TILE_WIDTH = 256,  // Must be multiple of 8
    parameter DATA_WIDTH = 8
)
(
    input logic clk,
    input logic rst,
    input logic valid_in,
    input logic [23:0] dram_addr,
    input logic [9:0] rows,       // number of rows (for row-aware padding)
    input logic [9:0] cols,       // number of columns per row
    output logic [TILE_WIDTH-1:0] data_out,
    output logic tile_out,
    output logic valid_out
);

    localparam NUM_BYTES = TILE_WIDTH / 8;  // 32 bytes per tile
    logic [7:0] mem_data_out;
    logic [23:0] mem_addr;
    logic [23:0] base_addr;  // Start address of current row in memory

    simple_memory #(
        .ADDR_WIDTH(24),
        .DATA_WIDTH(8)
    ) memory_inst (
        .clk(clk),
        .we(0),
        .addr(mem_addr),
        .din(8'b0),
        .dout(mem_data_out),
        .dump(1'b0)
    );

    logic [$clog2(NUM_BYTES)-1:0] byte_cnt;       // Byte position within current tile (0-31)
    logic [TILE_WIDTH-1:0] tile;
    logic [9:0] current_row;                       // Current row being processed
    logic [9:0] col_in_row;                        // Current column position within row
    logic [9:0] tile_in_row;                       // Current tile within row
    logic [9:0] tiles_per_row;                     // Number of tiles per row (with padding)

    enum logic [2:0] {IDLE, INIT_READING, READING, NEXT_TILE, NEXT_ROW, DONE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            byte_cnt <= '0;
            mem_addr <= '0;
            base_addr <= '0;
            tile <= '0;
            valid_out <= 0;
            tile_out <= 0;
            current_row <= 0;
            col_in_row <= 0;
            tile_in_row <= 0;
            tiles_per_row <= 0;
            // $display("Resetting load_m module");
        end else begin
            tile_out <= 0;
            
            case (state)
                IDLE: begin
                    byte_cnt <= '0;
                    valid_out <= 0;
                    if (valid_in) begin
                        mem_addr <= dram_addr;
                        base_addr <= dram_addr;
                        current_row <= 0;
                        col_in_row <= 0;
                        tile_in_row <= 0;
                        tiles_per_row <= (cols + 10'd31) / 10'd32;  // Ceiling division
                        state <= INIT_READING;
                        // $display("[LOAD_M] Starting: rows=%0d, cols=%0d, tiles_per_row=%0d, addr=0x%h", 
                        //          rows, cols, (cols + 10'd31) / 10'd32, dram_addr);
                    end
                end

                INIT_READING: begin
                    // Prime 1 cycle for sync read
                    state <= READING;
                    mem_addr <= mem_addr + 1;
                end

                READING: begin
                    // Capture data - only if within valid column range for this row
                    if (col_in_row < cols) begin
                        tile[int'(byte_cnt)*8 +: 8] <= mem_data_out;
                    end else begin
                        tile[int'(byte_cnt)*8 +: 8] <= '0;  // Padding zeros
                    end
                    
                    col_in_row <= col_in_row + 1;

                    // End-of-tile check
                    if (int'(byte_cnt) == NUM_BYTES-1) begin
                        state <= NEXT_TILE;
                    end else begin
                        byte_cnt <= byte_cnt + 1;
                        // Only advance mem_addr if we're reading valid data
                        if (col_in_row + 1 < cols) begin
                            mem_addr <= mem_addr + 1;
                        end
                    end
                end

                NEXT_TILE: begin
                    tile_out <= 1;
                    tile_in_row <= tile_in_row + 1;
                    byte_cnt <= '0;

                    // Check if this row is done
                    if (tile_in_row + 1 >= tiles_per_row) begin
                        // Move to next row
                        state <= NEXT_ROW;
                    end else begin
                        // More tiles in this row
                        state <= INIT_READING;
                    end
                end

                NEXT_ROW: begin
                    current_row <= current_row + 1;
                    
                    if (current_row + 1 >= rows) begin
                        // All rows complete
                        state <= DONE;
                        valid_out <= 1;
                        // $display("[LOAD_M] All rows complete");
                    end else begin
                        // Start next row: update base_addr to point to next row in memory
                        base_addr <= base_addr + cols;
                        mem_addr <= base_addr + cols;
                        col_in_row <= 0;
                        tile_in_row <= 0;
                        state <= INIT_READING;
                    end
                end

                DONE: begin
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    assign data_out = tile;
endmodule
