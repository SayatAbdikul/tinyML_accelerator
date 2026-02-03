module load_m #(
    parameter TILE_WIDTH = 256,
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
    output logic valid_out,
    
    // Memory interface
    output logic mem_req,
    output logic [23:0] mem_addr,
    input  logic [DATA_WIDTH-1:0] mem_rdata,
    input  logic mem_valid
);

    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    // logic [DATA_WIDTH-1:0] mem_data_out; -> moved to input mem_rdata
    // logic [23:0] mem_addr; -> moved to output

    logic [$clog2(TILE_ELEMS)-1:0] elem_cnt;       // Element position within current tile

    logic [TILE_WIDTH-1:0] tile;
    logic [9:0] current_row;                       // Current row being processed
    logic [9:0] col_in_row;                        // Current column position within row
    logic [9:0] tile_in_row;                       // Current tile within row
    logic [9:0] tiles_per_row;                     // Number of tiles per row (with padding)

    enum logic [2:0] {IDLE, INIT_READING, READING, NEXT_TILE, NEXT_ROW, DONE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            elem_cnt <= '0;
            mem_addr <= '0;
            // base_addr <= '0;
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
                    elem_cnt <= '0;
                    valid_out <= 0;
                    if (valid_in) begin
                        mem_addr <= dram_addr;
                        // base_addr <= dram_addr; // Unused
                        current_row <= 0;
                        col_in_row <= 0;
                        tile_in_row <= 0;
                        tiles_per_row <= (cols + TILE_ELEMS - 1) / TILE_ELEMS;  // Ceiling division
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
                    // Ideally check mem_valid here if arbitration is variable lag
                    if (col_in_row < cols) begin
                        tile[int'(elem_cnt)*DATA_WIDTH +: DATA_WIDTH] <= mem_rdata;
                    end else begin
                        tile[int'(elem_cnt)*DATA_WIDTH +: DATA_WIDTH] <= '0;  // Padding zeros
                    end
                    
                    col_in_row <= col_in_row + 1;

                    // End-of-tile check
                    if (int'(elem_cnt) == TILE_ELEMS-1) begin
                        state <= NEXT_TILE;
                    end else begin
                        elem_cnt <= elem_cnt + 1;
                        // Always advance mem_addr since memory is padded to tile width
                        mem_addr <= mem_addr + 1;
                    end
                end

                NEXT_TILE: begin
                    tile_out <= 1;
                    tile_in_row <= tile_in_row + 1;
                    elem_cnt <= '0;

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
                    end else begin
                        // Start next row
                        // Memory address is already at the start of next row due to padded reading
                        // base_addr <= mem_addr; 
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
    // Request valid during active reading states
    assign mem_req = (state == INIT_READING || state == READING) && (state != NEXT_TILE);
endmodule
