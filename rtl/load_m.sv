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
    // Single shared memory instance
    logic [7:0] mem_data_out;
    logic [23:0] mem_addr;
    logic mem_we = 0;

    simple_memory #(
        .ADDR_WIDTH(24),
        .DATA_WIDTH(8)
    ) memory_inst (
        .clk(clk),
        .we(mem_we),
        .addr(mem_addr),
        .din(8'b0),
        .dout(mem_data_out)
    );

    // Internal state
    logic [$clog2(NUM_BYTES+1):0] byte_cnt;
    logic [TILE_WIDTH-1:0] tile;

    enum logic [1:0] {IDLE, READING, NEXT_TILE, DONE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            byte_cnt <= 0;
            mem_addr <= 0;
            tile <= 0;
            valid_out <= 0;
            length_cnt <= 0;
            $display("Resetting load_m module");
        end else begin
            valid_out <= 0;
            tile_out <= 0;
            case (state)
                IDLE: begin
                    byte_cnt <= 0;
                    if (valid_in) begin
                        mem_addr <= dram_addr;  // Initialize address
                        state <= READING;
                    end
                end

                READING: begin
                    tile[((NUM_BYTES) - byte_cnt)*8 +: 8] <= mem_data_out;// Store current byte
                    byte_cnt <= byte_cnt + 1;
                    mem_addr <= mem_addr + 1;  // Increment to next byte address

                    if (byte_cnt == NUM_BYTES) begin
                        state <= NEXT_TILE;
                    end
                end

                NEXT_TILE: begin
                    $display("Tile data: %0h", tile);
                    $display("Address in decimal: %0d", mem_addr); // Shows next address
                    mem_addr <= mem_addr - 1;  // Prepare for next tile
                    tile_out <= 1;
                    if (length_cnt + TILE_WIDTH <= length) begin
                        length_cnt <= length_cnt + TILE_WIDTH;
                        state <= READING;
                        byte_cnt <= 0; // Reset for new tile
                    end else begin
                        state <= DONE;
                        valid_out <= 1;
                    end
                end

                DONE: begin
                    state <= IDLE;
                end
            endcase
        end
    end
    assign data_out = tile;
endmodule
