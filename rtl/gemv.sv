module gemv #(parameter DATA_WIDTH = 8, parameter ROWS = 128, parameter COLUMNS = 128, parameter TILE_SIZE = 8)
(
  input logic clk,
  input logic rst,
  input logic signed [DATA_WIDTH-1:0] w [0:ROWS-1][0:COLUMNS-1],
  input logic signed [DATA_WIDTH-1:0] x [0:COLUMNS-1],
  input logic signed [DATA_WIDTH-1:0] bias [0:ROWS-1],
  output logic signed [DATA_WIDTH-1:0] y [0:ROWS-1],
  output logic done
);
    typedef enum logic [2:0] {
        IDLE, PRESENT_TILE, ACCUMULATE, NEXT_ROW, BIAS, DONE
    } state_t;
    
    state_t state;

    localparam ROW_IDX_WIDTH  = $clog2(ROWS);
    localparam TILE_IDX_WIDTH = $clog2(COLUMNS / TILE_SIZE);

    logic [DATA_WIDTH-1:0] res [0:ROWS-1];
    logic [ROW_IDX_WIDTH-1:0] row_idx;
    logic [TILE_IDX_WIDTH-1:0] tile_idx;
    
    logic signed [DATA_WIDTH-1:0] w_in [0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] x_in [0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] pe_out [0:TILE_SIZE-1];
    logic [DATA_WIDTH-1:0] temp_sum;

    // Input selection
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            int idx = tile_idx * TILE_SIZE + i;
            w_in[i] = (idx < COLUMNS) ? w[row_idx][idx] : '0;
            x_in[i] = (idx < COLUMNS) ? x[idx] : '0;
        end
    end

    // PE array
    genvar i;
    generate for (i = 0; i < TILE_SIZE; i++) begin
        pe #(DATA_WIDTH) pe_inst (
            .clk(clk),
            .rst(rst),
            .w(w_in[i]),
            .x(x_in[i]),
            .y(pe_out[i])
        );
    end 
    endgenerate

    // Sum PE outputs
    always_comb begin
        temp_sum = '0;
        for (int j = 0; j < TILE_SIZE; j++) begin
            temp_sum = temp_sum + pe_out[j];
        end
    end

    // FSM controller
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            row_idx <= '0;
            tile_idx <= '0;
            for (int j = 0; j < ROWS; j++) begin
                res[j] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    row_idx <= '0;
                    tile_idx <= '0;
                    for (int j = 0; j < ROWS; j++) begin
                        res[j] <= '0;
                    end
                    state <= PRESENT_TILE;
                end
                
                PRESENT_TILE: begin
                    // Present tile to PEs
                    state <= ACCUMULATE;  // Wait for PE results
                end
                
                ACCUMULATE: begin
                    // Accumulate results from PEs
                    res[row_idx] <= res[row_idx] + temp_sum;
                    
                    // Check if last tile in row
                    if ({4'b0, tile_idx} < (COLUMNS/TILE_SIZE)-1) begin
                        tile_idx <= tile_idx + 1;
                        state <= PRESENT_TILE;  // Process next tile
                    end else begin
                        tile_idx <= '0;
                        state <= NEXT_ROW;  // Move to next row
                    end
                end
                
                NEXT_ROW: begin
                    if ({4'b0, row_idx} < ROWS-1) begin
                        row_idx <= row_idx + 1;
                        state <= PRESENT_TILE;  // Process next row
                    end else begin
                        state <= BIAS;  // All rows processed
                    end
                end
                
                BIAS: begin
                    // Add bias to all rows
                    for (int j = 0; j < ROWS; j++) begin
                        res[j] <= res[j] + bias[j];
                    end
                    state <= DONE;
                end
                
                DONE: begin
                    state <= IDLE;  // Computation complete
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    // Output handling
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int j = 0; j < ROWS; j++) begin
                y[j] <= '0;
            end
        end else if (state == DONE) begin
            for (int j = 0; j < ROWS; j++) begin
                y[j] <= res[j];
            end
        end
    end
    
    // Done signal generation
    assign done = (state == DONE);

endmodule
