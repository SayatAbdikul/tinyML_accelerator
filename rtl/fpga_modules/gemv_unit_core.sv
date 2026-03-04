

module gemv_unit_core #(
    parameter DATA_WIDTH = 8,
    parameter MAX_ROWS = 784,
    parameter MAX_COLUMNS = 784,
    parameter TILE_SIZE = 32
) (
    input logic clk,
    input logic rst,

    // Control Signals
    input logic start,
    output logic w_ready,

    // Weight tile input (streamed per row)
    input logic w_valid,
    input logic signed [DATA_WIDTH-1:0] w_tile_row_in [0:TILE_SIZE-1],
    
    // X vector tile input (streamed)
    input logic x_tile_valid,
    input logic signed [DATA_WIDTH-1:0] x_tile_in [0:TILE_SIZE-1],
    output logic x_tile_ready,
    input logic [9:0] x_tile_idx,
    
    // Bias tile input (streamed)
    input logic bias_tile_valid,
    input logic signed [DATA_WIDTH-1:0] bias_tile_in [0:TILE_SIZE-1],
    output logic bias_tile_ready,
    input logic [9:0] bias_tile_idx,
    
    // Y tile output (streamed)
    output logic y_tile_valid,
    output logic signed [DATA_WIDTH-1:0] y_tile_out [0:TILE_SIZE-1],
    input logic y_tile_ready,
    output logic [9:0] y_tile_idx,
    
    input logic [9:0] rows,
    input logic [9:0] cols,

    output logic tile_done,
    output logic done
);

    // FSM States
    enum int unsigned {
        IDLE            = 0,
        LOAD_X          = 1,
        STORE_X         = 2,
        LOAD_BIAS       = 3,
        STORE_BIAS      = 4,     // Write bias elements to res_mem (accumulator)
        LOAD_X_TILE     = 6,     // Read x tile from x_mem for current tile_idx
        WAIT_TILE       = 7,
        WAIT_PE         = 8,
        READ_ACCUM      = 17,    // Present res_rad=row_idx; BSRAM registers output at clk edge
        ACCUMULATE      = 9,
        READ_ACCUM_2    = 18,    // Present res_rad=row_idx+1 for overflow row
        ACCUMULATE_2    = 10,
        WAIT_NEXT       = 11,
        READ_MAX        = 19,    // Present res_rad=max_idx; BSRAM registers output at clk edge
        FIND_MAX        = 12,
        COMPUTE_SCALE   = 13,
        READ_QUANTIZE   = 20,    // Prime quantize pipeline: present res_rad=quant_in_idx
        QUANTIZE        = 14,
        READ_OUTPUT_Y   = 21,    // Prime output pipeline: present res_rad=output_y_idx
        SUM_PARTIAL     = 22,    // A2: stage-1 adder — register pairwise pe_out sums
        PREP_ACCUM      = 23,    // A3: register res_dout+sum_current_row_reg before BSRAM write
        PREP_MAX        = 24,    // A4: register abs(res_dout) before FIND_MAX comparison
        READ_X_TILE     = 25,    // A6: prime x_mem BSRAM read before LOAD_X_TILE
        OUTPUT_Y        = 15,
        DONE_STATE      = 16
    } state, next_state;

    // Index widths
    localparam ROW_IDX_WIDTH  = $clog2(MAX_ROWS);
    localparam COL_IDX_WIDTH  = $clog2(MAX_COLUMNS);
    localparam TILE_IDX_WIDTH = $clog2(MAX_COLUMNS/TILE_SIZE + 1);

    // Internal registers
    logic [ROW_IDX_WIDTH-1:0] row_idx;
    logic [TILE_IDX_WIDTH-1:0] tile_idx;
    logic signed [DATA_WIDTH-1:0] w_latched [0:TILE_SIZE-1];
    
    
    // PE connections
    logic signed [DATA_WIDTH-1:0] x_in [0:TILE_SIZE-1];
    logic signed [2*DATA_WIDTH-1:0] pe_out [0:TILE_SIZE-1];
    
    // Tile boundary detection
    logic last_in_row;
    logic row_overflow;
    logic [COL_IDX_WIDTH-1:0] col_start;
    logic [COL_IDX_WIDTH-1:0] num_current_row;
    
    // Split accumulations (combinational, fed into pipeline regs)
    logic signed [4*DATA_WIDTH-1:0] sum_current_row;
    logic signed [4*DATA_WIDTH-1:0] sum_next_row;

    // Pipeline registers: tile geometry (registered in WAIT_TILE, used in ACCUMULATE/WAIT_NEXT)
    logic [COL_IDX_WIDTH-1:0] num_current_row_reg;
    logic last_in_row_reg;
    logic row_overflow_reg;
    // A1: pe_valid removed. x_current_tile[j] is zeroed for j >= num_current_row in LOAD_X_TILE,
    // so pe_out[j] = w*0 = 0 for out-of-range elements. No adder-tree gating needed.

    // Pipeline registers: PE sums (registered in WAIT_PE, used in ACCUMULATE)
    logic signed [4*DATA_WIDTH-1:0] sum_current_row_reg;
    logic signed [4*DATA_WIDTH-1:0] sum_next_row_reg;
    // A2: stage-1 partial sums (4 pairwise pe_out sums, registered in SUM_PARTIAL)
    logic signed [4*DATA_WIDTH-1:0] partial_sum_reg [0:TILE_SIZE/2-1];
    // A3: pre-computed accumulator result (registered in PREP_ACCUM, written in ACCUMULATE)
    logic signed [4*DATA_WIDTH-1:0] accum_result_reg;
    // A4: registered abs(res_dout) — breaks BSRAM→abs→compare→max_abs_reg chain in FIND_MAX
    logic signed [4*DATA_WIDTH-1:0] current_abs_reg;

    // Scale computation
    logic signed [4*DATA_WIDTH-1:0] reciprocal_scale;
    logic scale_ready;
    logic signed [4*DATA_WIDTH-1:0] max_abs_reg;
    logic signed [4*DATA_WIDTH-1:0] current_abs;
    logic [ROW_IDX_WIDTH-1:0] max_idx;

    // Quantization
    logic signed [4*DATA_WIDTH-1:0] int32_value;
    logic signed [DATA_WIDTH-1:0] int8_value;
    logic [ROW_IDX_WIDTH-1:0] quant_in_idx, quant_out_idx;
    logic quant_valid_in, quant_valid_out;
    logic [ROW_IDX_WIDTH-1:0] output_y_idx;

    // Accumulator BRAM Interface
    logic [31:0] res_dout, res_din;
    logic [ROW_IDX_WIDTH-1:0] res_rad, res_wad;
    logic res_wre;
    
    // Current x tile for PE computation
    logic signed [DATA_WIDTH-1:0] x_current_tile [0:TILE_SIZE-1];

    // X-Vector RAM Interface
    logic [31:0] x_mem_dout, x_mem_din;
    logic [9:0] x_mem_wad, x_mem_rad;
    logic x_mem_wre;

    // X loading/storing registers
    logic signed [DATA_WIDTH-1:0] x_latched [0:TILE_SIZE-1];
    logic [9:0] x_store_idx;      // B1: Global word index during STORE_X (4 elems/word)
    logic [0:0] x_store_elem;     // B1: Word counter within tile during STORE_X (0-1)
    logic [0:0] x_load_elem;      // B1: Word counter during LOAD_X_TILE (0-1)

    // Bias storing registers
    logic signed [DATA_WIDTH-1:0] bias_latched [0:TILE_SIZE-1];
    logic [ROW_IDX_WIDTH-1:0] bias_store_idx;   // Row index during STORE_BIAS
    logic [2:0] bias_store_elem;                 // Element within tile during STORE_BIAS
    
    // X/Bias loading counters
    logic [9:0] x_load_tile_count;
    logic [9:0] bias_load_tile_count;
    logic [9:0] total_x_tiles;
    logic [9:0] total_bias_tiles;
    
    // Y output counters
    logic [9:0] y_output_tile_count;
    logic [9:0] total_y_tiles;
    logic [2:0] y_elem_idx;

    // ================= Combinational Logic =================
    
    // Calculate starting column and valid elements
    localparam COL_START_WIDTH = TILE_IDX_WIDTH + $clog2(TILE_SIZE);
    logic [COL_START_WIDTH-1:0] col_start_full;
    assign col_start_full = tile_idx * TILE_SIZE;
    assign col_start = col_start_full[COL_IDX_WIDTH-1:0];
    assign num_current_row = (col_start < cols[COL_IDX_WIDTH-1:0]) ? 
                           ((int'(col_start) + TILE_SIZE <= cols[COL_IDX_WIDTH-1:0]) ? TILE_SIZE[9:0] : cols[COL_IDX_WIDTH-1:0] - col_start) : 
                           0; 
    
    // x input selection - use current tile loaded from x_mem
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            x_in[i] = x_current_tile[i];
        end
    end

    // A1: Unconditional adder tree — no pe_valid gating.
    // x_current_tile[j] = 0 for j >= num_current_row (zeroed in LOAD_X_TILE), so
    // pe_out[j] = w*0 = 0 for invalid elements. sum_next_row is always 0 because
    // overflow x-padding elements are also zero (x buffer padded to tile boundary).
    always_comb begin
        sum_current_row = '0;
        sum_next_row = '0;  // Always 0 after A1

        for (int j = 0; j < TILE_SIZE; j++) begin
            logic signed [4*DATA_WIDTH-1:0] extended_out;
            extended_out = {{(2*DATA_WIDTH){pe_out[j][2*DATA_WIDTH-1]}}, pe_out[j]};
            sum_current_row += extended_out;
        end
    end

    assign last_in_row = (int'(col_start) + TILE_SIZE >= cols[COL_IDX_WIDTH-1:0]);
    assign row_overflow = (col_start < cols[COL_IDX_WIDTH-1:0]) && (int'(col_start) + TILE_SIZE > cols[COL_IDX_WIDTH-1:0]);

    // Absolute value logic (using BRAM output)
    assign current_abs = ($signed(res_dout) >= 0) ? $signed(res_dout) : -$signed(res_dout);
    
    // ================ BRAM Connections ================

    // X-Vector RAM — BSRAM (Gowin_SDPB_32) with 1-cycle synchronous read latency.
    // A6: Replaces LUTRAM (Gowin_RAM16SDP) to eliminate the 6-level MUX2 output cascade
    // (128 x RAM16SDP4 blocks) that dominated the critical path at 84 MHz Fmax.
    // READ_X_TILE state primes the address one cycle before LOAD_X_TILE captures dout.
    Gowin_SDPB_32 x_mem (
        .dout(x_mem_dout),
        .clk(clk),
        .wre(x_mem_wre),
        .wad(x_mem_wad),
        .di(x_mem_din),
        .rad(x_mem_rad)
    );

    // Accumulator RAM — BSRAM (Gowin_SDPB_32) with 1-cycle synchronous read latency.
    // Replaces LUTRAM (Gowin_RAM16SDP) to eliminate the 6-level MUX2 output cascade
    // (640 x RAM16SDP4 blocks) that dominated the critical path at 47 MHz Fmax.
    // READ_xxx states inserted into the FSM handle the 1-cycle address-to-data latency.
    Gowin_SDPB_32 res_mem (
        .dout(res_dout),
        .clk(clk),
        .wre(res_wre),
        .wad(res_wad),
        .di(res_din),
        .rad(res_rad)
    );

    // X-Vector RAM Control Muxing
    // Gowin_SDPB_32 has synchronous (registered) read: x_mem_dout reflects the address
    // presented on the PREVIOUS clock edge. READ_X_TILE presents addr[0] so LOAD_X_TILE
    // can capture immediately; LOAD_X_TILE presents addr[x_load_elem+1] for next iteration.
    always_comb begin
        x_mem_wre = 0;
        x_mem_wad = 0;
        x_mem_din = 0;
        x_mem_rad = 0;

        case(state)
            STORE_X: begin
                // B1: Pack 4 int8 elements per 32-bit word
                x_mem_wad = x_store_idx;
                x_mem_wre = 1;
                if (x_store_elem == 0)
                    x_mem_din = {x_latched[3], x_latched[2], x_latched[1], x_latched[0]};
                else
                    x_mem_din = {x_latched[7], x_latched[6], x_latched[5], x_latched[4]};
            end
            READ_X_TILE: begin
                // A6+B1: prime word addr[0]; tile_idx*2 = word base for packed 4:1 x_mem
                x_mem_rad = {2'b0, tile_idx, 1'b0} + {9'b0, x_load_elem};
            end
            LOAD_X_TILE: begin
                // A6+B1: present NEXT word address for pipelined BSRAM capture
                x_mem_rad = {2'b0, tile_idx, 1'b0} + {9'b0, x_load_elem} + 10'd1;
            end
            // B2: Prefetch next x tile during accumulate pipeline
            // x_mem is idle from WAIT_PE through WAIT_NEXT — use PREP_ACCUM and
            // ACCUMULATE to prime and capture word 0, then capture word 1 in
            // WAIT_NEXT (normal) or READ_ACCUM_2 (overflow path).
            PREP_ACCUM: begin
                // Prime word 0 of NEXT tile
                if (last_in_row_reg)
                    x_mem_rad = 10'd0;  // Next row starts at tile 0, word 0
                else
                    x_mem_rad = {2'b0, tile_idx + 1'b1, 1'b0};  // Next tile, word 0
            end
            ACCUMULATE: begin
                // Prime word 1 of NEXT tile (word 0 data available now)
                if (last_in_row_reg)
                    x_mem_rad = 10'd1;  // tile 0, word 1
                else
                    x_mem_rad = {2'b0, tile_idx + 1'b1, 1'b0} + 10'd1;  // Next tile, word 1
            end
            default: ;
        endcase
    end

    // Accumulator RAM Control Muxing
    // Gowin_SDPB_32 has synchronous (registered) read: res_dout reflects the address
    // presented on the PREVIOUS clock edge. READ_xxx states present the address one
    // cycle before the state that consumes res_dout.
    always_comb begin
        res_wre = 0;
        res_wad = 0;
        res_din = 0;
        res_rad = 0;

        case(state)
            STORE_BIAS: begin
                res_wad = bias_store_idx;
                res_wre = 1;
                // Sign-extend int8 bias to int32 and store as initial accumulator value
                res_din = {{(3*DATA_WIDTH){bias_latched[bias_store_elem][DATA_WIDTH-1]}}, bias_latched[bias_store_elem]};
            end
            READ_ACCUM: begin
                res_rad = row_idx;          // Present address; BSRAM registers at clock edge
            end
            ACCUMULATE: begin
                // A3: accum_result_reg = res_dout + sum_current_row_reg, pre-computed in PREP_ACCUM
                res_wad = row_idx;
                res_wre = 1;
                res_din = accum_result_reg;
            end
            READ_ACCUM_2: begin
                res_rad = row_idx + 1;      // Present overflow row address
            end
            ACCUMULATE_2: begin
                // res_dout = mem[row_idx+1] registered from READ_ACCUM_2
                res_wad = row_idx + 1;
                res_wre = 1;
                res_din = $signed(res_dout) + sum_next_row_reg;
            end
            READ_MAX: begin
                res_rad = max_idx;          // Present address; BSRAM registers at clock edge
            end
            // FIND_MAX: res_dout holds mem[max_idx] registered from READ_MAX — no res_rad needed
            READ_QUANTIZE: begin
                res_rad = quant_in_idx;     // Prime pipeline: present address 0 (quant_in_idx=0)
            end
            QUANTIZE: begin
                // Pre-fetch next element: quant_in_idx not yet incremented in comb
                res_rad = quant_in_idx + 1;
                if (quant_valid_out) begin
                    res_wre = 1;
                    res_wad = quant_out_idx;
                    res_din = {{(3*DATA_WIDTH){int8_value[DATA_WIDTH-1]}}, int8_value};
                end
            end
            READ_OUTPUT_Y: begin
                res_rad = output_y_idx;     // Prime pipeline: present first output address
            end
            OUTPUT_Y: begin
                // Pre-fetch next address while capturing current res_dout
                if (y_tile_ready || !y_tile_valid)
                    res_rad = output_y_idx + 1; // output_y_idx not yet incremented in comb
                else
                    res_rad = output_y_idx;     // Stalled: re-present current address
            end
            default: ;
        endcase
    end
    
    // ================ Next State Logic ================

    always_comb begin : next_state_logic
        next_state = IDLE;
        case (state)
            IDLE: next_state = start ? LOAD_X : IDLE;

            LOAD_X: next_state = x_tile_valid ? STORE_X : LOAD_X;

            STORE_X: begin
                if (x_store_elem == 1) begin  // B1: 2 words per tile (4 elems/word)
                    if (x_load_tile_count + 1 >= total_x_tiles)
                        next_state = LOAD_BIAS;
                    else
                        next_state = LOAD_X;
                end else
                    next_state = STORE_X;
            end

            LOAD_BIAS: next_state = bias_tile_valid ? STORE_BIAS : LOAD_BIAS;

            STORE_BIAS: begin
                if (bias_store_elem == TILE_SIZE - 1) begin
                    if (bias_load_tile_count + 1 >= total_bias_tiles)
                        next_state = READ_X_TILE;   // A6: prime BSRAM before LOAD_X_TILE
                    else
                        next_state = LOAD_BIAS;
                end else
                    next_state = STORE_BIAS;
            end

            LOAD_X_TILE: begin
                next_state = (x_load_elem == 1) ? WAIT_TILE : LOAD_X_TILE;  // B1: 2 words per tile
            end

            WAIT_TILE:    next_state = w_valid ? WAIT_PE : WAIT_TILE;
            WAIT_PE:      next_state = SUM_PARTIAL;    // A2: stage-1 pairwise sums
            SUM_PARTIAL:  next_state = READ_ACCUM;     // BSRAM: prime accumulator read
            READ_ACCUM:   next_state = PREP_ACCUM;     // A3: register the add result
            PREP_ACCUM:   next_state = ACCUMULATE;
            ACCUMULATE:   next_state = row_overflow_reg ? READ_ACCUM_2 : WAIT_NEXT;
            READ_ACCUM_2: next_state = ACCUMULATE_2;
            ACCUMULATE_2: next_state = WAIT_NEXT;

            WAIT_NEXT: begin
                if (last_in_row_reg) begin
                    if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1)
                        next_state = WAIT_TILE;        // B2: x_current_tile already prefetched
                    else
                        next_state = READ_MAX;         // BSRAM: prime first FIND_MAX read
                end else
                    next_state = WAIT_TILE;            // B2: x_current_tile already prefetched
            end

            READ_X_TILE: next_state = LOAD_X_TILE;
            READ_MAX:   next_state = PREP_MAX;
            PREP_MAX:   next_state = FIND_MAX;
            FIND_MAX:   next_state = int'(max_idx) < int'(rows) - 1 ? READ_MAX : COMPUTE_SCALE;
            COMPUTE_SCALE: next_state = scale_ready ? READ_QUANTIZE : COMPUTE_SCALE;
            READ_QUANTIZE: next_state = QUANTIZE;
            QUANTIZE: next_state = quant_valid_out ? (quant_out_idx < rows[ROW_IDX_WIDTH-1:0] - 1 ? QUANTIZE : READ_OUTPUT_Y) : QUANTIZE;
            READ_OUTPUT_Y: next_state = OUTPUT_Y;
            OUTPUT_Y: next_state = (y_output_tile_count >= total_y_tiles && y_tile_ready) ? DONE_STATE : OUTPUT_Y;
            DONE_STATE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    always_ff@(posedge clk or posedge rst) begin
        if(rst) state <= IDLE;
        else begin
            state <= next_state;
        end
    end

    // ================= Sequential Logic =================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            w_ready <= 0;
            tile_done <= 0;
            done <= 0;
            row_idx <= 0;
            tile_idx <= 0;
            max_idx <= 0;
            max_abs_reg <= 0;
            quant_in_idx <= 0;
            quant_out_idx <= 0;
            output_y_idx <= 0;
            x_tile_ready <= 0;
            bias_tile_ready <= 0;
            y_tile_valid <= 0;
            y_tile_idx <= 0;
            x_load_tile_count <= 0;
            bias_load_tile_count <= 0;
            y_output_tile_count <= 0;
            total_x_tiles <= 0;
            total_bias_tiles <= 0;
            total_y_tiles <= 0;
            y_elem_idx <= 0;
            quant_valid_in <= 0;
            int32_value <= 0;
            x_store_idx <= 0;
            x_store_elem <= 0;
            x_load_elem <= 0;
            bias_store_idx <= 0;
            bias_store_elem <= 0;
            for (int i = 0; i < TILE_SIZE; i++) begin
                x_current_tile[i] <= 0;
                w_latched[i] <= 0;
                y_tile_out[i] <= 0;
                x_latched[i] <= 0;
                bias_latched[i] <= 0;
            end
            num_current_row_reg <= 0;
            last_in_row_reg <= 0;
            row_overflow_reg <= 0;
            sum_current_row_reg <= 0;
            sum_next_row_reg <= 0;
            for (int p = 0; p < TILE_SIZE/2; p++)
                partial_sum_reg[p] <= '0;
            accum_result_reg <= '0;
            current_abs_reg  <= '0;
        end else begin
            case (state)
                IDLE: begin
                    w_ready <= 0;
                    tile_done <= 0;
                    done <= 0;
                    row_idx <= 0;
                    if (start) begin
                        total_x_tiles <= (cols + TILE_SIZE - 1) / TILE_SIZE;
                        total_bias_tiles <= (rows + TILE_SIZE - 1) / TILE_SIZE;
                        total_y_tiles <= (rows + TILE_SIZE - 1) / TILE_SIZE;
                        x_load_tile_count <= 0;
                        bias_load_tile_count <= 0;
                        y_output_tile_count <= 0;
                        x_store_idx <= 0;
                        x_store_elem <= 0;
                        bias_store_idx <= 0;
                        bias_store_elem <= 0;
                        x_tile_ready <= 1;
                        // FIX: Reset output/quantization indices between GEMV invocations
                        output_y_idx <= 0;
                        y_elem_idx <= 0;
                        quant_in_idx <= 0;
                        quant_out_idx <= 0;
                    end
                end
                
                // ---- X Loading: receive tile, then store elements to x_mem ----
                LOAD_X: begin
                    x_tile_ready <= 1;
                    if (x_tile_valid) begin
                        // Latch the tile data for element-by-element writing
                        for (int i = 0; i < TILE_SIZE; i++)
                            x_latched[i] <= x_tile_in[i];
                        x_store_elem <= 0;
                        x_tile_ready <= 0; // Stop accepting until stored
                    end
                end

                STORE_X: begin
                    // B1: Write one packed word (4 int8 elements) per cycle to x_mem
                    x_store_idx <= x_store_idx + 1;
                    x_store_elem <= x_store_elem + 1;
                    if (x_store_elem == 1) begin  // B1: 2 words per tile (4 elems/word)
                        x_load_tile_count <= x_load_tile_count + 1;
                        if (x_load_tile_count + 1 >= total_x_tiles) begin
                            x_tile_ready <= 0;
                            bias_tile_ready <= 1;
                        end else begin
                            x_tile_ready <= 1; // Ready for next tile
                        end
                    end
                end
                
                // ---- Bias Loading: receive tile, then store to accumulator RAM ----
                LOAD_BIAS: begin
                    bias_tile_ready <= 1;
                    if (bias_tile_valid) begin
                        // Latch the bias tile for element-by-element writing
                        for (int i = 0; i < TILE_SIZE; i++)
                            bias_latched[i] <= bias_tile_in[i];
                        bias_store_elem <= 0;
                        bias_tile_ready <= 0; // Stop accepting until stored
                    end
                end

                STORE_BIAS: begin
                    // Write one bias element per cycle to res_mem (accumulator)
                    // Each element is sign-extended int8 -> int32
                    bias_store_idx <= bias_store_idx + 1;
                    bias_store_elem <= bias_store_elem + 1;
                    if (bias_store_elem == TILE_SIZE - 1) begin
                        bias_load_tile_count <= bias_load_tile_count + 1;
                        if (bias_load_tile_count + 1 >= total_bias_tiles) begin
                            bias_tile_ready <= 0;
                            // Prepare for weight processing: reset counters
                            row_idx <= 0;
                            tile_idx <= '0;
                            max_idx <= '0;
                            max_abs_reg <= '0;
                        end else begin
                            bias_tile_ready <= 1; // Ready for next tile
                        end
                    end
                end

                // ---- Load x tile from x_mem for current tile_idx ----
                LOAD_X_TILE: begin
                    // B1+A1: Unpack 4 int8 elements from packed 32-bit BSRAM word
                    // Zero-mask elements beyond num_current_row (A1)
                    if (x_load_elem == 0) begin
                        // Word 0 → elements [0:3]
                        x_current_tile[0] <= (10'd0 < num_current_row) ? x_mem_dout[0*8 +: 8] : '0;
                        x_current_tile[1] <= (10'd1 < num_current_row) ? x_mem_dout[1*8 +: 8] : '0;
                        x_current_tile[2] <= (10'd2 < num_current_row) ? x_mem_dout[2*8 +: 8] : '0;
                        x_current_tile[3] <= (10'd3 < num_current_row) ? x_mem_dout[3*8 +: 8] : '0;
                    end else begin
                        // Word 1 → elements [4:7]
                        x_current_tile[4] <= (10'd4 < num_current_row) ? x_mem_dout[0*8 +: 8] : '0;
                        x_current_tile[5] <= (10'd5 < num_current_row) ? x_mem_dout[1*8 +: 8] : '0;
                        x_current_tile[6] <= (10'd6 < num_current_row) ? x_mem_dout[2*8 +: 8] : '0;
                        x_current_tile[7] <= (10'd7 < num_current_row) ? x_mem_dout[3*8 +: 8] : '0;
                    end
                    x_load_elem <= x_load_elem + 1;
                    if (x_load_elem == 1) begin  // B1: 2 words per tile
                        x_load_elem <= 0;
                    end
                end

                // ---- Weight processing ----
                WAIT_TILE: begin
                    w_ready <= 1;
                    tile_done <= 0;
                    // Register tile geometry so ACCUMULATE/WAIT_NEXT see FFs, not
                    // a long combinational chain from tile_idx through the multiply.
                    // A1: pe_valid removed; x zeroing in LOAD_X_TILE handles masking.
                    num_current_row_reg <= num_current_row;
                    last_in_row_reg     <= last_in_row;
                    row_overflow_reg    <= row_overflow;
                    if (w_valid) begin
                        for (int i = 0; i < TILE_SIZE; i++) w_latched[i] <= w_tile_row_in[i];
                        w_ready <= 0;
                    end
                end

                // Idea 3: register sums so ACCUMULATE sees FF+FF→add→BRAM, not the
                // full pe_out→gate→adder→add→BRAM combinational chain
                WAIT_PE: begin
                    tile_done <= 0;
                end

                SUM_PARTIAL: begin
                    // A2 stage 1: 4 pairwise pe_out additions — halves adder depth to ~4 logic levels
                    for (int p = 0; p < TILE_SIZE/2; p++) begin
                        partial_sum_reg[p] <=
                            {{(2*DATA_WIDTH){pe_out[2*p][2*DATA_WIDTH-1]}}, pe_out[2*p]}
                          + {{(2*DATA_WIDTH){pe_out[2*p+1][2*DATA_WIDTH-1]}}, pe_out[2*p+1]};
                    end
                end

                READ_ACCUM: begin
                    // A2 stage 2: combine 4 partial sums (→ ~3 logic levels) + prime BSRAM read
                    sum_current_row_reg <= partial_sum_reg[0] + partial_sum_reg[1]
                                        + partial_sum_reg[2] + partial_sum_reg[3];
                    sum_next_row_reg    <= '0;  // Always 0 after A1
                end

                PREP_ACCUM: begin
                    // A3: register res_dout+sum — breaks res_dout→add→BSRAM.di carry-chain path
                    accum_result_reg <= $signed(res_dout) + sum_current_row_reg;
                end

                ACCUMULATE: begin
                    tile_done <= !row_overflow_reg;
                    // B2: Capture prefetched word 0 of next tile (primed in PREP_ACCUM)
                    x_current_tile[0] <= x_mem_dout[0*8 +: 8];
                    x_current_tile[1] <= x_mem_dout[1*8 +: 8];
                    x_current_tile[2] <= x_mem_dout[2*8 +: 8];
                    x_current_tile[3] <= x_mem_dout[3*8 +: 8];
                end
                
                // B2: On overflow path, capture word 1 here (primed in ACCUMULATE,
                // available after READ_ACCUM_2 clock edge — but READ_ACCUM_2 also
                // presents res_rad for overflow row, so x_mem_dout is still from
                // ACCUMULATE's x_mem_rad. Capture it here.)
                READ_ACCUM_2: begin
                    x_current_tile[4] <= x_mem_dout[0*8 +: 8];
                    x_current_tile[5] <= x_mem_dout[1*8 +: 8];
                    x_current_tile[6] <= x_mem_dout[2*8 +: 8];
                    x_current_tile[7] <= x_mem_dout[3*8 +: 8];
                end

                ACCUMULATE_2: begin
                    tile_done <= 1;
                end

                WAIT_NEXT: begin
                    tile_done <= 0;
                    if (last_in_row_reg) begin
                        tile_idx <= '0;
                        if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1) row_idx <= row_idx + 1;
                    end else begin
                        tile_idx <= tile_idx + 1;
                    end
                    // B2: Capture prefetched word 1 — only on non-overflow path.
                    // On overflow path, word 1 was already captured in READ_ACCUM_2
                    // (by WAIT_NEXT, x_mem_dout reflects stale default addr from ACCUMULATE_2).
                    if (!row_overflow_reg) begin
                        x_current_tile[4] <= x_mem_dout[0*8 +: 8];
                        x_current_tile[5] <= x_mem_dout[1*8 +: 8];
                        x_current_tile[6] <= x_mem_dout[2*8 +: 8];
                        x_current_tile[7] <= x_mem_dout[3*8 +: 8];
                    end
                end

                PREP_MAX: begin
                    // A4: register abs(res_dout) — isolates BSRAM tC2Q+abs from compare chain
                    current_abs_reg <= current_abs;
                end

                FIND_MAX: begin
                    // A4: current_abs_reg (FF) → 32-bit compare → max_abs_reg (FF)
                    if (current_abs_reg > max_abs_reg) max_abs_reg <= current_abs_reg;
                    if (int'(max_idx) < int'(rows) - 1) max_idx <= max_idx + 1;
                    else if (max_abs_reg == 0) max_abs_reg <= 1;
                end

                COMPUTE_SCALE: begin
                    if (scale_ready) begin
                        quant_in_idx <= 0;
                        quant_out_idx <= 0;
                        $display("[DBG] COMPUTE_SCALE done: rows=%0d max_abs=%0d reciprocal=0x%h", rows, $signed(max_abs_reg), reciprocal_scale);
                    end
                end

                QUANTIZE: begin
                    quant_valid_in <= 0;
                    if (int'(quant_in_idx) < MAX_ROWS) begin
                        int32_value <= res_dout;
                        quant_in_idx <= quant_in_idx + 1;
                        quant_valid_in <= (quant_in_idx < rows[ROW_IDX_WIDTH-1:0]);
                        if (int'(quant_in_idx) < 4)
                            $display("[DBG] QUANTIZE feed idx=%0d res_dout=%0d", quant_in_idx, $signed(res_dout));
                    end
                    if (quant_valid_out) begin
                        if (int'(quant_out_idx) < MAX_ROWS) quant_out_idx <= quant_out_idx + 1;
                        if (int'(quant_out_idx) < 4)
                            $display("[DBG] QUANTIZE out idx=%0d int8=%0d", quant_out_idx, $signed(int8_value));
                    end
                end
                
                OUTPUT_Y: begin
                    // Output y values tile by tile
                    if (y_tile_ready || !y_tile_valid) begin
                        // Read from BRAM and output
                        y_tile_out[y_elem_idx] <= res_dout[DATA_WIDTH-1:0];
                        output_y_idx <= output_y_idx + 1;
                        y_elem_idx <= y_elem_idx + 1;
                        
                        if (y_elem_idx == TILE_SIZE - 1) begin
                            y_tile_valid <= 1;
                            y_tile_idx <= y_output_tile_count;
                            y_output_tile_count <= y_output_tile_count + 1;
                            y_elem_idx <= 0;
                        end
                    end
                    
                    if (y_tile_ready && y_tile_valid) begin
                        y_tile_valid <= 0;
                    end
                end

                DONE_STATE: begin
                    done <= 1;
                    y_tile_valid <= 0;
                end

                default: ;
            endcase
        end
    end

    // PE Instantiation
    generate
        for (genvar i = 0; i < TILE_SIZE; i++) begin : pe_array
            pe #(.DATA_WIDTH(DATA_WIDTH)) pe_inst (.clk(clk), .rst(rst), .w(w_latched[i]), .x(x_in[i]), .y(pe_out[i]));
        end
    endgenerate

    // Submodules
    scale_calculator scale_inst (.clk(clk), .reset_n(~rst), .max_abs(max_abs_reg), .start(state == COMPUTE_SCALE), .reciprocal_scale(reciprocal_scale), .ready(scale_ready));
    quantizer_pipeline quantize_inst (.clk(clk), .reset_n(~rst), .int32_value(int32_value), .reciprocal_scale(reciprocal_scale), .valid_in(quant_valid_in), .int8_value(int8_value), .valid_out(quant_valid_out));

endmodule
