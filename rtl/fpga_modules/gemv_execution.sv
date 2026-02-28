// GEMV Execution Module - Tile Bridging Architecture
// Bridges TILE_ELEMS-element buffer tiles to GEMV unit tiles (GEMV_TILE_SIZE = TILE_ELEMS)
// Uses local buffer + carry buffer for seamless tile conversion
//
// Operation Flow:
// 1. Pulse gemv_start so GEMV unit enters LOAD_X
// 2. Stream x vector: load TILE_ELEMS-elem buffer tiles, extract GEMV tiles
// 3. Stream bias vector: same bridging pattern
// 4. Stream weight matrix: same bridging, with per-row tracking
// 5. Receive y result tiles, pack into TILE_ELEMS-elem buffer tiles, write back

module gemv_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32,
    parameter MAX_ROWS = 784,
    parameter MAX_COLS = 784
)(
    input logic clk,
    input logic rst,

    // Control interface
    input logic start,
    input logic [4:0] dest_buffer_id,
    input logic [4:0] w_buffer_id,
    input logic [4:0] x_buffer_id,
    input logic [4:0] b_buffer_id,
    input logic [9:0] cols,
    input logic [9:0] rows,
    output logic done,

    // Buffer controller interface - vector reads
    output logic vec_read_enable,
    output logic [4:0] vec_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    input logic vec_read_valid,

    // Buffer controller interface - matrix reads
    output logic mat_read_enable,
    output logic [4:0] mat_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1],
    input logic mat_read_valid,

    // Buffer controller interface - vector writes (for results)
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],

    // Result output
    output logic signed [DATA_WIDTH-1:0] result [0:TILE_ELEMS-1]
);

    localparam GEMV_TILE_SIZE = TILE_ELEMS;

    // ==================== FSM States ====================
    typedef enum logic [3:0] {
        IDLE,
        LOAD_X_BUF,       // Load buffer tile for X vector
        STREAM_X,          // Extract and send GEMV-sized X tiles
        LOAD_BIAS_BUF,     // Load buffer tile for bias vector
        STREAM_BIAS,       // Extract and send GEMV-sized bias tiles
        LOAD_W_BUF,        // Load buffer tile for weight matrix
        FEED_W,            // Feed GEMV-sized weight tiles
        RECV_Y,            // Receive y tiles from GEMV unit
        WRITE_RESULT,      // Write result buffer tile to buffer controller
        SHIFT_OVERFLOW,    // Move overflow data into result buffer start
        COMPLETE
    } state_t;

    state_t state;

    // ==================== Tile Count Registers ====================
    logic [9:0] total_gemv_x_tiles;    // ceil(cols/6)
    logic [9:0] total_gemv_bias_tiles; // ceil(rows/6)
    logic [9:0] gemv_tiles_per_row;    // ceil(cols/6) for weights
    logic [9:0] buf_tiles_per_row;     // ceil(cols/32) for weights
    logic [9:0] total_buf_x_tiles;     // ceil(cols/32)
    logic [9:0] total_buf_bias_tiles;  // ceil(rows/32)
    logic [9:0] total_result_tiles;    // ceil(rows/32)

    // ==================== Local Buffer (current buffer tile) ====================
    logic signed [DATA_WIDTH-1:0] local_buf [0:TILE_ELEMS-1];
    logic [5:0] buf_idx;  // Read position in local_buf (0..32)

    // ==================== Carry Buffer (spans tile boundaries) ====================
    // Max carry = GEMV_TILE_SIZE-1 = 5 elements (sized to GEMV_TILE_SIZE to avoid static OOB)
    logic signed [DATA_WIDTH-1:0] carry_buf [0:GEMV_TILE_SIZE];
    logic [2:0] carry_count;

    // ==================== Streaming Counters ====================
    logic [9:0] gemv_tiles_sent;   // GEMV tiles sent in current phase (x or bias)
    logic [9:0] buf_tiles_read;    // Buffer tiles read in current phase (x or bias)
    logic read_requested;          // Tracks if buffer read is in flight

    // ==================== Weight Row Tracking ====================
    logic [9:0] w_row;
    logic [9:0] w_gemv_in_row;
    logic [9:0] w_buf_in_row;
    logic [9:0] w_total_sent;
    logic [9:0] total_w_gemv;      // rows * gemv_tiles_per_row

    // ==================== GEMV Unit Interface ====================
    logic gemv_start, gemv_done, gemv_w_ready;
    logic w_valid;

    logic x_tile_valid_to_gemv;
    logic signed [DATA_WIDTH-1:0] x_tile_to_gemv [0:GEMV_TILE_SIZE-1];
    logic x_tile_ready_from_gemv;
    logic [9:0] x_tile_idx_to_gemv;

    logic bias_tile_valid_to_gemv;
    logic signed [DATA_WIDTH-1:0] bias_tile_to_gemv [0:GEMV_TILE_SIZE-1];
    logic bias_tile_ready_from_gemv;
    logic [9:0] bias_tile_idx_to_gemv;

    logic y_tile_valid_from_gemv;
    logic signed [DATA_WIDTH-1:0] y_tile_from_gemv [0:GEMV_TILE_SIZE-1];
    logic y_tile_ready_to_gemv;
    logic [9:0] y_tile_idx_from_gemv;

    logic signed [DATA_WIDTH-1:0] w_tile_to_gemv [0:GEMV_TILE_SIZE-1];

    // ==================== Result Handling ====================
    logic signed [DATA_WIDTH-1:0] result_buffer [0:TILE_ELEMS-1];
    logic [5:0] result_buf_idx;
    logic signed [DATA_WIDTH-1:0] overflow_buf [0:GEMV_TILE_SIZE];
    logic [5:0] overflow_count; // Extended to 6 bits to match result_buf_idx width
    logic gemv_done_latched;
    logic [9:0] result_tiles_written;

    // ==================== Combinational: Available data count ====================
    logic [5:0] data_avail;
    assign data_avail = {3'b0, carry_count} + (TILE_ELEMS[5:0] - buf_idx);

    // ==================== Combinational: Build GEMV tile from carry + local_buf ====================
    logic signed [DATA_WIDTH-1:0] built_tile [0:GEMV_TILE_SIZE-1];

    always_comb begin
        for (int i = 0; i < GEMV_TILE_SIZE; i++) begin
            if (i < int'(carry_count)) begin
                built_tile[i] = carry_buf[i];
            end else if (int'(buf_idx) + i - int'(carry_count) < TILE_ELEMS) begin
                built_tile[i] = local_buf[int'(buf_idx) + i - int'(carry_count)];
            end else begin
                built_tile[i] = 8'sd0;
            end
        end
    end

    // Route built_tile to all GEMV tile inputs (only the active one matters)
    always_comb begin
        for (int i = 0; i < GEMV_TILE_SIZE; i++) begin
            x_tile_to_gemv[i] = built_tile[i];
            bias_tile_to_gemv[i] = built_tile[i];
            w_tile_to_gemv[i] = built_tile[i];
        end
    end

    // ==================== Combinational: Valid/Ready signals ====================
    // Valid signals MUST be combinational so data and valid are sampled in sync

    // For X and bias: data is available if we have enough OR it's the last tile (padded)
    logic x_can_send, bias_can_send, w_can_send;
    assign x_can_send = (data_avail >= GEMV_TILE_SIZE) ||
                         (buf_tiles_read >= total_buf_x_tiles && (carry_count > 0 || buf_idx < TILE_ELEMS[5:0]));
    assign bias_can_send = (data_avail >= GEMV_TILE_SIZE) ||
                            (buf_tiles_read >= total_buf_bias_tiles && (carry_count > 0 || buf_idx < TILE_ELEMS[5:0]));
    assign w_can_send = (data_avail >= GEMV_TILE_SIZE) ||
                         (w_buf_in_row >= buf_tiles_per_row && (carry_count > 0 || buf_idx < TILE_ELEMS[5:0]));

    assign x_tile_valid_to_gemv = (state == STREAM_X) && x_tile_ready_from_gemv && x_can_send;
    assign bias_tile_valid_to_gemv = (state == STREAM_BIAS) && bias_tile_ready_from_gemv && bias_can_send;
    assign w_valid = (state == FEED_W) && gemv_w_ready && w_can_send;

    // Y tile ready: accept y tiles during RECV_Y
    assign y_tile_ready_to_gemv = (state == RECV_Y);

    // ==================== Result buffer to output ====================
    always_comb begin
        for (int i = 0; i < TILE_ELEMS; i++) begin
            vec_write_tile[i] = result_buffer[i];
            result[i] = result_buffer[i];
        end
    end

    // ==================== GEMV Unit Instantiation ====================
    gemv_unit_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLUMNS(MAX_COLS),
        .TILE_SIZE(GEMV_TILE_SIZE)
    ) gemv_unit (
        .clk(clk),
        .rst(rst),
        .start(gemv_start),
        .w_ready(gemv_w_ready),
        .w_valid(w_valid),
        .w_tile_row_in(w_tile_to_gemv),
        .x_tile_valid(x_tile_valid_to_gemv),
        .x_tile_in(x_tile_to_gemv),
        .x_tile_ready(x_tile_ready_from_gemv),
        .x_tile_idx(x_tile_idx_to_gemv),
        .bias_tile_valid(bias_tile_valid_to_gemv),
        .bias_tile_in(bias_tile_to_gemv),
        .bias_tile_ready(bias_tile_ready_from_gemv),
        .bias_tile_idx(bias_tile_idx_to_gemv),
        .y_tile_valid(y_tile_valid_from_gemv),
        .y_tile_out(y_tile_from_gemv),
        .y_tile_ready(y_tile_ready_to_gemv),
        .y_tile_idx(y_tile_idx_from_gemv),
        .rows(rows),
        .cols(cols),
        /* verilator lint_off PINCONNECTEMPTY */
        .tile_done(),
        /* verilator lint_on PINCONNECTEMPTY */
        .done(gemv_done)
    );

    // ==================== Helper: advance after GEMV tile handshake ====================
    // Computed combinationally for use in FSM decisions
    logic [5:0] new_buf_idx;
    logic [5:0] new_remaining;
    assign new_buf_idx = buf_idx + GEMV_TILE_SIZE[5:0] - {3'b0, carry_count};
    assign new_remaining = TILE_ELEMS[5:0] - new_buf_idx;

    // ==================== Main FSM ====================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            gemv_start <= 0;
            vec_read_enable <= 0;
            mat_read_enable <= 0;
            vec_write_enable <= 0;
            vec_read_buffer_id <= 0;
            mat_read_buffer_id <= 0;
            vec_write_buffer_id <= 0;
            buf_idx <= 0;
            carry_count <= 0;
            gemv_tiles_sent <= 0;
            buf_tiles_read <= 0;
            read_requested <= 0;
            x_tile_idx_to_gemv <= 0;
            bias_tile_idx_to_gemv <= 0;
            total_gemv_x_tiles <= 0;
            total_gemv_bias_tiles <= 0;
            gemv_tiles_per_row <= 0;
            buf_tiles_per_row <= 0;
            total_buf_x_tiles <= 0;
            total_buf_bias_tiles <= 0;
            total_result_tiles <= 0;
            w_row <= 0;
            w_gemv_in_row <= 0;
            w_buf_in_row <= 0;
            w_total_sent <= 0;
            total_w_gemv <= 0;
            result_buf_idx <= 0;
            overflow_count <= 0;
            gemv_done_latched <= 0;
            result_tiles_written <= 0;
            for (int i = 0; i < TILE_ELEMS; i++) begin
                local_buf[i] <= 0;
                result_buffer[i] <= 0;
            end
            for (int i = 0; i < GEMV_TILE_SIZE; i++) carry_buf[i] <= 0;
            for (int i = 0; i < GEMV_TILE_SIZE; i++) overflow_buf[i] <= 0;
        end else begin
            // Defaults
            done <= 0;
            gemv_start <= 0;
            vec_read_enable <= 0;
            mat_read_enable <= 0;
            vec_write_enable <= 0;

            // Global gemv_done latch — captures 1-cycle pulse in ANY state
            if (gemv_done) gemv_done_latched <= 1;

            case (state)
                // ============================================================
                // IDLE: Calculate tile counts, pulse gemv_start, begin X load
                // ============================================================
                IDLE: begin
                    if (start) begin
                        total_gemv_x_tiles <= (cols + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE;
                        total_gemv_bias_tiles <= (rows + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE;
                        gemv_tiles_per_row <= (cols + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE;
                        buf_tiles_per_row <= (cols + TILE_ELEMS - 1) / TILE_ELEMS;
                        total_buf_x_tiles <= (cols + TILE_ELEMS - 1) / TILE_ELEMS;
                        total_buf_bias_tiles <= (rows + TILE_ELEMS - 1) / TILE_ELEMS;
                        total_result_tiles <= (rows + TILE_ELEMS - 1) / TILE_ELEMS;
                        total_w_gemv <= rows * ((cols + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE);

                        gemv_start <= 1;

                        // Init streaming state
                        gemv_tiles_sent <= 0;
                        buf_tiles_read <= 0;
                        carry_count <= 0;
                        buf_idx <= 0;
                        read_requested <= 0;

                        // Init weight state
                        mat_read_buffer_id <= w_buffer_id;
                        w_row <= 0;
                        w_gemv_in_row <= 0;
                        w_buf_in_row <= 0;
                        w_total_sent <= 0;

                        // Init result state
                        vec_write_buffer_id <= dest_buffer_id;
                        result_tiles_written <= 0;
                        result_buf_idx <= 0;
                        overflow_count <= 0;
                        gemv_done_latched <= 0;
                        for (int i = 0; i < TILE_ELEMS; i++) result_buffer[i] <= 0;
                        for (int i = 0; i < GEMV_TILE_SIZE; i++) overflow_buf[i] <= 0;

                        x_tile_idx_to_gemv <= 0;
                        bias_tile_idx_to_gemv <= 0;

                        // Request first X buffer tile
                        vec_read_buffer_id <= x_buffer_id;
                        vec_read_enable <= 1;
                        read_requested <= 1;
                        state <= LOAD_X_BUF;
                    end
                end

                // ============================================================
                // LOAD_X_BUF: Wait for buffer tile, store in local_buf
                // ============================================================
                LOAD_X_BUF: begin
                    if (!read_requested) begin
                        vec_read_enable <= 1;
                        read_requested <= 1;
                    end
                    if (vec_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++)
                            local_buf[i] <= vec_read_tile[i];
                        buf_idx <= 0;
                        buf_tiles_read <= buf_tiles_read + 1;
                        read_requested <= 0;
                        state <= STREAM_X;
                    end
                end

                // ============================================================
                // STREAM_X: Extract 6-element GEMV tiles from local_buf+carry
                // ============================================================
                STREAM_X: begin
                    // Handshake is combinational (x_tile_valid_to_gemv)
                    // When handshake fires, advance indices here
                    if (x_tile_valid_to_gemv) begin
                        // Tile consumed by GEMV unit this cycle
                        buf_idx <= new_buf_idx;
                        carry_count <= 0;
                        gemv_tiles_sent <= gemv_tiles_sent + 1;
                        x_tile_idx_to_gemv <= gemv_tiles_sent + 1;

                        if (gemv_tiles_sent + 1 >= total_gemv_x_tiles) begin
                            // X streaming complete, start bias
                            vec_read_buffer_id <= b_buffer_id;
                            gemv_tiles_sent <= 0;
                            buf_tiles_read <= 0;
                            carry_count <= 0;
                            buf_idx <= 0;
                            read_requested <= 0;
                            vec_read_enable <= 1;
                            read_requested <= 1;
                            state <= LOAD_BIAS_BUF;
                        end else if (new_buf_idx >= TILE_ELEMS[5:0]) begin
                            // Buffer tile fully consumed, no carry (new_remaining=0)
                            // Load next buffer tile
                            vec_read_enable <= 1;
                            read_requested <= 1;
                            state <= LOAD_X_BUF;
                        end else if (new_remaining < GEMV_TILE_SIZE) begin
                            // Not enough for next GEMV tile
                            if (buf_tiles_read < total_buf_x_tiles) begin
                                // Save remaining as carry, load next buffer tile
                                for (int i = 0; i < GEMV_TILE_SIZE - 1; i++) begin
                                    if (i < TILE_ELEMS - int'(new_buf_idx))
                                        carry_buf[i] <= local_buf[int'(new_buf_idx) + i];
                                end
                                carry_count <= new_remaining[2:0];
                                vec_read_enable <= 1;
                                read_requested <= 1;
                                state <= LOAD_X_BUF;
                            end
                            // else: no more buffer tiles, next cycle x_can_send
                            // handles the partial/padded last tile case
                        end
                        // else: enough data remains, stay in STREAM_X
                    end
                end

                // ============================================================
                // LOAD_BIAS_BUF: Wait for buffer tile, store in local_buf
                // ============================================================
                LOAD_BIAS_BUF: begin
                    if (!read_requested) begin
                        vec_read_enable <= 1;
                        read_requested <= 1;
                    end
                    if (vec_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++)
                            local_buf[i] <= vec_read_tile[i];
                        buf_idx <= 0;
                        buf_tiles_read <= buf_tiles_read + 1;
                        read_requested <= 0;
                        state <= STREAM_BIAS;
                    end
                end

                // ============================================================
                // STREAM_BIAS: Extract 6-element GEMV bias tiles
                // ============================================================
                STREAM_BIAS: begin
                    if (bias_tile_valid_to_gemv) begin
                        buf_idx <= new_buf_idx;
                        carry_count <= 0;
                        gemv_tiles_sent <= gemv_tiles_sent + 1;
                        bias_tile_idx_to_gemv <= gemv_tiles_sent + 1;

                        if (gemv_tiles_sent + 1 >= total_gemv_bias_tiles) begin
                            // Bias streaming complete, start weight streaming
                            carry_count <= 0;
                            buf_idx <= 0;
                            buf_tiles_read <= 0;
                            read_requested <= 0;
                            mat_read_enable <= 1;
                            read_requested <= 1;
                            state <= LOAD_W_BUF;
                        end else if (new_buf_idx >= TILE_ELEMS[5:0]) begin
                            vec_read_enable <= 1;
                            read_requested <= 1;
                            state <= LOAD_BIAS_BUF;
                        end else if (new_remaining < GEMV_TILE_SIZE) begin
                            if (buf_tiles_read < total_buf_bias_tiles) begin
                                for (int i = 0; i < GEMV_TILE_SIZE - 1; i++) begin
                                    if (i < TILE_ELEMS - int'(new_buf_idx))
                                        carry_buf[i] <= local_buf[int'(new_buf_idx) + i];
                                end
                                carry_count <= new_remaining[2:0];
                                vec_read_enable <= 1;
                                read_requested <= 1;
                                state <= LOAD_BIAS_BUF;
                            end
                        end
                    end
                end

                // ============================================================
                // LOAD_W_BUF: Wait for weight buffer tile, store in local_buf
                // ============================================================
                LOAD_W_BUF: begin
                    if (!read_requested) begin
                        mat_read_enable <= 1;
                        read_requested <= 1;
                    end
                    if (mat_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++)
                            local_buf[i] <= mat_read_tile[i];
                        buf_idx <= 0;
                        w_buf_in_row <= w_buf_in_row + 1;
                        read_requested <= 0;
                        state <= FEED_W;
                    end
                end

                // ============================================================
                // FEED_W: Feed 6-element weight tiles to GEMV unit
                // Also handles transition to RECV_Y when all weights are sent
                // ============================================================
                FEED_W: begin
                    // Weight handshake is combinational (w_valid)
                    if (w_valid) begin
                        if (w_row == 0 && w_gemv_in_row == 0 && buf_idx == 0)
                             $display("[DEBUG] GEMV_EXEC: FEED_W first tile local_buf[0]=0x%h, Carry=0x%h, BufIdx=%d", 
                                      local_buf[0], carry_count, buf_idx);
                        buf_idx <= new_buf_idx;
                        carry_count <= 0;
                        w_gemv_in_row <= w_gemv_in_row + 1;
                        w_total_sent <= w_total_sent + 1;

                        if (w_gemv_in_row + 1 >= gemv_tiles_per_row) begin
                            // Row complete
                            w_row <= w_row + 1;
                            w_gemv_in_row <= 0;
                            w_buf_in_row <= 0;
                            carry_count <= 0;

                            if (w_row + 1 >= rows) begin
                                // All weights sent, wait for y tiles
                                state <= RECV_Y;
                            end else begin
                                // Next row: load first buffer tile
                                // Discard any remaining data (it's row padding)
                                mat_read_enable <= 1;
                                read_requested <= 1;
                                state <= LOAD_W_BUF;
                            end
                        end else if (new_buf_idx >= TILE_ELEMS[5:0]) begin
                            // Buffer tile consumed
                            mat_read_enable <= 1;
                            read_requested <= 1;
                            state <= LOAD_W_BUF;
                        end else if (new_remaining < GEMV_TILE_SIZE) begin
                            if (w_buf_in_row < buf_tiles_per_row) begin
                                // Save carry, load next buffer tile for this row
                                for (int i = 0; i < GEMV_TILE_SIZE - 1; i++) begin
                                    if (i < TILE_ELEMS - int'(new_buf_idx))
                                        carry_buf[i] <= local_buf[int'(new_buf_idx) + i];
                                end
                                carry_count <= new_remaining[2:0];
                                mat_read_enable <= 1;
                                read_requested <= 1;
                                state <= LOAD_W_BUF;
                            end
                            // else: last buffer tile for row, next tile is last
                            // (padded with zeros from buffer), handled by w_can_send
                        end
                        // else: enough data, stay in FEED_W
                    end

                    // gemv_done now captured globally (above case statement)
                end

                // ============================================================
                // RECV_Y: Receive y tiles from GEMV, pack into result buffer
                // ============================================================
                RECV_Y: begin
                    if (y_tile_valid_from_gemv) begin
                        // Pack y tile elements into result_buffer
                        for (int i = 0; i < GEMV_TILE_SIZE; i++) begin
                            if (result_buf_idx + i < TILE_ELEMS)
                                result_buffer[result_buf_idx + i] <= y_tile_from_gemv[i];
                        end
                        // Store overflow elements for next buffer tile
                        for (int i = 0; i < GEMV_TILE_SIZE; i++) begin
                            if (result_buf_idx + i >= TILE_ELEMS)
                                overflow_buf[result_buf_idx + i - TILE_ELEMS] <= y_tile_from_gemv[i];
                        end

                        if (result_buf_idx + GEMV_TILE_SIZE >= TILE_ELEMS) begin
                            // Result buffer full or overflowed, write it out
                            overflow_count <= 6'(result_buf_idx + GEMV_TILE_SIZE - TILE_ELEMS);
                            state <= WRITE_RESULT;
                        end else begin
                            result_buf_idx <= result_buf_idx + GEMV_TILE_SIZE;
                        end
                    end

                    // gemv_done now captured globally (above case statement)

                    // Check completion: done and no pending y tile
                    if ((gemv_done || gemv_done_latched) && !y_tile_valid_from_gemv) begin
                        if (result_buf_idx > 0) begin
                            // Flush remaining results
                            overflow_count <= 0;
                            state <= WRITE_RESULT;
                        end else begin
                            state <= COMPLETE;
                        end
                    end
                end

                // ============================================================
                // WRITE_RESULT: Write result buffer tile to buffer controller
                // ============================================================
                WRITE_RESULT: begin
                    vec_write_enable <= 1;
                    result_tiles_written <= result_tiles_written + 1;
                    state <= SHIFT_OVERFLOW;
                end

                // ============================================================
                // SHIFT_OVERFLOW: Move overflow to buffer start, clear rest
                // ============================================================
                SHIFT_OVERFLOW: begin
                    // Move overflow to start of buffer, clear rest
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        if (i < GEMV_TILE_SIZE)
                            result_buffer[i] <= overflow_buf[i];
                        else
                            result_buffer[i] <= 0;
                    end
                    for (int i = 0; i < GEMV_TILE_SIZE; i++)
                        overflow_buf[i] <= 0;
                    result_buf_idx <= overflow_count;

                    if (result_tiles_written >= total_result_tiles) begin
                        state <= COMPLETE;
                    end else begin
                        state <= RECV_Y;
                    end
                end

                // ============================================================
                // COMPLETE: Signal done
                // ============================================================
                COMPLETE: begin
                    done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
