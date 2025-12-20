module i_decoder (
    input  logic [63:0] instr,
    output logic [4:0]  opcode,
    output logic [4:0]  dest,
    output logic [9:0]  length_or_cols,
    output logic [9:0]  rows,
    output logic [23:0] addr,
    output logic [4:0]  b,
    output logic [4:0]  x,
    output logic [4:0]  w
);

    always_comb begin
        opcode = instr[4:0];

        // Defaults
        dest   = 0;
        length_or_cols = 0;
        rows   = 0;
        addr   = 0;
        b = 0;
        x = 0;
        w = 0;

        case (opcode)
            5'h00: begin // NOP
            end

            5'h01, 5'h03: begin // LOAD_V or STORE
                dest           = instr[9:5];
                length_or_cols = instr[19:10];
                addr           = instr[63:40];
            end

            5'h02: begin // LOAD_M
                dest           = instr[9:5];
                length_or_cols = instr[19:10];  // cols
                rows           = instr[29:20];
                addr           = instr[63:40];
            end

            5'h04: begin // GEMV
                dest = instr[9:5];
                length_or_cols = instr[19:10]; // cols
                rows = instr[29:20];
                b    = instr[34:30];
                x    = instr[39:35];
                w    = instr[44:40];
            end

            5'h05: begin // RELU
                dest           = instr[9:5];
                x              = instr[14:10];
                length_or_cols = instr[29:20];  // RELU length field
            end

            default: begin
                
            end
            
        endcase
    end
endmodule
