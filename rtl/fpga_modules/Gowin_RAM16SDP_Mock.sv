module Gowin_RAM16SDP_Mock (
    output logic [31:0] dout,
    input clka,
    input cea,
    input reseta,
    input clkb,
    input ceb,
    input resetb,
    input oce,
    input [11:0] ada,
    input [31:0] din,
    input [11:0] adb
);

    // 1024x32 RAM Array inference
    logic [31:0] mem [0:1023];

    // Port B (Read Port - Standard synchronous read)
    // Most FPGAs have 1-cycle read latency where `dout` is registered
    // on the clock edge where the address is sampled. 
    always_ff @(posedge clkb) begin
        if (ceb) begin
            dout <= mem[adb[9:0]];
        end
    end

    // Port A (Write Port)
    always_ff @(posedge clka) begin
        if (cea) begin
            mem[ada[9:0]] <= din;
        end
    end

endmodule
