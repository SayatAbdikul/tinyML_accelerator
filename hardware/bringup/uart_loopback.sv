// Minimal phase-0 lab image. Stop-and-wait host protocol: send byte, read echo.
module uart_loopback (
    input logic sys_clk, sys_rst_n, uart_rx,
    output logic uart_tx,
    output logic [5:0] led
);
    logic [7:0] rx_data;
    logic rx_valid, tx_ready;
    uart_rx receiver(.clk(sys_clk), .rst_n(sys_rst_n), .rx_i(uart_rx),
                     .rx_data(rx_data), .rx_valid(rx_valid));
    uart_tx transmitter(.clk(sys_clk), .rst_n(sys_rst_n), .tx_data(rx_data),
                        .tx_valid(rx_valid && tx_ready), .tx_ready(tx_ready), .tx_o(uart_tx));
    assign led = ~rx_data[5:0];
endmodule
