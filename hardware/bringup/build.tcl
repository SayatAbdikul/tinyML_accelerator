# Run in a disposable output directory: gw_sh /absolute/repo/hardware/bringup/build.tcl
set root [file normalize [file join [file dirname [info script]] ../..]]
create_project -name uart_loopback -dir [pwd] -pn GW2AR-LV18QN88C8/I7 -device_version C
set_device -device_version C GW2AR-LV18QN88C8/I7
add_file [file join $root hardware/bringup/uart_loopback.sv]
add_file [file join $root src/uart_rx.sv]
add_file [file join $root src/uart_tx.sv]
add_file [file join $root hardware/bringup/uart_loopback.cst]
add_file [file join $root src/fpga_project_1.sdc]
set_option -top_module uart_loopback
set_option -verilog_std sysv2017
set_option -output_base_name uart_loopback
run all
exit
