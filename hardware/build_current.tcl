# Reproduce the checked-out src/ board hierarchy, without changing its profile.
# Run from a fresh writable build directory.
set root [file normalize [file join [file dirname [info script]] ..]]
create_project -name accelerator_current -dir [pwd] -pn GW2AR-LV18QN88C8/I7 -device_version C
add_file [file join $root rtl/accelerator_config_pkg.sv]
foreach source [lsort [glob [file join $root src/*.sv]]] { add_file $source }
foreach ip {gowin_ram16sdp gowin_sdpb gowin_sp} {
    add_file [file join $root src $ip ${ip}.v]
}
add_file [file join $root src/fpga_project_1.cst]
add_file [file join $root src/fpga_project_1.sdc]
set_option -top_module fpga_top
set_option -verilog_std sysv2017
set_option -output_base_name accelerator_current
run all
exit
