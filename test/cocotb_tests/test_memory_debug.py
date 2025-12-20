"""
Debug test to verify memory writes work correctly in cocotb/Verilator
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import os
import sys

# Add compiler directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))

from dram import get_dram, save_dram_to_file


@cocotb.test()
async def test_memory_write_debug(dut):
    """
    Debug test to verify memory writes via cocotb backdoor access
    """
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Wait a few cycles
    for _ in range(5):
        await RisingEdge(dut.clk)
    
    # Test address - input area
    test_addr = 0x700
    test_value = 0xAB
    
    cocotb.log.info(f"\n{'='*70}")
    cocotb.log.info("MEMORY WRITE DEBUG TEST")
    cocotb.log.info(f"{'='*70}")
    
    # Read initial values from all memories
    cocotb.log.info(f"\nReading initial values at address 0x{test_addr:06X}:")
    
    try:
        fetch_val = dut.fetch_u.memory_inst.memory[test_addr].value.integer
        cocotb.log.info(f"  fetch_u.memory_inst.memory[{test_addr}] = 0x{fetch_val:02X}")
    except Exception as e:
        cocotb.log.error(f"  fetch_u.memory_inst.memory: {e}")
        fetch_val = None
    
    try:
        load_v_val = dut.execution_u.load_exec.load_v_inst.memory_inst.memory[test_addr].value.integer
        cocotb.log.info(f"  load_v.memory_inst.memory[{test_addr}] = 0x{load_v_val:02X}")
    except Exception as e:
        cocotb.log.error(f"  load_v.memory_inst.memory: {e}")
        load_v_val = None
    
    try:
        load_m_val = dut.execution_u.load_exec.load_m_inst.memory_inst.memory[test_addr].value.integer
        cocotb.log.info(f"  load_m.memory_inst.memory[{test_addr}] = 0x{load_m_val:02X}")
    except Exception as e:
        cocotb.log.error(f"  load_m.memory_inst.memory: {e}")
        load_m_val = None
    
    try:
        store_val = dut.execution_u.store_exec.store_inst.dram.memory[test_addr].value.integer
        cocotb.log.info(f"  store.dram.memory[{test_addr}] = 0x{store_val:02X}")
    except Exception as e:
        cocotb.log.error(f"  store.dram.memory: {e}")
        store_val = None
    
    # Now try writing to all memories
    cocotb.log.info(f"\nWriting 0x{test_value:02X} to all memories at address 0x{test_addr:06X}:")
    
    try:
        dut.fetch_u.memory_inst.memory[test_addr].value = test_value
        cocotb.log.info("  fetch_u.memory_inst.memory: write successful")
    except Exception as e:
        cocotb.log.error(f"  fetch_u.memory_inst.memory: write failed - {e}")
    
    try:
        dut.execution_u.load_exec.load_v_inst.memory_inst.memory[test_addr].value = test_value
        cocotb.log.info("  load_v.memory_inst.memory: write successful")
    except Exception as e:
        cocotb.log.error(f"  load_v.memory_inst.memory: write failed - {e}")
    
    try:
        dut.execution_u.load_exec.load_m_inst.memory_inst.memory[test_addr].value = test_value
        cocotb.log.info("  load_m.memory_inst.memory: write successful")
    except Exception as e:
        cocotb.log.error(f"  load_m.memory_inst.memory: write failed - {e}")
    
    try:
        dut.execution_u.store_exec.store_inst.dram.memory[test_addr].value = test_value
        cocotb.log.info("  store.dram.memory: write successful")
    except Exception as e:
        cocotb.log.error(f"  store.dram.memory: write failed - {e}")
    
    # Wait a clock cycle for writes to settle
    await RisingEdge(dut.clk)
    
    # Read back values to verify writes
    cocotb.log.info(f"\nReading back values after write:")
    
    try:
        fetch_val_after = dut.fetch_u.memory_inst.memory[test_addr].value.integer
        match = "✓" if fetch_val_after == test_value else "✗"
        cocotb.log.info(f"  fetch_u.memory_inst.memory[{test_addr}] = 0x{fetch_val_after:02X} {match}")
    except Exception as e:
        cocotb.log.error(f"  fetch_u.memory_inst.memory: {e}")
    
    try:
        load_v_val_after = dut.execution_u.load_exec.load_v_inst.memory_inst.memory[test_addr].value.integer
        match = "✓" if load_v_val_after == test_value else "✗"
        cocotb.log.info(f"  load_v.memory_inst.memory[{test_addr}] = 0x{load_v_val_after:02X} {match}")
    except Exception as e:
        cocotb.log.error(f"  load_v.memory_inst.memory: {e}")
    
    try:
        load_m_val_after = dut.execution_u.load_exec.load_m_inst.memory_inst.memory[test_addr].value.integer
        match = "✓" if load_m_val_after == test_value else "✗"
        cocotb.log.info(f"  load_m.memory_inst.memory[{test_addr}] = 0x{load_m_val_after:02X} {match}")
    except Exception as e:
        cocotb.log.error(f"  load_m.memory_inst.memory: {e}")
    
    try:
        store_val_after = dut.execution_u.store_exec.store_inst.dram.memory[test_addr].value.integer
        match = "✓" if store_val_after == test_value else "✗"
        cocotb.log.info(f"  store.dram.memory[{test_addr}] = 0x{store_val_after:02X} {match}")
    except Exception as e:
        cocotb.log.error(f"  store.dram.memory: {e}")
    
    cocotb.log.info(f"\n{'='*70}")
    
    # Final assertion
    all_match = (
        load_v_val_after == test_value and 
        load_m_val_after == test_value and 
        store_val_after == test_value
    )
    
    if all_match:
        cocotb.log.info("✅ Memory write test PASSED - all writes verified")
    else:
        cocotb.log.error("❌ Memory write test FAILED - some writes did not persist")
    
    assert all_match, "Memory writes did not persist correctly"


@cocotb.test()
async def test_check_dram_hierarchy(dut):
    """
    Print the DUT hierarchy to help debug memory paths
    """
    cocotb.log.info(f"\n{'='*70}")
    cocotb.log.info("DUT HIERARCHY DEBUG")
    cocotb.log.info(f"{'='*70}")
    
    # Start clock (required for simulation)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await RisingEdge(dut.clk)
    
    # Explore hierarchy
    cocotb.log.info("\nTop-level signals and modules:")
    for name in dir(dut):
        if not name.startswith('_'):
            try:
                obj = getattr(dut, name)
                cocotb.log.info(f"  {name}: {type(obj).__name__}")
            except:
                pass
    
    cocotb.log.info("\nexecution_u children:")
    try:
        for name in dir(dut.execution_u):
            if not name.startswith('_'):
                try:
                    obj = getattr(dut.execution_u, name)
                    cocotb.log.info(f"  execution_u.{name}: {type(obj).__name__}")
                except:
                    pass
    except Exception as e:
        cocotb.log.error(f"Cannot access execution_u: {e}")
    
    cocotb.log.info("\nload_exec children:")
    try:
        for name in dir(dut.execution_u.load_exec):
            if not name.startswith('_'):
                try:
                    obj = getattr(dut.execution_u.load_exec, name)
                    cocotb.log.info(f"  load_exec.{name}: {type(obj).__name__}")
                except:
                    pass
    except Exception as e:
        cocotb.log.error(f"Cannot access load_exec: {e}")
    
    cocotb.log.info(f"\n{'='*70}")
    cocotb.log.info("✅ Hierarchy exploration complete")
