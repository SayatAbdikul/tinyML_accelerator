/**
 * UART Verification Tool
 *
 * 1. Loads data from a hex file to FPGA memory via UART.
 * 2. Reads back the data via UART (if supported by FPGA).
 * 3. Compares loaded vs read data to verify integrity.
 *
 * Usage: ./uart_verify <serial_port> <hex_file> [start_addr]
 */

#include "uart_device.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Helper to load hex file
std::vector<uint8_t> load_hex_file(const std::string &filename) {
  std::vector<uint8_t> data;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return data;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    try {
      uint8_t byte = static_cast<uint8_t>(std::stoul(line, nullptr, 16));
      data.push_back(byte);
    } catch (const std::exception &e) {
      std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
    }
  }
  return data;
}

// Helper to send read request (Protocol: 0x01 [ADDR_H] [ADDR_L] [LEN])
bool send_read_request(UARTDevice &uart, uint16_t address, uint8_t length) {
  uint8_t packet[4];
  packet[0] = 0x01;
  packet[1] = (address >> 8) & 0xFF;
  packet[2] = address & 0xFF;
  packet[3] = length;

  return uart.write_bytes(packet, 4);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <serial_port> <hex_file> [start_addr]" << std::endl;
    return 1;
  }

  std::string port = argv[1];
  std::string hex_file = argv[2];
  uint16_t start_addr = (argc > 3) ? std::stoul(argv[3], nullptr, 0) : 0;

  // 1. Load Data
  std::cout << "[1/3] Loading hex file: " << hex_file << std::endl;
  std::vector<uint8_t> expected_data = load_hex_file(hex_file);

  if (expected_data.empty()) {
    std::cerr << "Error: No data loaded." << std::endl;
    return 1;
  }
  std::cout << "Loaded " << expected_data.size() << " bytes." << std::endl;

  std::cout << "Opening UART port: " << port << std::endl;
  UARTDevice uart(port);
  if (!uart.is_open())
    return 1;

  std::cout << "Writing to FPGA..." << std::endl;
  if (!uart.write_bytes(expected_data)) {
    std::cerr << "Write failed." << std::endl;
    return 1;
  }
  std::cout << "Write complete." << std::endl;

  // 2. Read Back Verification
  std::cout << "[2/3] Verifying data..." << std::endl;

  // We can only read in chunks of 255 bytes max due to protocol
  size_t total_bytes = expected_data.size();
  size_t verified_bytes = 0;
  size_t errors = 0;

  uint16_t current_addr = start_addr;
  size_t remaining = total_bytes;

  while (remaining > 0) {
    uint8_t chunk_size =
        (remaining > 64) ? 64 : remaining; // Use 64 byte chunks for safety

    // Send Read Request
    // std::cout << "Reading " << (int)chunk_size << " bytes from 0x" <<
    // std::hex << current_addr << std::dec << "..." << std::flush;
    if (!send_read_request(uart, current_addr, chunk_size)) {
      std::cerr << "Failed to send read request." << std::endl;
      return 1;
    }

    // Read Response
    std::vector<uint8_t> read_data =
        uart.read_bytes(chunk_size, 2000); // 2s timeout

    if (read_data.size() != chunk_size) {
      std::cerr << "\nError: Timeout or incomplete read. Expected "
                << (int)chunk_size << ", got " << read_data.size() << std::endl;
      errors++;
      break;
    }

    // Compare
    for (size_t i = 0; i < chunk_size; i++) {
      if (read_data[i] != expected_data[verified_bytes + i]) {
        std::cerr << "\nMismatch at addr 0x" << std::hex << (current_addr + i)
                  << ": Expected 0x" << (int)expected_data[verified_bytes + i]
                  << ", Got 0x" << (int)read_data[i] << std::dec << std::endl;
        errors++;
      }
    }

    current_addr += chunk_size;
    verified_bytes += chunk_size;
    remaining -= chunk_size;

    std::cout << "\rProgress: " << (verified_bytes * 100 / total_bytes) << "%"
              << std::flush;
  }

  std::cout << std::endl;

  // 3. Result
  std::cout << "[3/3] Result: ";
  if (errors == 0) {
    std::cout << "PASS ✅ (Verified " << verified_bytes << " bytes)"
              << std::endl;
    return 0;
  } else {
    std::cout << "FAIL ❌ (" << errors << " mismatches)" << std::endl;
    return 1;
  }
}
