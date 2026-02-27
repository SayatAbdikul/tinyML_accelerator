/**
 * UART Memory Reader for TinyML Accelerator
 *
 * Reads data from FPGA memory via UART at a specified address.
 *
 * NOTE: This requires FPGA RTL support for read requests.
 *       The current RTL simple_memory.sv only supports writes via UART.
 *       This file provides a template for when read support is added.
 *
 * Usage: ./uart_reader <serial_port> <start_address> <length>
 * Example: ./uart_reader /dev/ttyUSB0 0x8C0 10
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "uart_device.h"

// Protocol: Send a read request packet to FPGA
// Format: [CMD_READ(0x01)] [ADDR_HIGH] [ADDR_LOW] [LENGTH]
// FPGA responds with [LENGTH] bytes of data
bool send_read_request(UARTDevice &uart, uint16_t address, uint8_t length) {
  uint8_t packet[4];
  packet[0] = 0x01;                  // Read command
  packet[1] = (address >> 8) & 0xFF; // Address high byte
  packet[2] = address & 0xFF;        // Address low byte
  packet[3] = length;                // Number of bytes to read

  return uart.write_bytes(packet, 4);
}

void print_hex_dump(const std::vector<uint8_t> &data, uint16_t start_addr) {
  const int bytes_per_line = 16;

  for (size_t i = 0; i < data.size(); i += bytes_per_line) {
    // Print address
    std::cout << std::hex << std::setfill('0') << std::setw(4)
              << (start_addr + i) << ":  ";

    // Print hex bytes
    for (size_t j = 0; j < bytes_per_line && (i + j) < data.size(); j++) {
      std::cout << std::setw(2) << static_cast<int>(data[i + j]) << " ";
    }

    // Pad if needed
    for (size_t j = data.size() - i;
         j < bytes_per_line && data.size() - i < bytes_per_line; j++) {
      std::cout << "   ";
    }

    std::cout << " |";

    // Print ASCII
    for (size_t j = 0; j < bytes_per_line && (i + j) < data.size(); j++) {
      char c = data[i + j];
      std::cout << (isprint(c) ? c : '.');
    }

    std::cout << "|" << std::endl;
  }
  std::cout << std::dec;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <serial_port> <start_address> <length>" << std::endl;
    std::cerr << "Example: " << argv[0] << " /dev/ttyUSB0 0x8C0 10"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "NOTE: This requires FPGA RTL support for UART read requests."
              << std::endl;
    std::cerr << "      The current simple_memory.sv only supports writes."
              << std::endl;
    return 1;
  }

  std::string port = argv[1];
  uint16_t start_addr = static_cast<uint16_t>(std::stoul(argv[2], nullptr, 0));
  size_t length = std::stoul(argv[3], nullptr, 0);

  if (length > 255) {
    std::cerr << "Error: Maximum read length is 255 bytes per request"
              << std::endl;
    return 1;
  }

  std::cout << "Opening UART port: " << port << std::endl;
  UARTDevice uart(port);

  if (!uart.is_open()) {
    std::cerr << "Error: Failed to open UART port" << std::endl;
    return 1;
  }

  std::cout << "Sending read request: addr=0x" << std::hex << start_addr
            << ", len=" << std::dec << length << std::endl;

  if (!send_read_request(uart, start_addr, static_cast<uint8_t>(length))) {
    std::cerr << "Error: Failed to send read request" << std::endl;
    return 1;
  }

  std::cout << "Waiting for response..." << std::endl;
  std::vector<uint8_t> data = uart.read_bytes(length, 2000);

  if (data.empty()) {
    std::cerr << "Error: No response from FPGA (timeout)" << std::endl;
    std::cerr << "Note: FPGA RTL may not support read requests yet."
              << std::endl;
    return 1;
  }

  std::cout << "Received " << data.size() << " bytes:" << std::endl;
  std::cout << std::endl;
  print_hex_dump(data, start_addr);

  // Print as signed int8 values (useful for inference results)
  std::cout << std::endl << "As signed int8 values:" << std::endl;
  for (size_t i = 0; i < data.size(); i++) {
    int8_t signed_val = static_cast<int8_t>(data[i]);
    std::cout << "[" << i << "]: " << static_cast<int>(signed_val) << std::endl;
  }

  return 0;
}
