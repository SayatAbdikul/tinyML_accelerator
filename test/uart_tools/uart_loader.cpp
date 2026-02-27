/**
 * UART Memory Loader for TinyML Accelerator
 *
 * Loads a hex file into the FPGA memory via UART.
 *
 * Usage: ./uart_loader <serial_port> <hex_file>
 * Example: ./uart_loader /dev/ttyUSB0 ../../compiler/dram.hex
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "uart_device.h"

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
    // Remove whitespace
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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <serial_port> <hex_file>"
              << std::endl;
    std::cerr << "Example: " << argv[0]
              << " /dev/ttyUSB0 ../../compiler/dram.hex" << std::endl;
    return 1;
  }

  std::string port = argv[1];
  std::string hex_file = argv[2];

  std::cout << "Loading hex file: " << hex_file << std::endl;
  std::vector<uint8_t> data = load_hex_file(hex_file);

  if (data.empty()) {
    std::cerr << "Error: No data loaded from hex file" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << data.size() << " bytes" << std::endl;

  std::cout << "Opening UART port: " << port << std::endl;
  UARTDevice uart(port);

  if (!uart.is_open()) {
    std::cerr << "Error: Failed to open UART port" << std::endl;
    return 1;
  }

  std::cout << "Sending data to FPGA..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  size_t sent = 0;
  for (uint8_t byte : data) {
    if (!uart.write_byte(byte)) {
      std::cerr << "Error: Failed to write byte at offset " << sent
                << std::endl;
      return 1;
    }
    sent++;

    // Progress indicator
    if (sent % 1000 == 0) {
      std::cout << "\rProgress: " << sent << " / " << data.size() << " ("
                << (100 * sent / data.size()) << "%)" << std::flush;
    }

    // Delay to match UART receiver (115200 baud ~ 11520 bytes/sec)
    std::this_thread::sleep_for(std::chrono::microseconds(90));
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\rProgress: " << sent << " / " << data.size() << " (100%)"
            << std::endl;
  std::cout << "Transfer complete in " << duration.count() << " ms"
            << std::endl;
  std::cout << "Effective rate: " << (data.size() * 1000.0 / duration.count())
            << " bytes/sec" << std::endl;

  return 0;
}
