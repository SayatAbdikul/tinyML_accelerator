/**
 * UART Memory Interface for TinyML Accelerator
 *
 * Protocol:
 *   LOAD MODE (S2 active):
 *     - ALL bytes (including 0xAA) are written as data sequentially from 0x0000
 *     - No read commands possible in this mode
 *
 *   NORMAL MODE (S2 released):
 *     - Send: 0xAA + ADDR_H + ADDR_L + LEN_H + LEN_L
 *     - Receive: LEN bytes from memory starting at ADDR
 *
 * Build: g++ -std=c++17 -o uart_memory uart_memory.cpp
 * Usage: ./uart_memory <port> <command> [args...]
 */

#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

class SerialPort {
public:
  int fd;
  std::string port_name;

  SerialPort(const std::string &port) : port_name(port) {
    fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
      perror("open_port: Unable to open serial port");
      exit(1);
    }
    fcntl(fd, F_SETFL, 0); // Blocking read

    struct termios options;
    tcgetattr(fd, &options);

    // cfmakeraw clears ALL input processing flags
    // (ISTRIP, ICRNL, INLCR, IGNCR, PARMRK, etc.)
    cfmakeraw(&options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    options.c_cflag |= (CLOCAL | CREAD);

    // Read timeout: 1 second
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10; // 1 second timeout

    tcsetattr(fd, TCSANOW, &options);
  }

  ~SerialPort() { close(fd); }

  void writeByte(uint8_t byte) {
    write(fd, &byte, 1);
    tcdrain(fd);
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  void writeBytes(const std::vector<uint8_t> &data) {
    for (uint8_t b : data) {
      writeByte(b);
    }
  }

  int readByte(uint8_t &byte, int timeout_ms = 1000) {
    fd_set readfds;
    struct timeval tv;
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(fd + 1, &readfds, NULL, NULL, &tv);
    if (ret > 0) {
      return read(fd, &byte, 1);
    }
    return 0; // Timeout
  }

  std::vector<uint8_t> readBytes(size_t count, int timeout_ms = 1000) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < count; i++) {
      uint8_t b;
      if (readByte(b, timeout_ms) > 0) {
        result.push_back(b);
      } else {
        std::cerr << "Timeout after " << result.size() << " bytes" << std::endl;
        break;
      }
    }
    return result;
  }

  void flush() { tcflush(fd, TCIOFLUSH); }
};

void printUsage(const char *prog) {
  std::cout
      << "Usage: " << prog << " <port> <command> [args...]\n"
      << "\nCommands:\n"
      << "  load <file>           Load binary file to memory (S2 must be "
         "active)\n"
      << "  read <addr> <len>     Read <len> bytes from <addr> (hex)\n"
      << "  dump <addr> <len>     Read and hexdump <len> bytes from <addr>\n"
      << "  probe                 Read 1 byte from addr 0 (tests UART path)\n"
      << "  listen [secs]         Listen for any incoming bytes (default 5s)\n"
      << "  test                  Self-test: write pattern and read back\n"
      << "\nExamples:\n"
      << "  " << prog << " /dev/tty.usbmodem12345 load program.bin\n"
      << "  " << prog << " /dev/tty.usbmodem12345 read 0x0000 256\n"
      << "  " << prog << " /dev/tty.usbmodem12345 dump 0 64\n";
}

void hexDump(const std::vector<uint8_t> &data, uint16_t start_addr) {
  for (size_t i = 0; i < data.size(); i += 16) {
    std::cout << std::hex << std::setfill('0') << std::setw(4)
              << (start_addr + i) << ": ";

    // Hex bytes
    for (size_t j = 0; j < 16 && (i + j) < data.size(); j++) {
      std::cout << std::setw(2) << (int)data[i + j] << " ";
    }

    // Padding if less than 16 bytes
    for (size_t j = data.size() - i; j < 16; j++) {
      std::cout << "   ";
    }

    // ASCII
    std::cout << " |";
    for (size_t j = 0; j < 16 && (i + j) < data.size(); j++) {
      char c = data[i + j];
      std::cout << (c >= 32 && c < 127 ? c : '.');
    }
    std::cout << "|" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  std::string port = argv[1];
  std::string command = argv[2];

  SerialPort serial(port);
  serial.flush();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  if (command == "load") {
    if (argc < 4) {
      std::cerr << "Error: load requires a filename" << std::endl;
      return 1;
    }

    std::string filename = argv[3];
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      std::cerr << "Error: Cannot open file " << filename << std::endl;
      return 1;
    }

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());

    std::cout << "Loading " << data.size() << " bytes from " << filename
              << std::endl;
    std::cout << "Make sure S2 is pressed (Load Mode active)!" << std::endl;
    std::cout << "Press Enter to continue...";
    std::cin.get();

    for (size_t i = 0; i < data.size(); i++) {
      serial.writeByte(data[i]);
      if ((i + 1) % 1024 == 0 || i == data.size() - 1) {
        std::cout << "\rProgress: " << (i + 1) << "/" << data.size()
                  << " bytes (" << ((i + 1) * 100 / data.size()) << "%)   "
                  << std::flush;
      }
    }
    std::cout << std::endl << "Load complete!" << std::endl;

  } else if (command == "read" || command == "dump") {
    if (argc < 5) {
      std::cerr << "Error: " << command << " requires address and length"
                << std::endl;
      return 1;
    }

    uint16_t addr = std::stoul(argv[3], nullptr, 0);
    uint16_t len = std::stoul(argv[4], nullptr, 0);

    std::cout << "Reading " << len << " bytes from address 0x" << std::hex
              << addr << std::endl;

    // Send read command: 0xAA + ADDR_H + ADDR_L + LEN_H + LEN_L
    serial.writeByte(0xAA);
    serial.writeByte((addr >> 8) & 0xFF);
    serial.writeByte(addr & 0xFF);
    serial.writeByte((len >> 8) & 0xFF);
    serial.writeByte(len & 0xFF);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::vector<uint8_t> data = serial.readBytes(len, 2000);

    if (data.empty()) {
      std::cerr << "Error: No data received" << std::endl;
      return 1;
    }

    std::cout << "Received " << std::dec << data.size() << " bytes"
              << std::endl;

    if (command == "dump") {
      hexDump(data, addr);
    } else {
      // Raw output for piping
      for (uint8_t b : data) {
        std::cout << b;
      }
    }

  } else if (command == "echo") {
    // Debug: Send 0xBB, expect 0xCC back (tests RX->TX path)
    std::cout << "Echo test: Sending 0xBB, expecting 0xCC back..." << std::endl;
    serial.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    serial.writeByte(0xBB);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    uint8_t response;
    if (serial.readByte(response, 2000) > 0) {
      std::cout << "Received: 0x" << std::hex << std::setfill('0')
                << std::setw(2) << (int)response << std::endl;
      if (response == 0xCC) {
        std::cout << "ECHO TEST PASSED! RX->TX path works." << std::endl;
      } else {
        std::cout << "ECHO TEST FAILED! Expected 0xCC, got 0x" << std::setw(2)
                  << (int)response << std::endl;
        return 1;
      }
    } else {
      std::cout << "ECHO TEST FAILED! No response (timeout)." << std::endl;
      std::cout << "FPGA did not receive or process the 0xBB byte."
                << std::endl;
      return 1;
    }

  } else if (command == "listen") {
    // Passive listen: print any bytes received for N seconds
    int listen_secs = 5;
    if (argc >= 4)
      listen_secs = std::stoi(argv[3]);

    std::cout << "Listening for " << listen_secs << " seconds on " << port
              << "..." << std::endl;
    std::cout << "Power cycle the FPGA or press reset to trigger hello byte "
                 "(0x55 = 'U')."
              << std::endl;

    serial.flush();
    auto start = std::chrono::steady_clock::now();
    int count = 0;

    while (true) {
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >=
          listen_secs)
        break;

      uint8_t b;
      if (serial.readByte(b, 500) > 0) {
        std::cout << "  [" << count << "] = 0x" << std::hex << std::setfill('0')
                  << std::setw(2) << (int)b << " ('"
                  << (char)(b >= 32 && b < 127 ? b : '.') << "')"
                  << " at " << std::dec
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << "ms" << std::endl;
        count++;
      }
    }

    if (count == 0) {
      std::cout << "No bytes received. UART TX path may not be working."
                << std::endl;
    } else {
      std::cout << "Received " << count << " byte(s) total." << std::endl;
    }

  } else if (command == "probe") {
    // Diagnostic: test basic UART TX/RX path
    // S2 must be released (normal mode) for read commands to work
    std::cout << "Probe: reading 1 byte from address 0x0000..." << std::endl;
    std::cout << "Make sure S2 is RELEASED (Normal Mode)!" << std::endl;

    serial.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Drain any stale RX data
    uint8_t dummy;
    int drained = 0;
    while (serial.readByte(dummy, 50) > 0)
      drained++;
    if (drained > 0) {
      std::cout << "Drained " << drained << " stale bytes from RX buffer"
                << std::endl;
    }

    // Send read command for addr=0x0000, len=1
    std::cout << "Sending: AA 00 00 00 01" << std::endl;
    serial.writeByte(0xAA);
    serial.writeByte(0x00);
    serial.writeByte(0x00);
    serial.writeByte(0x00);
    serial.writeByte(0x01);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Read with generous timeout, print ALL bytes received
    std::cout << "Waiting for response (3s timeout)..." << std::endl;
    std::vector<uint8_t> response;
    uint8_t b;
    while (serial.readByte(b, 3000) > 0) {
      response.push_back(b);
      if (response.size() >= 16)
        break; // Safety limit
    }

    if (response.empty()) {
      std::cout << "No response received (timeout). UART RX path may be broken."
                << std::endl;
      return 1;
    }

    std::cout << "Received " << response.size() << " byte(s):" << std::endl;
    for (size_t i = 0; i < response.size(); i++) {
      std::cout << "  [" << i << "] = 0x" << std::hex << std::setfill('0')
                << std::setw(2) << (int)response[i] << " (dec " << std::dec
                << (int)response[i] << ")" << std::endl;
    }

    if (response.size() == 1) {
      std::cout << "Probe OK. UART path is working." << std::endl;
    } else {
      std::cout << "Warning: Expected 1 byte, got " << response.size()
                << std::endl;
    }

  } else if (command == "test") {
    std::cout << "Self-test: Writing pattern and reading back..." << std::endl;
    std::cout << "Make sure S2 is pressed (Load Mode active)!" << std::endl;
    std::cout << "Press Enter to start write phase...";
    std::cin.get();

    // Write 16 bytes of test pattern
    // In load mode, ALL bytes (including 0xAA) are written as data
    std::vector<uint8_t> pattern;
    uint8_t pat[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                     0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    pattern.assign(pat, pat + sizeof(pat));

    std::cout << "Writing test pattern..." << std::endl;
    serial.writeBytes(pattern);

    // Small delay to ensure all writes are processed
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Read commands only work outside load mode
    std::cout
        << "Release S2 (exit Load Mode), then press Enter to read back...";
    if (std::cin.get() == EOF) {
      std::cerr << "\nError: EOF on stdin. Test requires 2 interactive Enter "
                   "presses.\n"
                << "Run interactively (not piped) or use: printf '\\n\\n' | "
                   "./uart_memory ..."
                << std::endl;
      return 1;
    }
    serial.flush();
    std::cout << "Reading back..." << std::endl;

    // Read back
    serial.writeByte(0xAA);
    serial.writeByte(0x00); // Addr high
    serial.writeByte(0x00); // Addr low
    serial.writeByte(0x00); // Len high
    serial.writeByte(0x10); // Len low (16)

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::vector<uint8_t> result = serial.readBytes(16, 2000);

    std::cout << "Written:  ";
    for (uint8_t b : pattern)
      std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b
                << " ";
    std::cout << std::endl;

    std::cout << "Received: ";
    for (uint8_t b : result)
      std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b
                << " ";
    std::cout << std::endl;

    if (result == pattern) {
      std::cout << "TEST PASSED!" << std::endl;
    } else {
      std::cout << "TEST FAILED!" << std::endl;
      return 1;
    }

  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  return 0;
}
