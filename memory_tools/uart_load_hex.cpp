/**
 * Load memory.hex to FPGA via UART and verify
 *
 * Parses memory.hex (one hex byte per line, 32768 lines = 32KB),
 * sends all bytes via UART in load mode, then verifies by reading
 * back sampled non-zero values.
 *
 * Build: g++ -std=c++17 -o uart_load_hex uart_load_hex.cpp
 * Usage: ./uart_load_hex <port> <hex_file> [--limit N]
 *   e.g. ./uart_load_hex /dev/cu.usbserial-1 memory.hex
 *         ./uart_load_hex /dev/cu.usbserial-1 memory.hex --limit 4096
 */

#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

class SerialPort {
public:
  int fd;

  SerialPort(const std::string &port) {
    fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
      perror("Unable to open serial port");
      exit(1);
    }
    fcntl(fd, F_SETFL, 0);

    struct termios options;
    tcgetattr(fd, &options);

    // cfmakeraw clears ALL input processing flags
    // (ISTRIP, ICRNL, INLCR, IGNCR, PARMRK, etc.)
    cfmakeraw(&options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    options.c_cflag |= (CLOCAL | CREAD);

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10;
    tcsetattr(fd, TCSANOW, &options);
  }

  ~SerialPort() { close(fd); }

  // Write a single byte with tcdrain (used for command bytes)
  void writeByte(uint8_t byte) {
    ssize_t n = write(fd, &byte, 1);
    if (n != 1) {
      perror("writeByte failed");
    }
    tcdrain(fd);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  // Write a chunk of bytes, return number written
  ssize_t writeChunk(const uint8_t *data, size_t len) {
    size_t total = 0;
    while (total < len) {
      ssize_t n = write(fd, data + total, len - total);
      if (n < 0) {
        perror("writeChunk failed");
        return -1;
      }
      total += n;
    }
    return total;
  }

  int readByte(uint8_t &byte, int timeout_ms = 2000) {
    fd_set readfds;
    struct timeval tv;
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int ret = select(fd + 1, &readfds, NULL, NULL, &tv);
    if (ret > 0)
      return read(fd, &byte, 1);
    return 0;
  }

  std::vector<uint8_t> readBytes(size_t count, int timeout_ms = 2000) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < count; i++) {
      uint8_t b;
      if (readByte(b, timeout_ms) > 0) {
        result.push_back(b);
      } else {
        break;
      }
    }
    return result;
  }

  void flush() { tcflush(fd, TCIOFLUSH); }
};

// Drain any pending bytes from RX buffer
void drainRx(SerialPort &serial) {
  uint8_t dummy;
  while (serial.readByte(dummy, 50) > 0) {
  }
}

// Echo test: send 0xBB, expect 0xCC back
bool echoTest(SerialPort &serial) {
  serial.flush();
  drainRx(serial);

  serial.writeByte(0xBB);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  uint8_t response;
  if (serial.readByte(response, 2000) > 0) {
    if (response == 0xCC) {
      return true;
    }
    std::cerr << "Echo: expected 0xCC, got 0x" << std::hex << std::setfill('0')
              << std::setw(2) << (int)response << std::dec << std::endl;
    return false;
  }
  std::cerr << "Echo: no response (timeout)" << std::endl;
  return false;
}

// Read a single byte from FPGA memory at given address
// Returns {true, value} on success, {false, 0} on timeout
std::pair<bool, uint8_t> readMemByte(SerialPort &serial, uint16_t addr) {
  // Drain any stale data (but DON'T flush output buffer)
  tcflush(serial.fd, TCIFLUSH); // Only flush INPUT, not output
  drainRx(serial);

  serial.writeByte(0xAA);
  serial.writeByte((addr >> 8) & 0xFF);
  serial.writeByte(addr & 0xFF);
  serial.writeByte(0x00);
  serial.writeByte(0x01);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto data = serial.readBytes(1, 2000);
  if (data.empty())
    return {false, 0};
  return {true, data[0]};
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <hex_file> [--limit N]"
              << std::endl;
    std::cerr << "  e.g. " << argv[0] << " /dev/cu.usbserial-1 memory.hex"
              << std::endl;
    std::cerr << "       " << argv[0]
              << " /dev/cu.usbserial-1 memory.hex --limit 4096" << std::endl;
    return 1;
  }

  std::string port = argv[1];
  std::string hex_file = argv[2];

  // Optional: limit number of bytes to load
  size_t limit = 0; // 0 = no limit
  for (int i = 3; i < argc - 1; i++) {
    if (std::string(argv[i]) == "--limit") {
      limit = std::stoul(argv[i + 1]);
    }
  }

  // Parse hex file: one hex byte per line
  std::ifstream fin(hex_file);
  if (!fin) {
    std::cerr << "Error: Cannot open " << hex_file << std::endl;
    return 1;
  }

  std::vector<uint8_t> mem_data;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty())
      continue;
    uint8_t val = static_cast<uint8_t>(std::stoul(line, nullptr, 16));
    mem_data.push_back(val);
  }
  fin.close();

  std::cout << "Parsed " << mem_data.size() << " bytes from " << hex_file
            << std::endl;

  if (limit > 0 && limit < mem_data.size()) {
    mem_data.resize(limit);
    std::cout << "Limited to " << limit << " bytes" << std::endl;
  }

  // Show first 16 bytes for sanity check
  std::cout << "First 16 bytes: ";
  for (size_t i = 0; i < 16 && i < mem_data.size(); i++) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << (int)mem_data[i] << " ";
  }
  std::cout << std::dec << std::endl;

  // Collect non-zero addresses for verification later
  std::vector<uint16_t> nonzero_addrs;
  for (size_t i = 0; i < mem_data.size(); i++) {
    if (mem_data[i] != 0x00) {
      nonzero_addrs.push_back(static_cast<uint16_t>(i));
    }
  }
  std::cout << "Non-zero bytes: " << nonzero_addrs.size() << std::endl;

  // Open serial
  SerialPort serial(port);
  serial.flush();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  drainRx(serial); // Drain hello byte (0x55) if present

  // --- LOAD PHASE ---
  std::cout << "\n=== LOAD PHASE ===" << std::endl;
  std::cout << "Make sure S2 is pressed (Load Mode active)!" << std::endl;
  std::cout << "Press Enter to start loading...";
  std::cin.get();

  auto t_start = std::chrono::steady_clock::now();

  // Write in chunks with tcdrain per chunk (not per byte!)
  // This avoids macOS tcdrain unreliability with per-byte calls
  const size_t CHUNK_SIZE = 128; // bytes per chunk
  size_t total_sent = 0;
  size_t total_bytes = mem_data.size();

  for (size_t offset = 0; offset < total_bytes; offset += CHUNK_SIZE) {
    size_t chunk_len = std::min(CHUNK_SIZE, total_bytes - offset);
    ssize_t n = serial.writeChunk(mem_data.data() + offset, chunk_len);
    if (n < 0) {
      std::cerr << "\nWrite error at offset " << offset << std::endl;
      return 1;
    }
    tcdrain(serial.fd);
    total_sent += chunk_len;

    if (total_sent % 1024 < CHUNK_SIZE || total_sent == total_bytes) {
      std::cout << "\rProgress: " << total_sent << "/" << total_bytes
                << " bytes (" << (total_sent * 100 / total_bytes) << "%)   "
                << std::flush;
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  std::cout << std::endl
            << "Write calls complete in " << std::fixed << std::setprecision(1)
            << elapsed << " seconds." << std::endl;

  // Calculate theoretical UART transmission time and wait for it
  // At 115200 baud, 10 bits/byte: total_bytes * 10 / 115200 seconds
  double tx_time_secs = (double)total_bytes * 10.0 / 115200.0;
  double wait_secs = tx_time_secs + 1.0; // Add 1 second margin

  // Only wait if the elapsed time is less than the tx time
  // (tcdrain might have already covered the time)
  if (elapsed < tx_time_secs) {
    double extra_wait = tx_time_secs - elapsed + 1.0;
    std::cout << "Waiting " << std::setprecision(1) << extra_wait
              << "s for UART transmission to complete..." << std::endl;
    std::this_thread::sleep_for(
        std::chrono::milliseconds((int)(extra_wait * 1000)));
  } else {
    // Still wait a small amount for safety
    std::cout << "Waiting 1s for USB buffer drain..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  std::cout << "All bytes should be physically transmitted now." << std::endl;

  // --- Release load mode before verification ---
  std::cout << "\nNow release S2 (exit Load Mode), then press Enter...";
  std::cin.get();
  serial.flush();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  drainRx(serial);

  // --- Echo test to verify FPGA is responsive ---
  std::cout << "\n=== CONNECTIVITY TEST ===" << std::endl;
  std::cout << "Testing FPGA echo (0xBB -> 0xCC)..." << std::endl;
  if (echoTest(serial)) {
    std::cout << "Echo OK - FPGA is responsive." << std::endl;
  } else {
    std::cerr << "Echo FAILED - FPGA is NOT responding!" << std::endl;
    std::cerr << "The FSM may be stuck. Try power cycling the FPGA."
              << std::endl;
    return 1;
  }

  // --- Quick sanity read at address 0 ---
  std::cout << "\nQuick read at address 0x0000..." << std::endl;
  auto [ok0, val0] = readMemByte(serial, 0x0000);
  if (ok0) {
    std::cout << "  [0x0000] = 0x" << std::hex << std::setfill('0')
              << std::setw(2) << (int)val0 << " (expected 0x" << std::setw(2)
              << (int)mem_data[0] << ")" << std::dec
              << (val0 == mem_data[0] ? " OK" : " MISMATCH") << std::endl;
  } else {
    std::cerr << "  [0x0000] TIMEOUT - no response from FPGA!" << std::endl;
    return 1;
  }

  // --- VERIFY PHASE ---
  std::cout << "\n=== VERIFY PHASE ===" << std::endl;
  std::cout << "Reading back sampled non-zero values to verify..." << std::endl;

  // Sample every Nth non-zero address to keep verification quick
  const int NUM_CHECKS = 20;
  int step =
      nonzero_addrs.size() > NUM_CHECKS ? nonzero_addrs.size() / NUM_CHECKS : 1;

  int pass = 0, fail = 0, timeout_count = 0;
  for (size_t idx = 0; idx < nonzero_addrs.size(); idx += step) {
    uint16_t addr = nonzero_addrs[idx];
    if (addr >= mem_data.size())
      break;
    uint8_t expected = mem_data[addr];
    auto [success, actual] = readMemByte(serial, addr);

    if (!success) {
      std::cout << "  [0x" << std::hex << std::setfill('0') << std::setw(4)
                << addr << "] expected=0x" << std::setw(2) << (int)expected
                << " TIMEOUT" << std::dec << std::endl;
      fail++;
      timeout_count++;
      if (timeout_count >= 3) {
        std::cerr << "Too many timeouts, aborting verification." << std::endl;
        break;
      }
      continue;
    }

    bool match = (actual == expected);
    std::cout << "  [0x" << std::hex << std::setfill('0') << std::setw(4)
              << addr << "] expected=0x" << std::setw(2) << (int)expected
              << " got=0x" << std::setw(2) << (int)actual
              << (match ? "  OK" : "  FAIL") << std::dec << std::endl;

    if (match)
      pass++;
    else
      fail++;
  }

  std::cout << "\nVerification: " << pass << " passed, " << fail
            << " failed out of " << (pass + fail) << " checks." << std::endl;

  if (fail == 0) {
    std::cout << "ALL CHECKS PASSED!" << std::endl;
    std::cout << "\nPress S2 then release to start the accelerator."
              << std::endl;
  } else {
    std::cout << "VERIFICATION FAILED!" << std::endl;
    return 1;
  }

  return 0;
}
