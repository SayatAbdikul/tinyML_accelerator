/**
 * Read 10 bytes from FPGA memory starting at 0x08C0 via UART,
 * print all values, and output the index of the maximum value.
 *
 * Build: g++ -std=c++17 -o uart_read_max uart_read_max.cpp
 * Usage: ./uart_read_max <port>
 *   e.g. ./uart_read_max /dev/cu.usbserial-1
 */

#include <iostream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <cstdint>
#include <iomanip>

class SerialPort {
public:
    int fd;

    SerialPort(const std::string& port) {
        fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd == -1) {
            perror("Unable to open serial port");
            exit(1);
        }
        fcntl(fd, F_SETFL, 0);

        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        options.c_iflag &= ~(IXON | IXOFF | IXANY);
        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = 10;
        tcsetattr(fd, TCSANOW, &options);
    }

    ~SerialPort() { close(fd); }

    void writeByte(uint8_t byte) {
        write(fd, &byte, 1);
        tcdrain(fd);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    int readByte(uint8_t& byte, int timeout_ms = 2000) {
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET(fd, &readfds);
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        int ret = select(fd + 1, &readfds, NULL, NULL, &tv);
        if (ret > 0) return read(fd, &byte, 1);
        return 0;
    }

    std::vector<uint8_t> readBytes(size_t count, int timeout_ms = 2000) {
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <port>" << std::endl;
        std::cerr << "  e.g. " << argv[0] << " /dev/cu.usbserial-1" << std::endl;
        return 1;
    }

    std::string port = argv[1];
    const uint16_t START_ADDR = 0x08C0;
    const uint16_t COUNT = 10;

    SerialPort serial(port);
    serial.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send read command: 0xAA + ADDR_H + ADDR_L + LEN_H + LEN_L
    std::cout << "Reading " << COUNT << " bytes from address 0x"
              << std::hex << std::setfill('0') << std::setw(4) << START_ADDR
              << "..." << std::endl;

    serial.writeByte(0xAA);
    serial.writeByte((START_ADDR >> 8) & 0xFF);
    serial.writeByte(START_ADDR & 0xFF);
    serial.writeByte((COUNT >> 8) & 0xFF);
    serial.writeByte(COUNT & 0xFF);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::vector<uint8_t> data = serial.readBytes(COUNT, 2000);

    if (data.size() != COUNT) {
        std::cerr << "Error: Expected " << std::dec << COUNT << " bytes, got "
                  << data.size() << std::endl;
        return 1;
    }

    // Print all values
    std::cout << "\nValues at 0x" << std::hex << std::setfill('0') << std::setw(4)
              << START_ADDR << ":" << std::endl;
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << "  [" << std::dec << i << "] 0x"
                  << std::hex << std::setfill('0') << std::setw(4)
                  << (START_ADDR + i) << " = 0x"
                  << std::setw(2) << (int)data[i]
                  << " (" << std::dec << (int)data[i] << ")" << std::endl;
    }

    // Find index of max value
    int max_idx = 0;
    uint8_t max_val = data[0];
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    std::cout << "\nMax value: 0x" << std::hex << std::setfill('0') << std::setw(2)
              << (int)max_val << " (" << std::dec << (int)max_val << ")"
              << " at index " << max_idx
              << " (address 0x" << std::hex << std::setw(4) << (START_ADDR + max_idx) << ")"
              << std::endl;

    return 0;
}
