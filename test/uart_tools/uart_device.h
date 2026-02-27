#ifndef UART_DEVICE_H
#define UART_DEVICE_H

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

class UARTDevice {
public:
  UARTDevice(const std::string &port, int baud_rate = B115200)
      : fd_(-1), port_(port) {
    open_port(baud_rate);
  }

  ~UARTDevice() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  bool is_open() const { return fd_ >= 0; }

  bool write_byte(uint8_t byte) {
    if (fd_ < 0)
      return false;
    ssize_t written = write(fd_, &byte, 1);
    return written == 1;
  }

  bool write_bytes(const std::vector<uint8_t> &data) {
    for (uint8_t byte : data) {
      if (!write_byte(byte))
        return false;
      // Small delay between bytes to prevent buffer overflow
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
  }

  bool write_bytes(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
      if (!write_byte(data[i]))
        return false;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
  }

  int read_byte(int timeout_ms = 1000) {
    if (fd_ < 0)
      return -1;

    fd_set read_fds;
    struct timeval timeout;

    FD_ZERO(&read_fds);
    FD_SET(fd_, &read_fds);

    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ret > 0) {
      uint8_t byte;
      if (read(fd_, &byte, 1) == 1) {
        return byte;
      }
    }
    return -1;
  }

  std::vector<uint8_t> read_bytes(size_t count, int timeout_ms = 1000) {
    std::vector<uint8_t> data;
    for (size_t i = 0; i < count; i++) {
      int byte = read_byte(timeout_ms);
      if (byte < 0)
        break;
      data.push_back(static_cast<uint8_t>(byte));
    }
    return data;
  }

private:
  int fd_;
  std::string port_;

  void open_port(int baud_rate) {
    fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
      std::cerr << "Error opening " << port_ << ": " << strerror(errno)
                << std::endl;
      return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(fd_, &tty) != 0) {
      std::cerr << "Error from tcgetattr: " << strerror(errno) << std::endl;
      close(fd_);
      fd_ = -1;
      return;
    }

    cfsetospeed(&tty, baud_rate);
    cfsetispeed(&tty, baud_rate);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit chars
    tty.c_iflag &= ~IGNBRK;                     // disable break processing
    tty.c_lflag = 0;                            // no signaling chars, no echo
    tty.c_oflag = 0;                            // no remapping, no delays
    tty.c_cc[VMIN] = 0;                         // read doesn't block
    tty.c_cc[VTIME] = 5;                        // 0.5 seconds read timeout

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl
    tty.c_cflag |= (CLOCAL | CREAD);        // ignore modem controls
    tty.c_cflag &= ~(PARENB | PARODD);      // no parity
    tty.c_cflag &= ~CSTOPB;                 // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                // no hardware flow control

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      std::cerr << "Error from tcsetattr: " << strerror(errno) << std::endl;
      close(fd_);
      fd_ = -1;
    }
  }
};

#endif // UART_DEVICE_H
