# Makefile for UART Tools

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGETS = uart_loader uart_reader uart_verify

all: $(TARGETS)

uart_loader: uart_loader.cpp uart_device.h
	$(CXX) $(CXXFLAGS) -o $@ $<

uart_reader: uart_reader.cpp uart_device.h
	$(CXX) $(CXXFLAGS) -o $@ $<

uart_verify: uart_verify.cpp uart_device.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
