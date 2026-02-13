CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGETS = uart_load_hex uart_memory uart_read_max

all: $(TARGETS)

uart_load_hex: uart_load_hex.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

uart_memory: uart_memory.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

uart_read_max: uart_read_max.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
