; Custom Architecture Assembly Code
; Generated from ONNX model

LOAD_V 3, 0x4c0, 1
LOAD_V 9, 0xc0, 784
LOAD_M 1, 0x940, 12, 784
LOAD_V 4, 0x4c1, 12
GEMV 5, 1, 9, 4, 12, 784
RELU 7, 5, 12
LOAD_M 2, 0x2e00, 32, 16
LOAD_V 3, 0x4cd, 32
GEMV 6, 2, 7, 3, 32, 12
RELU 8, 6, 32
LOAD_M 1, 0x3000, 10, 32
LOAD_V 4, 0x4ed, 10
GEMV 5, 1, 8, 4, 10, 32
STORE 5, 0x8c0, 10