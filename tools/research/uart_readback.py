#!/usr/bin/env python3
"""Verify the minimal loopback bitstream; requires explicit serial device."""
import argparse
import hashlib
import json
import random
import time
import serial

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    rng=random.Random(20260908)
    data=bytes(range(256))+bytes(rng.randrange(256) for _ in range(1024))
    start=time.monotonic()
    with serial.Serial(args.port,115200,timeout=1,write_timeout=1) as device:
        device.reset_input_buffer()
        for index,byte in enumerate(data):
            device.write(bytes([byte])); response=device.read(1)
            if response!=bytes([byte]):
                raise RuntimeError(f'UART mismatch/timeout at byte {index}: {response.hex()}')
    with open(args.report,'w') as f:
        json.dump(dict(status='measured-pass',port=args.port,baud=115200,bytes=len(data),
                       payload_sha256=hashlib.sha256(data).hexdigest(),seconds=time.monotonic()-start),f,indent=2)
