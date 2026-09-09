"""Aligned, self-describing v2 software VM images; never legacy FPGA binaries."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import numpy as np
from static_pipeline import Program, Tensor, Layer, run
from quantization import Quantization

MAGIC = b'USHQYN2\0'
OPS = ('Gemm','Conv','Relu','Add','Flatten','Reshape','Identity','Transpose',
       'MaxPool','AveragePool','GlobalAveragePool','Clip')
ALIGN = 16


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_program(program):
    """Validate executable parameter widths and raw accumulation bounds on load."""
    if program.provenance.get('target') != 'software-v2' or program.provenance.get('numerics') != 2:
        raise ValueError('program numerical target mismatch')
    for layer in program.layers:
        p = layer.parameters
        if layer.op in ('Gemm','Conv'):
            expected = {'weight','bias','corrected_bias','weight_scales','multiplier','shift'}
            if set(p) != expected:
                raise ValueError('incomplete MAC parameters')
            w = p['weight']
            rank = 2 if layer.op == 'Gemm' else 4
            if w.ndim != rank or w.dtype != np.int8 or not all(w.shape) or np.any(w == -128):
                raise ValueError('invalid symmetric INT8 weights')
            oc = w.shape[0]
            for name,dt in [('bias','<i4'),('corrected_bias','<i4'),('multiplier','<i4'),('shift','i1'),('weight_scales','<f8')]:
                if p[name].shape != (oc,) or p[name].dtype != np.dtype(dt):
                    raise ValueError(f'invalid {name} shape/type')
            if (np.any(p['multiplier'] <= 0) or np.any(p['shift'] < 0) or np.any(p['shift'] > 62)
                    or np.any(p['weight_scales'] <= 0) or not np.isfinite(p['weight_scales']).all()):
                raise ValueError('invalid fixed-point metadata')
            flat = w.astype(np.int64).reshape(oc,-1)
            iq = program.tensors[layer.inputs[0]].quantization
            qb = p['bias'].astype(np.int64)
            corrected = qb - iq.zero_point * flat.sum(axis=1)
            if not np.array_equal(corrected,p['corrected_bias']):
                raise ValueError('bias correction mismatch')
            bound = 128*np.abs(flat).sum(axis=1)+np.abs(corrected)
            if np.any(bound > (1<<31)-1):
                raise ValueError('accumulator range proof failed')
        elif layer.op == 'Add':
            if set(p) != {'add_multiplier','add_shift'} or len(layer.inputs) != 2:
                raise ValueError('invalid Add parameters')
            if (p['add_multiplier'].shape != (2,) or p['add_multiplier'].dtype != np.dtype('<i4')
                    or np.any(p['add_multiplier'] <= 0) or p['add_shift'].shape != (1,)
                    or p['add_shift'].dtype != np.int8 or not 0 <= int(p['add_shift'][0]) <= 62):
                raise ValueError('invalid Add coefficients')
        elif layer.op == 'Clip':
            if (set(p) != {'clip_bounds'} or p['clip_bounds'].shape != (2,)
                    or p['clip_bounds'].dtype != np.int8 or p['clip_bounds'][0] > p['clip_bounds'][1]):
                raise ValueError('invalid Clip parameters')
        elif p:
            raise ValueError('unexpected parameters')


def build_image(program, capacity=8*1024*1024, target='software-v2'):
    validate_program(program)
    if target != 'software-v2' or program.provenance.get('target') != target:
        raise ValueError("image target mismatch; v2 is not executable by current FPGA")
    if capacity <= 0 or capacity > 1 << 24:
        raise ValueError("capacity exceeds 24-bit address space")
    segments, payloads = [], []
    def add(name, kind, arr):
        arr = np.array(arr, copy=True, order='C')
        data = arr.tobytes()
        segments.append(dict(name=name, kind=kind, dtype=arr.dtype.str, shape=list(arr.shape),
                             size=len(data), sha256=hashlib.sha256(data).hexdigest(), offset=0))
        payloads.append(data)
        return len(segments)-1
    tensor_desc = {}
    for name, t in program.tensors.items():
        sid = add(name, 'tensor', program.constants.get(name, np.zeros(t.shape, dtype=np.int8)))
        tensor_desc[name] = dict(shape=list(t.shape), layout=t.layout,
                                scale=t.quantization.scale, zero_point=t.quantization.zero_point,
                                segment=sid, constant=name in program.constants)
    descriptors = []
    for i, layer in enumerate(program.layers):
        if layer.op not in OPS:
            raise ValueError(f"unknown opcode: {layer.op}")
        param_ids = {k: add(f'layer{i}/{k}', k, arr) for k, arr in layer.parameters.items()}
        descriptors.append(dict(op=layer.op, inputs=layer.inputs, output=layer.output,
                                attributes=layer.attributes, parameters=param_ids))
    instructions = np.array([(i << 8) | (OPS.index(l.op)+1) for i,l in enumerate(program.layers)], dtype='<u8')
    code_segment = add('instructions', 'instructions', instructions)
    desc_segment = add('descriptors', 'descriptors', np.frombuffer(_json(descriptors), dtype=np.uint8))
    header = dict(schema=2, numerical_version=2, target=target, capacity=capacity,
                  inputs=program.inputs, outputs=program.outputs, tensors=tensor_desc,
                  provenance=program.provenance, segments=segments,
                  instructions=code_segment, descriptors=desc_segment)
    header_size = 0
    for _ in range(20):
        offset = header_size
        for seg in segments:
            seg['offset'] = offset
            offset += (seg['size'] + ALIGN-1)//ALIGN*ALIGN
        header['image_size'] = offset
        encoded = _json(header)
        new_size = (16 + len(encoded) + ALIGN-1)//ALIGN*ALIGN
        if new_size == header_size:
            break
        header_size = new_size
    else:
        raise ValueError("image header did not converge")
    if offset > capacity:
        raise ValueError(f"image needs {offset} bytes; target capacity is {capacity}")
    image = bytearray(offset)
    image[:16] = MAGIC + struct.pack('<Q', len(encoded))
    image[16:16+len(encoded)] = encoded
    for seg, data in zip(segments, payloads):
        image[seg['offset']:seg['offset']+len(data)] = data
    # Validate before returning or opening an output file.
    load_image(bytes(image), target=target)
    return bytes(image)


def load_image(image, target='software-v2'):
    if target != 'software-v2' or len(image) < 16 or image[:8] != MAGIC:
        raise ValueError("image magic/target mismatch")
    size, = struct.unpack('<Q', image[8:16])
    if size > len(image)-16:
        raise ValueError("truncated header")
    try:
        h = json.loads(image[16:16+size])
        if (h['schema'] != 2 or h['numerical_version'] != 2 or h['target'] != target
                or h['image_size'] != len(image) or len(image) > h['capacity']
                or h['capacity'] > 1 << 24):
            raise ValueError("image version/size/target mismatch")
        end = (16+size+ALIGN-1)//ALIGN*ALIGN
        arrays = []
        names = set()
        for seg in h['segments']:
            start, count = seg['offset'], seg['size']
            if start != end or start % ALIGN or count < 0 or start+count > len(image) or seg['name'] in names:
                raise ValueError("segment overlap, gap, duplicate or bounds violation")
            names.add(seg['name'])
            data = image[start:start+count]
            if hashlib.sha256(data).hexdigest() != seg['sha256']:
                raise ValueError("segment checksum mismatch")
            dt = np.dtype(seg['dtype'])
            if dt.str not in ('|i1','|u1','<i4','<u8','<f8') or any(type(d) is not int or d < 0 for d in seg['shape']):
                raise ValueError("unsupported segment dtype/shape")
            import math
            if math.prod(seg['shape']) * dt.itemsize != count:
                raise ValueError("segment byte count differs from shape")
            arrays.append(np.frombuffer(data, dtype=dt).reshape(seg['shape']).copy())
            end = start + (count+ALIGN-1)//ALIGN*ALIGN
            if any(image[start+count:end]):
                raise ValueError("nonzero alignment padding")
        if end != len(image):
            raise ValueError("unowned image suffix")
        tensors, constants = {}, {}
        for name,t in h['tensors'].items():
            tensors[name] = Tensor(name, tuple(t['shape']), Quantization(t['scale'], t['zero_point']), t['layout'])
            arr = arrays[t['segment']]
            if arr.shape != tuple(t['shape']) or arr.dtype != np.int8:
                raise ValueError("tensor descriptor mismatch")
            if t['constant']:
                constants[name] = arr
        desc = json.loads(arrays[h['descriptors']].tobytes())
        layers = []
        code = arrays[h['instructions']]
        if code.dtype != np.dtype('<u8') or code.ndim != 1 or len(code) != len(desc):
            raise ValueError("instruction count/type mismatch")
        available = set(h['inputs']) | set(constants)
        for i, word in enumerate(code):
            word = int(word)
            if word >> 8 != i or not 1 <= word & 255 <= len(OPS):
                raise ValueError("invalid instruction/descriptor index")
            d = desc[i]
            if d['op'] != OPS[(word & 255)-1] or any(n not in available for n in d['inputs']) or d['output'] not in tensors:
                raise ValueError("invalid descriptor/data dependency")
            layers.append(Layer(d['op'], d['inputs'], d['output'], d['attributes'],
                                {k: arrays[v] for k,v in d['parameters'].items()}))
            available.add(d['output'])
        if not set(h['outputs']) <= available or not set(h['inputs']) <= set(tensors):
            raise ValueError("invalid program inputs/outputs")
        program = Program(tensors,layers,h['inputs'],h['outputs'],constants,h['provenance'])
        validate_program(program)
        return program
    except (KeyError, TypeError, IndexError, OverflowError, UnicodeError) as exc:
        raise ValueError("malformed image metadata") from exc


def write_image(program, path, **options):
    data = build_image(program, **options)
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def simulate_image(image, inputs, **options):
    return run(load_image(image), inputs, **options)
