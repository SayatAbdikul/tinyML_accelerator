#!/usr/bin/env python3
"""Inventory the actual converted/canonical float graph, separate from TFLite."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'compiler'))
from canonicalize import canonicalize, shapes, attrs
from classifier_boundary import logits_model


def inventory(path, classifier=False):
    source = onnx.load(path)
    boundary = None
    if classifier:
        source, boundary = logits_model(source)
    model = canonicalize(source)
    dimensions = shapes(model)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        attributes = attrs(node)
        macs = 0
        if node.op_type == 'Conv':
            w = constants[node.input[1]]
            macs = int(np.prod(dimensions[node.output[0]]) * np.prod(w.shape[1:]))
        elif node.op_type == 'Gemm':
            macs = int(np.prod(dimensions[node.output[0]]) * dimensions[node.input[0]][1])
        nodes.append(dict(name=node.name, op=node.op_type, inputs=list(node.input),
                          outputs=list(node.output), attributes=attributes, macs=macs))
    tensors = []
    infos = {v.name: v for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)}
    for name, shape in dimensions.items():
        dtype = str(constants[name].dtype) if name in constants else onnx.TensorProto.DataType.Name(infos[name].type.tensor_type.elem_type)
        tensors.append(dict(name=name, shape=shape, dtype=dtype, constant=name in constants,
                            logical_bytes=int(constants[name].nbytes if name in constants else np.prod(shape)*4)))
    return dict(source_onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                canonical_onnx_sha256=hashlib.sha256(model.SerializeToString()).hexdigest(),
                classifier_boundary=boundary, operators=nodes, tensors=tensors,
                total_macs=sum(n['macs'] for n in nodes),
                constant_bytes=sum(a.nbytes for a in constants.values()),
                storage_scope='canonical float tensors; excludes physical banking and v2 quantization metadata',
                unknown_required_operators=[])


def json_value(value):
    if isinstance(value, bytes): return value.decode('utf8')
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path); p.add_argument('report', type=Path)
    p.add_argument('--classifier', action='store_true')
    a = p.parse_args()
    a.report.write_text(json.dumps(inventory(a.model, a.classifier), indent=2, default=json_value)+'\n')
