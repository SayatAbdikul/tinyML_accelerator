#!/usr/bin/env python3
"""Inventory pinned TFLite artifacts without executing their embedded model."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import tflite


def options_dict(operator):
    tag = operator.BuiltinOptionsType()
    kinds = {v:k for k,v in vars(tflite.BuiltinOptions).items() if isinstance(v,int)}
    name = kinds.get(tag)
    if not tag:
        return {}
    if not name or not hasattr(tflite,name):
        raise ValueError(f"unknown builtin options {tag}")
    obj = getattr(tflite,name)(); tab = operator.BuiltinOptions()
    obj.Init(tab.Bytes,tab.Pos)
    out = {}
    for key in dir(obj):
        fn = getattr(obj,key)
        if key.startswith('_') or key.endswith(('Length','AsNumpy','IsNone')) or key in ('Init',):
            continue
        if not callable(fn) or len(inspect.signature(fn).parameters):
            continue
        value = fn()
        if isinstance(value,(int,float,bool,str)):
            out[key] = value
    return {'type':name,'fields':out}


def inventory(path):
    data=Path(path).read_bytes(); model=tflite.Model.GetRootAsModel(data,0)
    op_names={v:k for k,v in vars(tflite.BuiltinOperator).items() if isinstance(v,int)}
    types={v:k for k,v in vars(tflite.TensorType).items() if isinstance(v,int)}
    if model.SubgraphsLength()!=1:
        raise ValueError("multi-subgraph benchmark requires explicit control-flow inventory")
    graph=model.Subgraphs(0); tensors=[]
    for i in range(graph.TensorsLength()):
        t=graph.Tensors(i); q=t.Quantization(); b=model.Buffers(t.Buffer())
        tensors.append(dict(id=i,name=t.Name().decode(),shape=[int(v) for v in t.ShapeAsNumpy()],
            dtype=types[t.Type()],buffer_bytes=b.DataLength(),
            quantization=None if q is None else dict(scales=[] if not q.ScaleLength() else q.ScaleAsNumpy().tolist(),
                zero_points=[] if not q.ZeroPointLength() else q.ZeroPointAsNumpy().tolist(),axis=q.QuantizedDimension())))
    ops=[]; total_macs=0
    for i in range(graph.OperatorsLength()):
        op=graph.Operators(i); code=model.OperatorCodes(op.OpcodeIndex())
        name=op_names.get(code.BuiltinCode(),'UNKNOWN')
        ins=op.InputsAsNumpy().tolist(); outs=op.OutputsAsNumpy().tolist()
        macs=0
        if name in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED'):
            w=tensors[ins[1]]['shape']; y=tensors[outs[0]]['shape']
            reduction=np.prod(w[1:]) if name!='DEPTHWISE_CONV_2D' else w[1]*w[2]
            macs=int(np.prod(y)*reduction)
        total_macs+=macs
        ops.append(dict(index=i,op=name,version=code.Version(),inputs=ins,outputs=outs,
                        attributes=options_dict(op),macs=macs))
    return dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
                format='TFLite',layout='NHWC; weight layout is operator-specific',
                inputs=graph.InputsAsNumpy().tolist(),outputs=graph.OutputsAsNumpy().tolist(),
                tensors=tensors,operators=ops,total_macs=total_macs,
                constant_buffer_bytes=sum(t['buffer_bytes'] for t in tensors),
                largest_tensor_elements=max(int(np.prod(t['shape'])) for t in tensors),
                unknown_operators=[o['op'] for o in ops if o['op'] in ('UNKNOWN','CUSTOM')])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.write_text(json.dumps(inventory(args.model),indent=2)+'\n')
