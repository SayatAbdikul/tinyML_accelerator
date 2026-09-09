#!/usr/bin/env python3
"""Convert pinned FLOAT TFLite to ONNX and verify against TensorFlow Lite.

Run in the isolated requirements-conversion.txt environment. Synthetic conversion
probes establish import parity only; they never establish benchmark accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
from onnx.reference import ReferenceEvaluator


def convert(source, output, report):
    interpreter=tf.lite.Interpreter(model_path=str(source),num_threads=1)
    interpreter.allocate_tensors()
    ins=interpreter.get_input_details();outs=interpreter.get_output_details()
    if len(ins)!=1 or ins[0]['dtype']!=np.float32 or any(o['dtype']!=np.float32 for o in outs):
        raise ValueError('conversion parity requires a single-input FLOAT32 source artifact')
    converted,_=tf2onnx.convert.from_tflite(str(source),opset=14)
    # tf2onnx 1.16 emits <=18; ONNX 1.16's reference evaluator only supplies
    # DequantizeLinear from opset 19. Use the official version adapter, not a
    # metadata-only opset relabel, for hybrid float-I/O TFLite artifacts.
    converted=onnx.version_converter.convert_version(converted,19)
    # Freeze the batch-one invocation validated below. Some tf2onnx paths retain
    # an unknown batch from TFLite's shape_signature despite its allocated shape.
    for value in converted.graph.input:
        for dim,size in zip(value.type.tensor_type.shape.dim,ins[0]['shape']):
            dim.ClearField('dim_param');dim.dim_value=int(size)
    del converted.graph.value_info[:]
    for value in converted.graph.output:
        if value.type.tensor_type.shape.dim:
            value.type.tensor_type.shape.dim[0].ClearField('dim_param')
            value.type.tensor_type.shape.dim[0].dim_value=1
    converted=onnx.shape_inference.infer_shapes(converted)
    onnx.checker.check_model(converted)
    reference=ReferenceEvaluator(converted)
    shape=tuple(ins[0]['shape']);rng=np.random.default_rng(20260908)
    samples=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32)]
    samples.extend(rng.uniform(-1,1,shape).astype(np.float32) for _ in range(13))
    results=[]
    for i,x in enumerate(samples):
        interpreter.set_tensor(ins[0]['index'],x);interpreter.invoke()
        expected=[interpreter.get_tensor(t['index']) for t in outs]
        actual=reference.run(None,{converted.graph.input[0].name:x})
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
        results.append({'sample':i,'input_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
                        'max_abs_error':max(float(np.max(np.abs(a-b))) for a,b in zip(actual,expected))})
    # Do not emit a converted model until all source parity probes pass.
    onnx.save(converted,output)
    result={'status':'passed-synthetic-float-artifact-parity','source_format':'float32 TFLite',
            'source_runtime':'TensorFlow Lite '+tf.__version__,'tf2onnx':tf2onnx.__version__,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'atol':1e-5,'rtol':1e-4,'samples':results,
            'operators':sorted(set(n.op_type for n in converted.graph.node)),
            'full_dataset_accuracy_evaluated':False,'keras_to_tflite_parity_evaluated':False}
    report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();convert(a.source,a.output,a.report)
