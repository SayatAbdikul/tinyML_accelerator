#!/usr/bin/env python3
"""Freeze a legacy TensorFlow SavedModel and validate its ONNX conversion.

Uses the graph/session loader to avoid Keras 3 legacy-optimizer revival failures.
No weights are retrained or replaced. Batch one is the declared target contract.
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


def convert(source,output,report):
    with tf.Graph().as_default() as graph:
      with tf.compat.v1.Session(graph=graph) as session:
        meta=tf.compat.v1.saved_model.loader.load(session,[tf.saved_model.SERVING],str(source))
        signature=meta.signature_def['serving_default']
        if len(signature.inputs)!=1:raise ValueError('expected one SavedModel input')
        input_info=next(iter(signature.inputs.values()))
        input_name=input_info.name
        output_names=[signature.outputs[k].name for k in sorted(signature.outputs)]
        shape=[1 if d.size==-1 and i==0 else d.size for i,d in enumerate(input_info.tensor_shape.dim)]
        if any(d<=0 for d in shape):raise ValueError('dynamic nonbatch input dimensions')
        frozen=tf.compat.v1.graph_util.convert_variables_to_constants(session,graph.as_graph_def(),
                [n.split(':')[0] for n in output_names])
        converted,_=tf2onnx.convert.from_graph_def(frozen,input_names=[input_name],output_names=output_names,
                opset=14,shape_override={input_name:shape})
        converted=onnx.shape_inference.infer_shapes(converted)
        onnx.checker.check_model(converted);reference=ReferenceEvaluator(converted)
        rng=np.random.default_rng(20260908)
        samples=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32)]
        samples.extend(rng.uniform(-20,20,shape).astype(np.float32) for _ in range(13))
        results=[]
        for index,x in enumerate(samples):
            expected=session.run(output_names,{input_name:x})
            actual=reference.run(None,{converted.graph.input[0].name:x})
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
            results.append({'sample':index,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                            'max_abs_error':max(float(np.max(np.abs(a-b))) for a,b in zip(actual,expected))})
    onnx.save(converted,output)
    source_files={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source.rglob('*')) if p.is_file()}
    result={'status':'passed-synthetic-source-framework-parity','source_format':'TensorFlow SavedModel',
            'source_files_sha256':source_files,'source_runtime':'TensorFlow '+tf.__version__,
            'tf2onnx':tf2onnx.__version__,'onnx_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'atol':1e-5,'rtol':1e-4,'samples':results,'full_dataset_accuracy_evaluated':False,
            'operators':sorted(set(n.op_type for n in converted.graph.node))}
    report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();convert(a.source,a.output,a.report)
