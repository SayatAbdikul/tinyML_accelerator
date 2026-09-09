#!/usr/bin/env python3
"""Convert the original H5 model, avoiding hybrid float-I/O TFLite variants."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras
import tf2onnx
import onnx
from onnx.reference import ReferenceEvaluator

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();model=tf_keras.models.load_model(a.source,compile=False)
    shape=[1]+list(model.input_shape[1:]);signature=[tf.TensorSpec(shape,tf.float32,name=model.inputs[0].name.split(':')[0])]
    converted,_=tf2onnx.convert.from_keras(model,input_signature=signature,opset=14)
    converted=onnx.shape_inference.infer_shapes(converted);onnx.checker.check_model(converted)
    runtime=ReferenceEvaluator(converted);rng=np.random.default_rng(20260908)
    samples=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32)]
    samples.extend(rng.uniform(-1,1,shape).astype(np.float32) for _ in range(13));results=[]
    for i,x in enumerate(samples):
        expected=model(x,training=False).numpy();actual=runtime.run(None,{converted.graph.input[0].name:x})[0]
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
        results.append({'sample':i,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'max_abs_error':float(np.max(np.abs(actual-expected)))})
    onnx.save(converted,a.output)
    a.report.write_text(json.dumps({'status':'passed-synthetic-source-framework-parity','source_format':'Keras H5',
        'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'source_runtime':'tf_keras '+tf_keras.__version__,'tf2onnx':tf2onnx.__version__,'atol':1e-5,'rtol':1e-4,
        'samples':results,'operators':sorted(set(n.op_type for n in converted.graph.node)),'full_dataset_accuracy_evaluated':False},indent=2)+'\n')
