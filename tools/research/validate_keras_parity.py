#!/usr/bin/env python3
"""Add original Keras H5 source parity evidence to a converted artifact report."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import tf_keras

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('onnx',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();model=tf_keras.models.load_model(a.source,compile=False);converted=onnx.load(a.onnx)
    runtime=ReferenceEvaluator(converted);shape=[d.dim_value for d in converted.graph.input[0].type.tensor_type.shape.dim]
    rng=np.random.default_rng(20260908);samples=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32)]
    samples.extend(rng.uniform(-1,1,shape).astype(np.float32) for _ in range(13));errors=[]
    for x in samples:
        expected=model(x,training=False).numpy();actual=runtime.run(None,{converted.graph.input[0].name:x})[0]
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4);errors.append(float(np.max(np.abs(actual-expected))))
    report=json.loads(a.report.read_text())
    report['keras_source_parity']={'status':'passed','source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
            'runtime':'tf_keras '+tf_keras.__version__,'samples':len(samples),'max_abs_errors':errors,'atol':1e-5,'rtol':1e-4}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
