#!/usr/bin/env python3
"""Classify a complete pinned preprocessed split with its original float model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras


def evaluate(source,split,input_name,report):
    with np.load(split,allow_pickle=False) as f:
        xs=f[input_name];labels=f['labels'];ids=f['sample_ids']
    if len(xs)!=len(labels) or len(set(ids.tolist()))!=len(ids):raise ValueError('invalid split')
    # Stored samples include their batch-one axis; join that axis for batched TF.
    xs=xs[:,0]
    correct=0
    if source.is_dir():
      with tf.Graph().as_default() as graph:
       with tf.compat.v1.Session(graph=graph) as session:
        meta=tf.compat.v1.saved_model.loader.load(session,[tf.saved_model.SERVING],str(source))
        sig=meta.signature_def['serving_default']
        inp=next(iter(sig.inputs.values())).name;out=next(iter(sig.outputs.values())).name
        for start in range(0,len(xs),128):
            values=session.run(out,{inp:xs[start:start+128]})
            correct+=int(np.sum(np.argmax(values,axis=1)==labels[start:start+128]))
    else:
        model=tf_keras.models.load_model(source,compile=False)
        for start in range(0,len(xs),128):
            values=model(xs[start:start+128],training=False).numpy()
            correct+=int(np.sum(np.argmax(values,axis=1)==labels[start:start+128]))
    result={'count':len(xs),'correct':correct,'accuracy':correct/len(xs),
            'arithmetic':'original float model','split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),
            'runtime':'TensorFlow '+tf.__version__,'official_benchmark_certified':False}
    report.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('split',type=Path);p.add_argument('--input-name',required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();evaluate(a.source,a.split,a.input_name,a.report)
