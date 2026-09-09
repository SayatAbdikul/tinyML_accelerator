#!/usr/bin/env python3
"""Evaluate a v2 classifier on a content-pinned preprocessed full split.

Rejects calibration overlap by both ID and content. This tool does not certify
that a caller-provided split is an official benchmark: pin evaluator/preprocessing
and publish its split manifest as required by R02 before claiming benchmark quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'compiler'))
from program_image import load_image
from static_pipeline import run


def evaluate(program,data,expected_sha256):
    if hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError('evaluation split hash mismatch')
    import io
    with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
        ids=arrays['sample_ids'].tolist();labels=arrays['labels']
        if not ids or len(set(ids))!=len(ids) or len(labels)!=len(ids):
            raise ValueError('invalid evaluation IDs/labels')
        cal=program.provenance['calibration']
        if set(ids)&set(cal['sample_ids']):raise ValueError('calibration/evaluation ID overlap')
        if len(program.outputs)!=1:raise ValueError('classifier must have one output')
        correct=0
        input_arrays={n:arrays[n] for n in program.inputs}
        if any(len(a)!=len(ids) for a in input_arrays.values()):
            raise ValueError('evaluation input count mismatch')
        for i,label in enumerate(labels):
            inputs={n:input_arrays[n][i].astype(np.float32) for n in program.inputs}
            h=hashlib.sha256()
            for name,arr in inputs.items():h.update(name.encode());h.update(arr.astype('<f4').tobytes())
            if h.hexdigest() in cal['sample_sha256']:raise ValueError('calibration/evaluation content overlap')
            pred=run(program,inputs)[program.outputs[0]]
            if pred.ndim!=2 or pred.shape[0]!=1 or not 0<=int(label)<pred.shape[1]:
                raise ValueError('invalid classifier output/label')
            correct+=int(np.argmax(pred)==int(label))
            if (i+1)%500==0:
                print(f'evaluated {i+1}/{len(ids)}; correct={correct}',flush=True)
    return dict(count=len(ids),correct=correct,accuracy=correct/len(ids),split_sha256=expected_sha256,
                calibration_disjoint=True,target='software-v2',official_benchmark_certified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('split',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();result=evaluate(load_image(a.image.read_bytes()),a.split.read_bytes(),a.sha256)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
