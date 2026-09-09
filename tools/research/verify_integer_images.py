#!/usr/bin/env python3
"""Compare every layer of a decoded image with the independent integer oracle."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'compiler'))
from program_image import load_image
from static_pipeline import execute_layer
from integer_reference import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('split',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();image=a.image.read_bytes();program=load_image(image)
    with np.load(a.split,allow_pickle=False) as source:
        ids=source['sample_ids'];inputs={n:source[n] for n in program.inputs}
    results=[]
    for i in sorted({0,len(ids)//2,len(ids)-1}):
        qinputs={}
        for n,arr in inputs.items():
            q=program.tensors[n].quantization
            f=arr[i].astype(np.float64)/q.scale
            rounded=np.sign(f)*np.floor(np.abs(f)+.5)
            qinputs[n]=np.maximum(-128,np.minimum(127,rounded+q.zero_point)).astype(np.int8)
        expected=evaluate(program,qinputs)
        actual={n:v.copy() for n,v in program.constants.items()};actual.update(qinputs)
        compared=0
        for layer in program.layers:
            actual[layer.output]=execute_layer(program,layer,actual)
            np.testing.assert_array_equal(actual[layer.output],expected[layer.output],err_msg=layer.output)
            compared+=actual[layer.output].size
        results.append({'sample_id':str(ids[i]),'layers':len(program.layers),'integer_values_compared':compared,'mismatches':0})
    a.report.write_text(json.dumps({'image_sha256':hashlib.sha256(image).hexdigest(),
        'split_sha256':hashlib.sha256(a.split.read_bytes()).hexdigest(),'reference':'independent centered-input integer graph oracle',
        'target':'software-v2','results':results,'status':'passed'},indent=2)+'\n')
