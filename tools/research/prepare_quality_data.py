#!/usr/bin/env python3
"""Prepare deterministic, disjoint primary calibration and full accuracy splits.

KWS: canonical 4,890-file Speech Commands v2 test archive, and 96 training
calibration clips. MFCC code is taken verbatim from the pinned upstream function.
VWW: full per-class first-10% validation split of the pinned preprocessed archive,
matching ImageDataGenerator's filename partition; upstream calibration images.
The VWW split is explicitly a training-recipe validation set, not a certification
of an official MLPerf submission. No random augmentation is used for accuracy.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave
import numpy as np
import tensorflow as tf
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def upstream_function(path,name,scope):
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def kws(upstream,train,test):
    source=upstream/'benchmark/training/keyword_spotting'
    scope={'tf':tf,'np':np}
    settings_fn=upstream_function(source/'keras_model.py','prepare_model_settings',scope)
    settings=settings_fn(12,SimpleNamespace(sample_rate=16000,clip_duration_ms=1000,window_size_ms=30.,window_stride_ms=20.,feature_type='mfcc',dct_coefficient_count=10))
    upstream_preprocess=upstream_function(source/'get_dataset.py','get_preprocess_audio_func',scope)(settings,is_training=False)
    def prepare_copy(example):
        # The upstream function updates its dictionary. Modern tf.function
        # rejects mutation of a caller's argument, so give it a private copy.
        return upstream_preprocess(dict(example))
    preprocess=tf.function(prepare_copy,input_signature=[{'audio':tf.TensorSpec([None],tf.int16),'label':tf.TensorSpec([],tf.int64)}])
    labels={'down':0,'go':1,'left':2,'no':3,'off':4,'on':5,'right':6,'stop':7,'up':8,'yes':9,'_silence_':10,'_unknown_':11}
    def wave_data(p):
        with wave.open(str(p),'rb') as f:
            if f.getframerate()!=16000 or f.getnchannels()!=1 or f.getsampwidth()!=2:raise ValueError('unexpected WAV format')
            return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
    evaluation=[(p,labels[p.parent.name],0) for p in sorted(test.rglob('*.wav'))]
    if len(evaluation)!=4890:raise ValueError('canonical Speech Commands v2 test must contain 4890 WAV files')
    excluded=set()
    for name in ['validation_list.txt','testing_list.txt']:
        excluded.update((train/name).read_text().splitlines())
    calibration=[]
    for word,label in labels.items():
        if word.startswith('_'):continue
        selected=[p for p in sorted((train/word).glob('*.wav')) if str(p.relative_to(train)) not in excluded][:8]
        if len(selected)!=8:raise ValueError('missing training calibration words')
        calibration.extend((p,label,0) for p in selected)
    unknown=[p for p in sorted(train.glob('*/*.wav')) if p.parent.name not in labels and not p.parent.name.startswith('_') and str(p.relative_to(train)) not in excluded]
    # Spread unknown examples across eight distinct words.
    words=[]
    for p in unknown:
        if p.parent.name not in words:
            calibration.append((p,11,0));words.append(p.parent.name)
        if len(words)==8:break
    noise=sorted((train/'_background_noise_').glob('*.wav'))
    for i in range(8):calibration.append((noise[i%len(noise)],10,(i//len(noise))*16000))
    def materialize(items,root,split):
        records=[];features=[];ys=[];ids=[]
        for i,(p,label,offset) in enumerate(items):
            audio=wave_data(p)
            if split=='calibration':audio=audio[offset:offset+16000]
            value=preprocess({'audio':tf.convert_to_tensor(audio),'label':tf.constant(label,tf.int64)})['audio'].numpy()[None]
            if not np.isfinite(value).all():raise ValueError(f'nonfinite upstream MFCC: {p}')
            id=f'kws/{split}/{p.relative_to(root)}#{offset}'
            features.append(value);ys.append(label);ids.append(id)
            records.append({'id':id,'path':str(p.relative_to(root)),'offset_samples':offset,'raw_sha256':sha(p),'feature_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'label':label})
            if (i+1)%500==0:print(f'KWS {split}: {i+1}/{len(items)}',flush=True)
        return np.stack(features),np.array(ys),ids,records
    return materialize(calibration,train,'calibration'),materialize(evaluation,test,'accuracy'),{'name':'Speech Commands v2 canonical test','preprocessing_source_sha256':sha(source/'get_dataset.py'),'calibration_policy':'8 training clips per keyword, 8 distinct unknown words, 8 noise windows; official train/validation exclusions respected'}


def vww(upstream,root):
    classes={'non_person':0,'person':1};evaluation=[];validation_paths=set()
    all_paths={p.name:p for p in root.glob('*/*.jpg')}
    for folder,label in classes.items():
        paths=sorted((root/folder).glob('*.jpg'))
        selected=paths[:int(len(paths)*.1)]
        evaluation.extend((p,label) for p in selected);validation_paths.update(selected)
    source=upstream/'benchmark/training/visual_wake_words/calibration_data.txt'
    names=[line.strip() for line in source.read_text().splitlines() if line.startswith('COCO_')]
    calibration=[(all_paths[n],classes[all_paths[n].parent.name]) for n in names]
    if any(p in validation_paths for p,_ in calibration):raise ValueError('upstream calibration image overlaps validation')
    def materialize(items,split):
        records=[];features=np.empty((len(items),1,96,96,3),np.float32);ys=[];ids=[]
        for i,(p,label) in enumerate(items):
            with Image.open(p) as im:
                if im.size!=(96,96):raise ValueError('archive image must already be 96x96')
                value=np.asarray(im.convert('RGB'),np.float32)[None]/255.
            features[i]=value;ys.append(label);id=f'vww/{p.relative_to(root)}';ids.append(id)
            records.append({'id':id,'path':str(p.relative_to(root)),'raw_sha256':sha(p),'feature_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'label':label})
            if (i+1)%2000==0:print(f'VWW {split}: {i+1}/{len(items)}',flush=True)
        return features,np.array(ys),ids,records
    return materialize(calibration,'calibration'),materialize(evaluation,'accuracy'),{'name':'Silicon Labs VWW archive complete first-10%-per-class validation partition','partition_source_sha256':sha(upstream/'benchmark/training/visual_wake_words/train_vww.py'),'accuracy_augmentation':'none','official_mlperf_accuracy_split_certified':False,'calibration_policy':'all images in upstream calibration_data.txt'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['kws','vww']);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--test',type=Path);p.add_argument('--input-name',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    cal,test,metadata=kws(a.upstream,a.data,a.test) if a.kind=='kws' else vww(a.upstream,a.data)
    if set(cal[2])&set(test[2]) or set(r['feature_sha256'] for r in cal[3])&set(r['feature_sha256'] for r in test[3]):raise ValueError('calibration/accuracy overlap')
    reports={}
    for split,values in [('calibration',cal),('accuracy',test)]:
        features,labels,ids,records=values
        target=a.output_dir/(a.kind+'.'+split+'.npz')
        np.savez_compressed(target,**{a.input_name:features,'sample_ids':np.asarray(ids),'labels':labels})
        reports[split]={'count':len(ids),'npz_sha256':sha(target),'records':records}
    report={'workload':a.kind,'metadata':metadata,'preparation_script_sha256':sha(Path(__file__)),'input_name':a.input_name,'splits':reports,'calibration_accuracy_disjoint':True}
    (a.output_dir/(a.kind+'.data.json')).write_text(json.dumps(report,indent=2)+'\n')
