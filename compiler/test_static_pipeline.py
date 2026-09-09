"""Independent numerical/ONNX regression tests. Oracle uses Python scalar ints."""
import copy
import json
import struct
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.reference import ReferenceEvaluator
import pytest
from canonicalize import canonicalize
from static_pipeline import calibrate, compile_static, run
from quantization import Quantization, requantize, multiplier_shift, quantize_parameters
from program_image import build_image, load_image, simulate_image, write_image


def model(nodes, weights, input_shape, output_shapes):
    return h.make_model(h.make_graph(nodes, 'regression',
        [h.make_tensor_value_info('x', T.FLOAT, input_shape)],
        [h.make_tensor_value_info(n, T.FLOAT, s) for n,s in output_shapes.items()],
        [nh.from_array(np.asarray(a,dtype=np.float32), n) for n,a in weights.items()]),
        opset_imports=[h.make_opsetid('',14)])


def compile_case(m,x):
    cal = calibrate(m,[{'x':x}],['directed-calibration-0'])
    return compile_static(m,cal)


def test_exported_dense_bn_affine_fold_preserves_shared_outputs():
    m=model([h.make_node('Gemm',['x','w','b'],['dense'],transB=1),
             h.make_node('Mul',['scale','dense'],['scaled']),
             h.make_node('Add',['scaled','offset'],['y'])],
            {'w':[[1,2],[-3,4]],'b':[.25,-.5],
             'scale':[-2,0],'offset':[3,4]},[1,2],{'y':[1,2]})
    x=np.array([[.4,-.7]],np.float32)
    folded=canonicalize(m)
    assert [n.op_type for n in folded.graph.node]==['Gemm']
    np.testing.assert_allclose(ReferenceEvaluator(folded).run(None,{'x':x})[0],
                               ReferenceEvaluator(m).run(None,{'x':x})[0],atol=1e-6)
    # An observable producer cannot be changed to satisfy its consumer.
    m.graph.output.append(h.make_tensor_value_info('dense',T.FLOAT,[1,2]))
    with pytest.raises(ValueError,match='unsupported operator'):
        canonicalize(m)


def test_legacy_cli_rejects_before_writing_payloads(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).with_name('compile.py')
    result=subprocess.run([sys.executable,str(script),'missing.onnx','output.asm'],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode!=0 and 'static_cli.py' in result.stderr
    assert list(tmp_path.iterdir())==[]


def scalar_round(numerator, shift, zp):
    """Independent integer division oracle, no production rounding/helper calls."""
    denominator = 2**int(shift)
    quotient, remainder = divmod(abs(int(numerator)), denominator)
    if 2 * remainder >= denominator:
        quotient += 1
    return max(-128, min(127, (-quotient if numerator < 0 else quotient) + zp))


@pytest.mark.parametrize('shift',[0,1,2,7,31,62])
@pytest.mark.parametrize('zp',[-128,-3,0,127])
def test_rounding_independent(shift,zp):
    vals = np.array([-(2**31),-257,-255,-129,-3,-1,0,1,3,127,255,2**31-1],np.int64)
    for multiplier in [1,3,2**30,2**31-1]:
        expected=[scalar_round(int(v)*multiplier,shift,zp) for v in vals]
        np.testing.assert_array_equal(requantize(vals,multiplier,shift,zp),expected)


@pytest.mark.parametrize('bias',[1.,100.])
def test_bias_units_class_flip_and_image(bias):
    m=model([h.make_node('Gemm',['x','w','b'],['y'],transB=1)],
            {'w':np.eye(2)*.1,'b':[0,bias]},[1,2],{'y':[1,2]})
    x=np.array([[1,0]],np.float32); p=compile_case(m,x)
    expected=ReferenceEvaluator(m).run(None,{'x':x})[0]
    actual=run(p,{'x':x})['y']
    assert np.argmax(actual)==np.argmax(expected)==1
    par=p.layers[0].parameters
    qx=p.tensors['x'].quantization.encode(x)
    zp=p.tensors['y'].quantization.zero_point
    independent=[]
    for c in range(2):
        acc=sum(int(qx[0,k])*int(par['weight'][c,k]) for k in range(2))+int(par['corrected_bias'][c])
        independent.append(scalar_round(acc*int(par['multiplier'][c]),par['shift'][c],zp))
    np.testing.assert_array_equal(actual,[independent])
    np.testing.assert_array_equal(simulate_image(build_image(p),{'x':x})['y'],actual)
    assert par['bias'].dtype==np.dtype('<i4')


def test_bn_fold_parity_negative_scale_and_no_bias():
    m=model([h.make_node('Conv',['x','w'],['c']),
             h.make_node('BatchNormalization',['c','gamma','beta','mean','var'],['y'],epsilon=.001)],
            {'w':np.ones((2,1,1,1)),'gamma':[-1,2],'beta':[3,-2],'mean':[.2,.3],'var':[1,4]},
            [1,1,2,2],{'y':[1,2,2,2]})
    x=np.arange(4,dtype=np.float32).reshape(1,1,2,2)
    c=canonicalize(m)
    assert [n.op_type for n in c.graph.node]==['Conv']
    np.testing.assert_allclose(ReferenceEvaluator(c).run(None,{'x':x})[0],ReferenceEvaluator(m).run(None,{'x':x})[0],atol=1e-6)
    p=compile_case(m,x)
    assert p.layers[0].parameters['bias'].shape==(2,)


def test_standalone_bn_rejected():
    m=model([h.make_node('BatchNormalization',['x','s','b','m','v'],['y'])],
            {'s':[-1,1],'b':[0,0],'m':[0,0],'v':[1,1]},[1,2],{'y':[1,2]})
    with pytest.raises(ValueError,match='BN must fold'): canonicalize(m)


def test_large_relu_and_all_outputs():
    m=model([h.make_node('Relu',['x'],['r']),h.make_node('Flatten',['r'],['y'])],{},
            [1,1,32,32],{'r':[1,1,32,32],'y':[1,1024]})
    x=np.linspace(-4,4,1024,dtype=np.float32).reshape(1,1,32,32);p=compile_case(m,x)
    out=simulate_image(build_image(p),{'x':x})
    assert set(out)=={'r','y'}
    assert np.all(out['r'].ravel()[:512]==p.tensors['r'].quantization.zero_point)
    np.testing.assert_array_equal(out['r'].reshape(1,-1),out['y'])


def test_terminal_add_and_bias_free_conv():
    m=model([h.make_node('Conv',['x','w'],['c']),h.make_node('Add',['c','b'],['y'])],
            {'w':np.ones((1,1,1,1)),'b':np.ones((1,1,1,1))*2},[1,1,2,2],{'y':[1,1,2,2]})
    x=np.ones((1,1,2,2),np.float32);p=compile_case(m,x)
    # Constant Add is folded into the final bias; bias-free Conv itself still
    # gets an explicit zero vector when no Add follows it.
    bare=model([h.make_node('Conv',['x','w'],['y'])],{'w':np.ones((1,1,1,1))},[1,1,2,2],{'y':[1,1,2,2]})
    assert np.all(compile_case(bare,x).layers[0].parameters['bias']==0)
    out=run(p,{'x':x})['y']; q=p.tensors['y'].quantization
    np.testing.assert_allclose(q.decode(out),3,atol=q.scale*2)


@pytest.mark.parametrize('groups',[1,2,4])
@pytest.mark.parametrize('zero_point',[-128,0,127])
def test_multichannel_rectangular_conv_independent(groups,zero_point):
    rng=np.random.default_rng(912+groups)
    w=rng.normal(size=(4,4//groups,2,3)).astype(np.float32)
    m=model([h.make_node('Conv',['x','w','b'],['y'],group=groups,strides=[2,1],pads=[1,0,0,2],dilations=[1,2])],
            {'w':w,'b':rng.normal(size=4)},[1,4,5,7],{'y':[1,4,3,5]})
    x=rng.normal(size=(1,4,5,7)).astype(np.float32);p=compile_case(m,x)
    # Exercise extreme input zero points by explicitly recompiling parameters.
    from static_pipeline import Tensor
    iq=Quantization(.03,zero_point); oq=p.tensors['y'].quantization
    p.tensors['x']=Tensor('x',x.shape,iq,'NCHW')
    bias=nh.to_array(next(t for t in m.graph.initializer if t.name=='b'))
    p.layers[0].parameters=quantize_parameters(w,bias,iq,oq)
    qx=rng.integers(-128,128,size=x.shape,dtype=np.int8)
    par=p.layers[0].parameters; expected=np.empty((1,4,3,5),np.int8)
    for c in range(4):
      for y in range(3):
       for z in range(5):
        acc=int(par['bias'][c])
        for ic in range(4//groups):
         for ky in range(2):
          for kx in range(3):
           iy,ix=y*2-1+ky,z+kx*2
           q=int(qx[0,(c//(4//groups))*(4//groups)+ic,iy,ix]) if 0<=iy<5 and 0<=ix<7 else zero_point
           acc+=(q-zero_point)*int(par['weight'][c,ic,ky,kx])
        expected[0,c,y,z]=scalar_round(acc*int(par['multiplier'][c]),par['shift'][c],oq.zero_point)
    np.testing.assert_array_equal(run(p,{'x':qx},quantized=True)['y'],expected)
    np.testing.assert_array_equal(simulate_image(build_image(p),{'x':qx},quantized=True)['y'],expected)


def test_gemm_transpose_alpha_beta_and_broadcast_parity():
    m=model([h.make_node('Gemm',['x','w','b'],['y'],alpha=.5,beta=2.)],
            {'w':np.arange(6).reshape(2,3),'b':[[1,2,3]]},[1,2],{'y':[1,3]})
    x=np.array([[.2,-.4]],np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(canonicalize(m)).run(None,{'x':x}),ReferenceEvaluator(m).run(None,{'x':x}),atol=1e-6)


@pytest.mark.parametrize('op,attributes',[('Sin',{}),('MaxPool',{'kernel_shape':[2,2],'ceil_mode':1})])
def test_unsupported_rejected(op,attributes):
    m=model([h.make_node(op,['x'],['y'],**attributes)],{},[1,1,2,2],{'y':[1,1,2,2]})
    with pytest.raises((ValueError,onnx.shape_inference.InferenceError)): canonicalize(m)


def test_metadata_overflow_and_atomic_image_rejection(tmp_path):
    m=model([h.make_node('Gemm',['x','w'],['y'],transB=1)],{'w':np.eye(2)},[1,2],{'y':[1,2]})
    x=np.ones((1,2),np.float32);p=compile_case(m,x); blob=build_image(p)
    cal=copy.deepcopy(p.provenance['calibration']);cal['model_sha256']='wrong'
    with pytest.raises(ValueError,match='hash mismatch'):compile_static(m,cal)
    with pytest.raises(ValueError,match='FPGA'):compile_static(m,p.provenance['calibration'],target='tang-nano-20k')
    path=tmp_path/'image.bin';path.write_bytes(b'previous-good-image')
    with pytest.raises(ValueError,match='capacity'):write_image(p,path,capacity=16)
    assert path.read_bytes()==b'previous-good-image'
    corrupt=bytearray(blob);corrupt[-16]^=1
    with pytest.raises(ValueError):load_image(corrupt)
    with pytest.raises(ValueError):load_image(blob,target='tang-nano-20k')
    for bad in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):Quantization(bad,0)
    with pytest.raises(ValueError):quantize_parameters(np.ones((2,2)),np.array([1e30,0]),Quantization(.01,0),Quantization(.01,0))


def test_unsafe_legacy_entry_rejects_without_artifact(tmp_path):
    from compile import generate_assembly
    path=tmp_path/'out.asm'
    with pytest.raises(ValueError,match='calibrated'):generate_assembly('unused.onnx',path)
    assert not path.exists()


def test_independent_bias_and_per_channel_parameter_units():
    iq,oq=Quantization(.25,-128),Quantization(.5,3)
    w=np.array([[.5,-1.],[.01,.02]],np.float64)
    b=np.array([.75,-.005])
    p=quantize_parameters(w,b,iq,oq)
    scales=[1/127,.02/127]
    def away(x):
        import math
        return (1 if x>=0 else -1)*math.floor(abs(x)+.5)
    assert p['bias'].tolist()==[away(b[i]/(.25*scales[i])) for i in range(2)]
    assert p['weight'].tolist()==[[away(float(w[i,j])/scales[i]) for j in range(2)] for i in range(2)]


def test_near_zero_folded_channel_bias_stays_representable():
    iq,oq=Quantization(.01,-128),Quantization(.02,0)
    p=quantize_parameters(np.full((2,8),1e-20),np.array([1.,-1.]),iq,oq)
    assert np.all(p['weight']==0)
    assert np.max(np.abs(p['bias'].astype(np.int64))) <= 2**31-1
    out=[int(requantize(p['bias'][c:c+1],p['multiplier'][c],p['shift'][c],0)[0]) for c in range(2)]
    assert out==[50,-50]


@pytest.mark.parametrize('op',['MaxPool','AveragePool','GlobalAveragePool'])
def test_pooling_and_image(op):
    attrs={} if op=='GlobalAveragePool' else {'kernel_shape':[2,2],'strides':[2,2]}
    shape=[1,2,1,1] if op=='GlobalAveragePool' else [1,2,2,2]
    m=model([h.make_node(op,['x'],['y'],**attrs)],{},[1,2,4,4],{'y':shape})
    x=np.arange(-16,16,dtype=np.float32).reshape(1,2,4,4);p=compile_case(m,x)
    qx=p.tensors['x'].quantization.encode(x);iq=p.tensors['x'].quantization;oq=p.tensors['y'].quantization
    actual=simulate_image(build_image(p),{'x':qx},quantized=True)['y']
    for c in range(2):
      for y in range(shape[2]):
       for z in range(shape[3]):
        patch=qx[0,c] if op=='GlobalAveragePool' else qx[0,c,y*2:y*2+2,z*2:z*2+2]
        if op=='MaxPool':v=int(patch.max())-iq.zero_point;ratio=iq.scale/oq.scale
        else:v=sum(int(t)-iq.zero_point for t in patch.ravel());ratio=iq.scale/(oq.scale*patch.size)
        # Independently construct the coefficient with frexp and ties-away.
        import math
        mant,e=math.frexp(ratio);mult=math.floor(mant*2**31+.5)
        if mult==2**31:mult//=2;e+=1
        assert actual[0,c,y,z]==scalar_round(v*mult,31-e,oq.zero_point)


def test_image_overlap_version_and_parameter_rejection():
    m=model([h.make_node('Gemm',['x','w'],['y'],transB=1)],{'w':np.eye(2)},[1,2],{'y':[1,2]})
    p=compile_case(m,np.ones((1,2),np.float32));blob=build_image(p)
    length=struct.unpack('<Q',blob[8:16])[0]
    original=json.loads(blob[16:16+length])
    for change in ('overlap','version'):
        header=copy.deepcopy(original)
        if change=='version':header['numerical_version']=1
        else:header['segments'][1]['offset']=header['segments'][0]['offset']
        data=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
        # Preserve the original header byte count with JSON whitespace.
        assert len(data)<=length
        bad=blob[:16]+data+b' '*(length-len(data))+blob[16+length:]
        with pytest.raises(ValueError):load_image(bad)
    p.layers[0].parameters['shift'][0]=63
    with pytest.raises(ValueError):build_image(p)


def test_isa_fields_and_negative_dram_access():
    from isa_spec import encode
    from assembler import assemble_line
    import dram
    with pytest.raises(ValueError):encode('NOP',unexpected=1)
    with pytest.raises(ValueError):encode('LOAD_V',dest=1.5,addr=0,length=1)
    with pytest.raises(ValueError):assemble_line('NOP 1')
    with pytest.raises(ValueError):dram.write_to_dram(np.array([1],np.int8),-1)
    with pytest.raises(ValueError):dram.read_from_dram(-1,1)


def test_clip_and_constant_first_add():
    m=model([h.make_node('Add',['b','x'],['a']),h.make_node('Clip',['a','lo','hi'],['y'])],
            {'b':1.,'lo':0.,'hi':6.},[1,4],{'y':[1,4]})
    x=np.array([[-3,1,5,9]],np.float32);p=compile_case(m,x)
    out=simulate_image(build_image(p),{'x':x})['y'];q=p.tensors['y'].quantization
    np.testing.assert_allclose(q.decode(out),[[0,2,6,6]],atol=.15)


def test_calibration_overlap_guard():
    import importlib.util,io,hashlib
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/research/evaluate_static.py'
    spec=importlib.util.spec_from_file_location('evaluate_static',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=model([h.make_node('Gemm',['x','w'],['y'],transB=1)],{'w':np.eye(2)},[1,2],{'y':[1,2]})
    x=np.array([[1,0]],np.float32);p=compile_case(m,x)
    for ids,values,error in [(['directed-calibration-0'],x,'ID overlap'),(['different-name'],x,'content overlap')]:
        b=io.BytesIO();np.savez(b,sample_ids=ids,labels=[0],x=values[None]);data=b.getvalue()
        with pytest.raises(ValueError,match=error):module.evaluate(p,data,hashlib.sha256(data).hexdigest())


def test_independent_multilayer_reference():
    from integer_reference import evaluate
    from static_pipeline import execute_layer
    rng=np.random.default_rng(773)
    m=model([h.make_node('Conv',['x','w'],['c'],pads=[1,1,1,1]),
             h.make_node('Relu',['c'],['r']),
             h.make_node('AveragePool',['r'],['pool'],kernel_shape=[2,2],strides=[2,2]),
             h.make_node('Transpose',['pool'],['nhwc'],perm=[0,2,3,1]),
             h.make_node('Flatten',['nhwc'],['flat']),
             h.make_node('Gemm',['flat','fc','b'],['y'],transB=1)],
            {'w':rng.normal(size=(4,2,3,3)),'fc':rng.normal(size=(3,16)),'b':[.2,-1,3]},
            [1,2,4,4],{'c':[1,4,4,4],'y':[1,3]})
    x=rng.normal(size=(1,2,4,4)).astype(np.float32);p=compile_case(m,x)
    qx=p.tensors['x'].quantization.encode(x);expected=evaluate(p,{'x':qx});actual={'x':qx}
    for layer in p.layers:
        actual[layer.output]=execute_layer(p,layer,actual)
        np.testing.assert_array_equal(actual[layer.output],expected[layer.output])
    assert p.tensors['nhwc'].layout=='NHWC'


def test_classifier_boundary_is_explicit_and_label_preserving():
    from classifier_boundary import logits_model
    m=model([h.make_node('Gemm',['x','w'],['logits'],transB=1),h.make_node('Softmax',['logits'],['prob'],axis=1)],
            {'w':np.eye(2)},[1,2],{'prob':[1,2]})
    with pytest.raises(ValueError,match='Softmax'):canonicalize(m)
    logits,boundary=logits_model(m);x=np.array([[3,-2]],np.float32)
    assert np.argmax(ReferenceEvaluator(logits).run(None,{'x':x})[0])==np.argmax(ReferenceEvaluator(m).run(None,{'x':x})[0])
    assert not boundary['probability_output_supported']
