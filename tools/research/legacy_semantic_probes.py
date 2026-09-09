from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'compiler'))
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import numpy as np
import compile as compiler, dram, assembler, golden_model as gm
from accelerator_config import AcceleratorConfig as C
from onnx.reference import ReferenceEvaluator
P=Path(tempfile.mkdtemp(prefix='tinyml-semantic-probes-'))
off={'weights':C.DRAM_ADDR_WEIGHTS,'biases':C.DRAM_ADDR_BIASES,'conv_weights':C.DRAM_ADDR_CONV_WEIGHTS}
def model(name,nodes,initializers,ins,outs):
    mo=h.make_model(h.make_graph(nodes,name,ins,outs,initializers),opset_imports=[h.make_opsetid('',13)])
    onnx.checker.check_model(mo)
    path=P/(name+'.onnx'); onnx.save(mo,path)
    return mo,path
def do_compile(name,mo,path,inp=None):
    maps=dram.save_all_initializers_to_dram(str(path),off)
    asm=P/(name+'.asm'); getattr(compiler, 'generate_legacy_assembly', compiler.generate_assembly)(str(path),str(asm),*maps)
    content=asm.read_text(); print(name+' ASM:',content)
    if inp is not None:
        assembler.assemble_file(str(asm))
        dram.save_input_to_dram(inp,C.DRAM_ADDR_INPUTS)
        gm.memory=dram.get_dram();gm.buffers={};gm.output_buffer=0;gm.pending_conv_config={}
        try:
            for line in content.splitlines():
                code=assembler.assemble_line(line)
                if code: gm.i_decoder(int(code,16))
            actual=gm.memory[C.DRAM_ADDR_OUTPUTS:C.DRAM_ADDR_OUTPUTS+2]
            expected=ReferenceEvaluator(mo).run(None,{'x':inp})[0]
            print(name,'float=',expected,'accelerator=',actual,'argmax=',int(np.argmax(expected)),int(np.argmax(actual)))
        except Exception as e: print(name,type(e).__name__,str(e))
    return content
for factor in [1.,100.]:
    name='bias_'+str(int(factor))
    mo,path=model(name,[h.make_node('Gemm',['x','w','b'],['y'],transB=1)],
       [nh.from_array(np.array([[.1,0],[0,.1]],np.float32),'w'),nh.from_array(np.array([0,factor],np.float32),'b')],
       [h.make_tensor_value_info('x',T.FLOAT,[1,2])],[h.make_tensor_value_info('y',T.FLOAT,[1,2])])
    do_compile(name,mo,path,np.array([[1,0]],np.float32))
mo,path=model('large_relu',[h.make_node('Relu',['x'],['r']),h.make_node('MaxPool',['r'],['y'],kernel_shape=[2,2],strides=[2,2])],[],[h.make_tensor_value_info('x',T.FLOAT,[1,1,32,32])],[h.make_tensor_value_info('y',T.FLOAT,[1,1,16,16])])
do_compile('large_relu',mo,path)
mo,path=model('BN',[h.make_node('BatchNormalization',['x','scale','beta','mean','var'],['a']),h.make_node('Gemm',['a','w','b'],['y'],transB=1)],
[nh.from_array(np.array([-1,1],np.float32),'scale'),nh.from_array(np.zeros(2,np.float32),'beta'),nh.from_array(np.zeros(2,np.float32),'mean'),nh.from_array(np.ones(2,np.float32),'var'),nh.from_array(np.eye(2,dtype=np.float32),'w'),nh.from_array(np.zeros(2,np.float32),'b')],
[h.make_tensor_value_info('x',T.FLOAT,[1,2])],[h.make_tensor_value_info('y',T.FLOAT,[1,2])])
do_compile('BN',mo,path,np.array([[1,.5]],np.float32))
mo,path=model('add',[h.make_node('Gemm',['x','w','b'],['a'],transB=1),h.make_node('Add',['a','delta'],['y'])],
[nh.from_array(np.eye(2,dtype=np.float32),'w'),nh.from_array(np.zeros(2,np.float32),'b'),nh.from_array(np.ones(2,np.float32),'delta')],
[h.make_tensor_value_info('x',T.FLOAT,[1,2])],[h.make_tensor_value_info('y',T.FLOAT,[1,2])])
do_compile('add',mo,path)
mo,path=model('conv_no_bias',[h.make_node('Conv',['x','w'],['y'],kernel_shape=[1,1])],
[nh.from_array(np.ones((1,1,1,1),np.float32),'w')],[h.make_tensor_value_info('x',T.FLOAT,[1,1,2,2])],[h.make_tensor_value_info('y',T.FLOAT,[1,1,2,2])])
do_compile('conv_no_bias',mo,path,np.ones((1,1,2,2),np.float32))
