"""Test-only independent integer graph oracle.

No compiler arithmetic or execution helpers are imported. Uses centered inputs,
product-unit bias, Python integer division, and window/tensor contractions. This
is independent from the production raw-MAC/corrected-bias implementation.
"""
import numpy as np


def rounded(numerator,shift,zp):
    denominator=2**int(shift)
    def scalar(n):
        q,r=divmod(abs(int(n)),denominator)
        q+=int(2*r>=denominator)
        return min(127,max(-128,(-q if n<0 else q)+zp))
    return np.vectorize(scalar,otypes=[np.int8])(numerator)


def coefficients(ratio):
    import math
    f,e=math.frexp(ratio);m=math.floor(f*2**31+.5)
    if m==2**31:m//=2;e+=1
    return m,31-e


def evaluate(program,quantized_inputs):
    values={n:v.copy() for n,v in program.constants.items()}
    values.update({n:v.copy() for n,v in quantized_inputs.items()})
    for layer in program.layers:
        names=layer.inputs;x=values[names[0]];a=layer.attributes;p=layer.parameters
        iq=program.tensors[names[0]].quantization;oq=program.tensors[layer.output].quantization
        shape=tuple(program.tensors[layer.output].shape)
        if layer.op in ('Conv','Gemm'):
            centered=x.astype(np.int64)-iq.zero_point;w=p['weight'].astype(np.int64)
            if layer.op=='Gemm':
                acc=np.dot(centered,w.T)+p['bias']
                y=np.stack([rounded(acc[:,c]*int(p['multiplier'][c]),p['shift'][c],oq.zero_point) for c in range(len(w))],axis=1)
            else:
                pt,pl,pb,pr=a.get('pads',[0,0,0,0]);dh,dw=a.get('dilations',[1,1]);sh,sw=a.get('strides',[1,1])
                kh,kw=w.shape[2:];groups=a.get('group',1);icg=w.shape[1];ocg=len(w)//groups
                padded=np.pad(centered,((0,0),(0,0),(pt,pb),(pl,pr)))
                windows=np.lib.stride_tricks.sliding_window_view(padded,((kh-1)*dh+1,(kw-1)*dw+1),axis=(2,3))
                windows=windows[:,:,::sh,::sw,::dh,::dw]
                y=np.empty(shape,np.int8)
                for group in range(groups):
                    # [N,H,W,Cout], using a contraction unlike the production
                    # per-output-channel sequence of raw patch dot products.
                    acc=np.tensordot(windows[:,group*icg:(group+1)*icg],w[group*ocg:(group+1)*ocg],axes=([1,4,5],[1,2,3]))
                    for c in range(ocg):
                        channel=group*ocg+c
                        v=acc[:,:,:,c]+int(p['bias'][channel])
                        y[:,channel]=rounded(v*int(p['multiplier'][channel]),p['shift'][channel],oq.zero_point)
        elif layer.op in ('Identity','Reshape','Flatten'):y=x.reshape(shape).copy()
        elif layer.op=='Transpose':y=np.transpose(x,a.get('perm',tuple(reversed(range(x.ndim))))).copy()
        elif layer.op in ('Relu','Clip'):
            lower,upper=(iq.zero_point,127) if layer.op=='Relu' else [int(v) for v in p['clip_bounds']]
            v=np.minimum(upper,np.maximum(lower,x.astype(np.int64)))-iq.zero_point
            m,s=coefficients(iq.scale/oq.scale);y=rounded(v*m,s,oq.zero_point)
        elif layer.op=='Add':
            v=sum((values[n].astype(np.int64)-program.tensors[n].quantization.zero_point)*int(m) for n,m in zip(names,p['add_multiplier']))
            y=rounded(v,p['add_shift'][0],oq.zero_point)
        elif layer.op in ('MaxPool','AveragePool','GlobalAveragePool'):
            kh,kw=a.get('kernel_shape',x.shape[2:]);sh,sw=a.get('strides',[1,1]);pt,pl,_,_=a.get('pads',[0,0,0,0])
            y=np.empty(shape,np.int8)
            for iy in range(shape[2]):
             for ix in range(shape[3]):
                ys=[j for j in range(iy*sh-pt,iy*sh-pt+kh) if 0<=j<x.shape[2]]
                xs=[j for j in range(ix*sw-pl,ix*sw-pl+kw) if 0<=j<x.shape[3]]
                window=x[:,:,ys][:,:,:,xs].astype(np.int64)-iq.zero_point
                if layer.op=='MaxPool':v=np.max(window,axis=(2,3));divisor=1
                else:v=np.sum(window,axis=(2,3));divisor=kh*kw if a.get('count_include_pad',0) else len(ys)*len(xs)
                m,s=coefficients(iq.scale/(oq.scale*divisor));y[:,:,iy,ix]=rounded(v*m,s,oq.zero_point)
        else:raise ValueError(f'oracle does not implement {layer.op}')
        values[layer.output]=y
    return values
