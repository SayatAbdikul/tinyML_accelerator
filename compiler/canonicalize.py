"""Fail-closed ONNX preprocessing, before memory allocation or output writes."""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def canonicalize(model):
    model = copy.deepcopy(model)
    onnx.checker.check_model(model)
    if model.functions:
        raise ValueError("custom domains/functions are unsupported")
    g = model.graph
    constants = {t.name: numpy_helper.to_array(t).copy() for t in g.initializer}
    nodes = []
    # Evaluate only fully constant subgraphs; never discard a runtime op.
    from onnx.reference import ReferenceEvaluator
    for node in g.node:
        if node.domain not in ('', 'ai.onnx'):
            raise ValueError(f"custom domain: {node.domain}")
        if node.op_type == 'Constant' or (node.input and all(x in constants for x in node.input)):
            try:
                if node.op_type == 'BatchNormalization':
                    raise ValueError('constant BN requires explicit inference folding')
                subgraph = helper.make_graph([node], 'constant_fold', [],
                    [onnx.ValueInfoProto(name=n) for n in node.output],
                    [numpy_helper.from_array(constants[n],n) for n in dict.fromkeys(node.input)])
                submodel = helper.make_model(subgraph,opset_imports=model.opset_import)
                vals = ReferenceEvaluator(submodel).run(None,{})
            except Exception as exc:
                raise ValueError(f"constant evaluation failed: {node.op_type}") from exc
            constants.update(zip(node.output, vals))
        else:
            nodes.append(node)
    consumers = {name: [n for n in nodes if name in n.input]
                 for n in nodes for name in n.output}
    outputs = {o.name for o in g.output}
    folded = []
    for node in nodes:
        a = attrs(node)
        if node.op_type in ('Add', 'Mul') and sum(x in constants for x in node.input) == 1:
            data_name = next(x for x in node.input if x not in constants)
            delta = constants[next(x for x in node.input if x in constants)]
            producer = next((n for n in folded if n.output[0] == data_name), None)
            if (producer is not None and producer.op_type in ('Gemm','Conv')
                    and len(consumers[data_name]) == 1 and data_name not in outputs):
                b = constants[producer.input[2]]
                expected = (1,len(b),1,1) if producer.op_type == 'Conv' else (1,len(b))
                try:
                    delta = np.broadcast_to(delta,expected).reshape(-1)
                except ValueError:
                    pass  # General Add remains explicit; never drop a broadcast.
                else:
                    bn = node.output[0] + '/folded_add_bias'
                    if bn in constants:
                        raise ValueError('generated Add bias name collision')
                    if node.op_type == 'Mul':
                        # Exporters may express inference BN as channelwise
                        # Mul/Add. Fold the affine scale only when the producer
                        # is exclusive and the broadcast is channel-constant.
                        w = constants[producer.input[1]]
                        wn = node.output[0] + '/folded_mul_weight'
                        if wn in constants:
                            raise ValueError('generated Mul weight name collision')
                        constants[wn] = (w * delta.reshape((-1,) + (1,)*(w.ndim-1))).astype(w.dtype)
                        constants[bn] = (b*delta).astype(b.dtype)
                        producer.input[1] = wn
                    else:
                        constants[bn] = (b+delta).astype(b.dtype)
                    producer.input[2] = bn
                    producer.output[0] = node.output[0]
                    continue
        if node.op_type == 'BatchNormalization':
            if set(a) - {'epsilon','momentum','training_mode'}:
                raise ValueError('unsupported BN attributes')
            if a.get('training_mode', 0) or len(node.output) != 1:
                raise ValueError("training/multi-output BN unsupported")
            producer = next((n for n in folded if n.output[0] == node.input[0]), None)
            if (producer is None or producer.op_type != 'Conv'
                    or len(consumers[node.input[0]]) != 1 or node.input[0] in outputs):
                raise ValueError("BN must fold into an exclusively consumed Conv")
            if not all(x in constants for x in node.input[1:]):
                raise ValueError("BN parameters must be constant")
            w = constants[producer.input[1]]
            b = constants[producer.input[2]] if len(producer.input) == 3 else np.zeros(w.shape[0])
            gamma, beta, mean, var = [constants[x] for x in node.input[1:]]
            if any(v.shape != (w.shape[0],) for v in (gamma,beta,mean,var)):
                raise ValueError('BN parameters must match output channels')
            eps = a.get('epsilon', 1e-5)
            if eps < 0 or np.any(var + eps <= 0):
                raise ValueError("invalid BN variance/epsilon")
            factor = gamma / np.sqrt(var + eps)
            wn, bn = node.output[0] + '/folded_w', node.output[0] + '/folded_b'
            if wn in constants or bn in constants:
                raise ValueError("generated BN parameter name collision")
            constants[wn] = (w * factor.reshape(-1, 1, 1, 1)).astype(w.dtype)
            constants[bn] = ((b - mean) * factor + beta).astype(w.dtype)
            del producer.input[1:]
            producer.input.extend([wn, bn])
            producer.output[0] = node.output[0]
            continue
        if node.op_type in ('Gemm', 'MatMul', 'Conv'):
            if len(node.input) < 2 or node.input[1] not in constants:
                raise ValueError("weights must be constant")
            w = constants[node.input[1]]
            if node.op_type in ('Gemm', 'MatMul'):
                if a.get('transB', 0) not in (0,1) or a.get('transA',0) not in (0,1):
                    raise ValueError('Gemm transpose attributes must be 0 or 1')
                if w.ndim != 2 or a.get('transA', 0):
                    raise ValueError("only untransposed rank-2 activation Gemm/MatMul")
                w = w if a.get('transB', 0) else w.T
                w = w * a.get('alpha', 1.)
                beta = a.get('beta', 1.)
                if set(a) - {'alpha', 'beta', 'transA', 'transB'}:
                    raise ValueError("unsupported Gemm attributes")
                node.op_type = 'Gemm'
                del node.attribute[:]
                node.attribute.append(helper.make_attribute('transB', 1))
            else:
                beta = 1.
                if w.ndim != 4:
                    raise ValueError("only NCHW 2D Conv")
            if len(node.input) == 3:
                if node.input[2] not in constants:
                    raise ValueError("bias must be constant")
                b = constants[node.input[2]] * beta
                if node.op_type == 'Conv' and b.shape != (w.shape[0],):
                    raise ValueError("Conv bias must have one entry per channel")
                try:
                    b = np.broadcast_to(b, (1, w.shape[0])).reshape(-1)
                except ValueError as exc:
                    raise ValueError("unsupported Gemm bias broadcast") from exc
            else:
                b = np.zeros(w.shape[0], dtype=w.dtype)
            wn, bn = node.output[0] + '/canonical_w', node.output[0] + '/canonical_b'
            if wn in constants or bn in constants:
                raise ValueError("generated parameter name collision")
            constants[wn], constants[bn] = w.astype(np.float32), b.astype(np.float32)
            del node.input[1:]
            node.input.extend([wn, bn])
        folded.append(node)
    del g.node[:]
    g.node.extend(folded)
    used = {x for n in folded for x in n.input} | outputs
    del g.initializer[:]
    g.initializer.extend(numpy_helper.from_array(v, k) for k, v in constants.items() if k in used)
    # Old inferred intermediates may describe tensors removed by folding.
    del g.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    validate(model)
    onnx.checker.check_model(model)
    return model


def shapes(model):
    result = {}
    for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = v.type.tensor_type.shape.dim
        if not dims or any(not d.HasField('dim_value') or d.dim_value <= 0 for d in dims):
            raise ValueError(f"static positive shapes required: {v.name}")
        result[v.name] = tuple(d.dim_value for d in dims)
    for t in model.graph.initializer:
        result[t.name] = tuple(t.dims)
    return result


def validate(model):
    sm = shapes(model)
    allowed = {
        'Conv': {'kernel_shape','strides','pads','dilations','group','auto_pad'},
        'Gemm': {'transB'}, 'Relu': set(), 'Add': set(), 'Clip': set(),
        'Flatten': {'axis'}, 'Reshape': {'allowzero'}, 'Identity': set(),
        'Transpose': {'perm'},
        'MaxPool': {'kernel_shape','strides','pads','ceil_mode','dilations','auto_pad','storage_order'},
        'AveragePool': {'kernel_shape','strides','pads','ceil_mode','count_include_pad','auto_pad'},
        'GlobalAveragePool': set(),
    }
    const = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    runtime_inputs = [i for i in model.graph.input if i.name not in const]
    if not runtime_inputs or any(i.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for i in runtime_inputs):
        raise ValueError('runtime inputs must be FLOAT32 tensors')
    if any(o.name in const for o in model.graph.output):
        raise ValueError('constant-only graph outputs are not supported by the v2 target')
    for n in model.graph.node:
        a = attrs(n)
        if n.op_type not in allowed or set(a) - allowed[n.op_type] or len(n.output) != 1:
            raise ValueError(f"unsupported operator/attributes/outputs: {n.op_type} {a}")
        if any(x not in sm for x in list(n.input) + list(n.output)):
            raise ValueError(f"missing inferred shape: {n.name or n.op_type}")
        if n.op_type in ('Conv','MaxPool','AveragePool','GlobalAveragePool'):
            if len(sm[n.input[0]]) != 4 or sm[n.input[0]][0] != 1:
                raise ValueError("spatial operators require batch-1 NCHW")
            if a.get('auto_pad', b'NOTSET') not in (b'NOTSET', b''):
                raise ValueError("resolve auto_pad to explicit pads before import")
            if a.get('ceil_mode', 0) or a.get('storage_order', 0):
                raise ValueError("ceil-mode/index pooling unsupported")
            for key in ('strides','dilations','kernel_shape'):
                if key in a and (len(a[key]) != 2 or any(x <= 0 for x in a[key])):
                    raise ValueError(f"invalid {key}")
            if 'pads' in a and (len(a['pads']) != 4 or any(x < 0 for x in a['pads'])):
                raise ValueError("invalid explicit padding")
            if n.op_type == 'Conv':
                w = const[n.input[1]]
                group = a.get('group', 1)
                if group <= 0 or w.shape[0] % group or w.shape[1] * group != sm[n.input[0]][1]:
                    raise ValueError("invalid Conv grouping")
                if tuple(a.get('kernel_shape', w.shape[2:])) != w.shape[2:]:
                    raise ValueError("kernel_shape disagrees with weight")
            elif a.get('dilations', [1,1]) != [1,1]:
                raise ValueError("dilated pooling unsupported")
        if n.op_type == 'Gemm' and (len(sm[n.input[0]]) != 2 or sm[n.input[0]][0] != 1):
            raise ValueError("Gemm requires batch-1 rank-2 input")
        if n.op_type == 'Reshape' and n.input[1] not in const:
            raise ValueError("dynamic reshape unsupported")
        if n.op_type == 'Clip':
            if len(n.input) != 3 or any(x not in const or const[x].size != 1 for x in n.input[1:]):
                raise ValueError('Clip requires constant scalar minimum and maximum')
            lo,hi = (float(const[x].item()) for x in n.input[1:])
            if not np.isfinite([lo,hi]).all() or lo > hi:
                raise ValueError('invalid Clip range')
        if n.op_type == 'Add' and any(x in const for x in n.input):
            # Constants have a calibrated range too, but only floating data is valid.
            if any(const[x].dtype.kind != 'f' for x in n.input if x in const):
                raise ValueError("Add data constants must be floating point")
