"""Explicit classifier boundary: integer logits on the accelerator, host argmax.

Softmax is strictly monotone in logits, so labels do not require computing it.
This transformation does not promise quantized probability outputs. It is opt-in
and returns a boundary record for benchmark manifests and host deployment.
"""
import copy
import onnx
from onnx import helper,shape_inference


def logits_model(model):
    model=copy.deepcopy(model)
    if len(model.graph.output)!=1:
        raise ValueError('classifier boundary needs exactly one graph output')
    output=model.graph.output[0]
    terminal=next((n for n in model.graph.node if output.name in n.output),None)
    if terminal is None or terminal.op_type!='Softmax' or len(terminal.output)!=1:
        raise ValueError('classifier boundary requires a terminal Softmax')
    shape=output.type.tensor_type.shape.dim
    attributes={a.name:helper.get_attribute_value(a) for a in terminal.attribute}
    if len(shape)!=2 or attributes.get('axis',-1) not in (-1,1):
        raise ValueError('only rank-2 class-axis Softmax may be removed')
    if any(terminal.output[0] in n.input for n in model.graph.node):
        raise ValueError('Softmax output has internal consumers')
    old_name=output.name;output.name=terminal.input[0]
    model.graph.node.remove(terminal)
    model=shape_inference.infer_shapes(model)
    return model,{'accelerator_output':output.name,'source_output':old_name,
                  'host_operation':'argmax over logits; first index on ties',
                  'probability_output_supported':False,'removed_operator':'terminal Softmax'}
