from collections import defaultdict, deque
def topological_sort(graph):
    tensor_producer = {}
    for node in graph.node:
        # print(f"Processing node: {node.name} (OpType: {node.op_type})")
        # print(f"Overall node: {node}")
        # print(f"  Inputs: {node.input}")
        # print(f"  Outputs: {node.output}")
        for out in node.output:
            tensor_producer[out] = node
    # print("Tensor producer map:", tensor_producer)
    indegree = {id(node): 0 for node in graph.node}
    deps = defaultdict(list)

    for node in graph.node:
        for input_tensor in node.input:
            if input_tensor in tensor_producer:
                parent = tensor_producer[input_tensor]
                deps[id(parent)].append(node)
                indegree[id(node)] += 1

    queue = deque()
    for node in graph.node:
        if indegree[id(node)] == 0:
            queue.append(node)

    ordered = []
    processed_count = 0
    while queue:
        node = queue.popleft()
        ordered.append(node)
        processed_count += 1
        for child in deps[id(node)]:
            indegree[id(child)] -= 1
            if indegree[id(child)] == 0:
                queue.append(child)

    # Detect cycles in invalid graphs
    if processed_count != len(graph.node):
        print("⚠️ Warning: Graph contains cycles! Topological sort incomplete.")
    return ordered