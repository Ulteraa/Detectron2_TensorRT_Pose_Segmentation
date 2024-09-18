import onnx

# Load the ONNX model
model = onnx.load('/home/fariborz/detectron2/tools/deploy/output/model.onnx')


# for input in model.graph.input:
#     print(input)

# List all model outputs
# for output in model.graph.output:
#     print(output)
#
# # List all the nodes in the graph
for node in model.graph.node:
    # if 'keypoint' in node.name:
        print(node)

# Print a human-readable representation of the model
# print(onnx.helper.printable_graph(model.graph))