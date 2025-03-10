
from distutils.command.build import build
import torchvision.models as models
import torch.nn as nn
import torch


import torchvision; torchvision.__version__
import torchvision.models as models



import torch.nn as nn
# from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as transforms
import torch.onnx
import onnxruntime
import onnx

import cv2
def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = '4' 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    # print("Inputs files are the ", inputs)
   
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]

        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() 


import numpy as np
def convert_onnx(model,size: tuple,device,onnx_name):

    image = cv2.imread("/home/pramod/Pictures/Screenshot from 2023-12-13 20-53-30.png")
    #image = cv2.imread("image.jpg")

    # image = Image.open("/home/sr/Pictures/infrence_image.png")
    batch_size = 1

    # image = cv2.resize(image, (64,64))
    # image = np.random.randint(0,255,size=(size[0],size[1],3))
    print(image.shape)
    image = cv2.resize(image,size)

    print(image.shape)

    # image  = image.astype(float)

    from torchvision import transforms
    tr = transforms.ToTensor()

    image_tensor = tr(image)

    print(image_tensor.shape)


    image_tensor = image_tensor.unsqueeze(0).to(device)

    print(image_tensor.shape)

    input_names = ["actual_input"]
    output_names = ["output"]


    model = model.to(device)


    torch.onnx.export(model,
        image_tensor,
        f"{onnx_name}.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        do_constant_folding = True,
        dynamic_axes={'input' : {0 : 'batch_size'},   
                                    'output' : {0 : 'batch_size'}})



    apply(change_input_dim, f"{onnx_name}.onnx",f"{onnx_name}.onnx")

    with torch.no_grad():
        ort = onnxruntime.InferenceSession(f"{onnx_name}.onnx")
        ort_inp = {ort.get_inputs()[0].name:to_numpy(image_tensor)}
        ort_out = ort.run(None, ort_inp)
        # class_prob = torch.nn.Softmax(dim = 1)
        # ops = class_prob(ort_out[0])

        # print(ort_out)