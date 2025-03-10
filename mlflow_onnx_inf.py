import mlflow.onnx
import onnxruntime as ort
import cv2
from model import build_model
# Load ONNX model from MLflow
# onnx_model = mlflow.onnx.load_model("runs:/eb92ab7afbc84c35907a599054b13891/onnx_model",tracking_uri="http://13.127.65.67:5000/")


import mlflow.onnx
import onnxruntime as ort
import numpy as np
mlflow.set_tracking_uri("http://13.127.65.67:5000/")
# Load ONNX model from MLflow
model_uri = 'runs:/e694558ad8bb47efbd636141a23b016f/model_onnx_best'  # Change this to your MLflow model path
onnx_model = mlflow.onnx.load_model(model_uri)
import onnx 
# Save ONNX model as a standalone file (optional)
onnx.save(onnx_model, "mo7899808989890del.onnx")



# # Run inference
# session = ort.InferenceSession(onnx_model.SerializeToString())

# image = cv2.imread("/home/pramod/Pictures/Screenshot from 2023-12-13 20-53-30.png")
# #image = cv2.imread("image.jpg")

# # image = Image.open("/home/sr/Pictures/infrence_image.png")
# batch_size = 1

# # image = cv2.resize(image, (64,64))
# # image = np.random.randint(0,255,size=(size[0],size[1],3))
# print(image.shape)
# image = cv2.resize(image,[224,224])

# print(image.shape)

# # image  = image.astype(float)

# from torchvision import transforms
# tr = transforms.ToTensor()

# image_tensor = tr(image)

# print(image_tensor.shape)


# image_tensor = image_tensor.unsqueeze(0).to("cuda:0")
# # model = build_model(num_classes=2)
# # import torch
# # ckpt = torch.load("/home/pramod/Documents/model.pth",weights_only=True)
# # print(ckpt.keys)
# # # weights_only=True
# # model.load_state_dict(ckpt)
# # print(model(image))




# input_data = {"actual_input":image_tensor}
# outputs = session.run(None, input_data)

# print(outputs)

