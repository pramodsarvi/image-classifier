
from qat import quantized_model
from datasets import *
from build_model import *

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from tqdm import tqdm
# 
# from pytorch we import some function whose implementation can be found in documentation
from train import evaluate, train_one_epoch, load_data
"""
Doesn't do anything
"""
# def calibrate_model(model, loader, device=torch.device("cpu:0")):

#     model.to(device)
#     model.eval()

#     for inputs, labels in loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         _ = model(inputs)


"""
CALIBERATION
"""
def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
#             print(F"{name:40}: {module}")
    model.cuda()



def main():

    dataset_train, dataset_valid, dataset_classes = get_datasets(True)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    
    # Learning_parameters. 
    lr = 0.001
    epochs = EPOCHS
    DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Computation DEVICE: {DEVICE}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(pretrained=True,fine_tune=True,num_classes=len(dataset_classes)).to(DEVICE)


    # CHINESE GUY's BLOG
    quantized_model = quantized_model(model_fp32=model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config
    
    # Print quantization configurations
    print(quantized_model.qconfig)

    torch.quantization.prepare(quantized_model, inplace=True)

    # Use training data for calibration.
    calibrate_model(model=quantized_model, loader=train_loader, device=cpu_device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

    quantized_model.eval()


    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)
    torch.jit.load(model_filepath, map_location=device)


    # NVIDIA's example
    # in this you do not need quantised model class as the model is already trained in full precission


    # It is a bit slow since we collect histograms on CPU
    """
    Set default QuantDescriptor to use histogram based calibration for activation
    """
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    from pytorch_quantization import quant_modules
    quant_modules.initialize()
    model = torchvision.models.resnet50(pretrained=True, progress=False)
    
    
    model.cuda()
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)


    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
        
    # Save the model
    torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")




    with torch.no_grad():
        compute_amax(model, method="percentile", percentile=99.9)
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)


    
    with torch.no_grad():
        for method in ["mse", "entropy"]:
            print(F"{method} calibration")
            compute_amax(model, method=method)
            evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)


    # https://github.com/lix19937/pytorch-quantization/tree/main/examples

    sudo docker create -it --restart=always --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v ~/ds6_3_volume:/app/ -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v /tmp/.X11-unix/:/tmp/.X11-unix --name nvcr.io/nvidia/deepstream  deepstream_6.3