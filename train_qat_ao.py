import torch
from config import *
from transforms import *
from model import *
from qat import *
from data import *
import mlflow
from mlflow.tracking import MlflowClient
import time
import copy
from tqdm.auto import tqdm
import argparse
import sys
import onnx
from to_onnx import convert_onnx




class QuantizeTrainModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        
        # Model settings
        self.pretrained=pretrained
        self.num_classes = num_classes
        
        # Floating point -> Integer for input
        self.quant = torch.ao.quantization.QuantStub()
        self.model = build_model(num_classes=num_classes)

        # Integer to Floating point for output
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)

    def clone(self):
        clone = QuantizeTrainModel(self.pretrained, self.num_classes)
        clone.load_state_dict(self.state_dict())
        clone.to("cpu")
        return clone
    
    # def fused_module_inplace(self):
    #     """
    #     Fusing the model for resnet family only. Print out the model architecture
    #     to know how the logic works. Note that during the quantize fusion, 
    #     conv + bn + act or conv + bn should be bundled to together
    #     """
    #     self.train()

    #     for module_name, module in self.named_children():
    #         if module_name in ["0","8"]:
    #             torch.ao.quantization.fuse_modules_qat(module, ["0","1"], inplace=True)
    #         else:
    #             for basic_block_name, basic_block in module.named_children():
    #                 if "MBConv" == str(basic_block).split("(")[0]:
    #                     for sub_block_name, sub_block in basic_block.named_children():
    #                         for sub_sub_block_name, sub_sub_block in sub_block.named_children():
    #                             if "Conv2dNormActivation" == str(sub_sub_block).split("(")[0]:
    #                                 torch.ao.quantization.fuse_modules_qat(sub_sub_block, [["0","1"]], inplace=True)


        
        

def train(model, trainloader, optimizer, criterion,scheduler,device="cuda"):
    model.to(device)
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0.0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = torch.stack([T(img) for img in image])
        image = normalize_pretrained(image)
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        scheduler.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, testloader, criterion,device = "cuda"):

    model.eval()
    model.to(device)

    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            # image = torch.stack([T(img) for img in image])

            image = normalize_pretrained(image)
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)

            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return [epoch_loss, epoch_acc]

def get_quantized_model_from_weight(model, weight="best.pt"):
    new_model = copy.deepcopy(model).cpu().eval()
    quantized_model = torch.quantization.convert(new_model, inplace=False)
    return quantized_model

def arguments(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", required=True,help="path to root directory")
    ap.add_argument("-o", "--output", required=True,help="path to output image")
    ap.add_argument("-sz", "--img_sz", required=True,help="image size",type = int)
    ap.add_argument("-b", "--batch", required=True,help="batch size",type=int)
    ap.add_argument("-e", "--epochs", required=True,help="epochs",type=int)
    ap.add_argument("-mn", "--model-name", required=True,help="model name")
    ap.add_argument("-u", "--mlflow_url", required=True,help="mlflow url")
    ap.add_argument("-exp", "--exp_name", required=True,help="experiment name")
    ap.add_argument("-lr", "--learning_rate", required=True,help="learning rate",type=float)
    ap.add_argument("-d", "--device", required=True,help="GPU Device",type=int)
    ap.add_argument("-v", "--version", required=True,help="model version",type=int)
    args = ap.parse_args(args)
    print(args)
    return [args.data+"/train",args.data+"/val",args.output,args.img_sz,args.batch,args.epochs,args.model_name,args.mlflow_url,args.exp_name,args.learning_rate,args.device,args.version]

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def save_torchscript_model(model,model_name):
    torch.jit.save(torch.jit.script(model), model_name)

def load_q_model(checkpoint_path):
    example_inputs = (torch.rand(2, 3, 224, 224),)
    float_model = QuantizeTrainModel(num_classes=2)
    float_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(float_model, inplace=True)

    float_model.apply(torch.ao.quantization.enable_observer)
    float_model.apply(torch.ao.quantization.enable_fake_quant)
    float_model = torch.quantization.convert(float_model, inplace=False)

    float_model.load_state_dict(torch.load(checkpoint_path))
    return float_model

def save_q_model(model,model_name):
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':

    train_dir,val_dir,dest,IMAGE_SIZE,BATCH_SIZE,EPOCHS,model_name,MLFLOW_URL,EXP_NAME,lr,DEVICE,VERSION= arguments(sys.argv[1:])
    DEVICE = "cuda:"+str(DEVICE) if torch.cuda.is_available() else 'cpu'

    dataset_train, dataset_valid, dataset_classes = get_datasets(train_dir,val_dir,IMAGE_SIZE,True)

    train_loader, valid_loader = get_data_loaders(BATCH_SIZE,2,dataset_train, dataset_valid)

    num_classes = len(dataset_classes)
    client = MlflowClient()
    model =  QuantizeTrainModel(num_classes=num_classes)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    best_accuracy=0.0
    fp32_model=model

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(EXP_NAME)
    experiment = mlflow.get_experiment_by_name(EXP_NAME)

    model.to(DEVICE)

    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=EXP_NAME):

        for epoch in range(EPOCHS):

            epoch_time = time.time()
            train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion,scheduler,DEVICE)
            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,criterion,DEVICE)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)

            if best_accuracy < valid_epoch_acc:
                best_model=model
                best_accuracy=valid_epoch_acc
                mlflow.pytorch.log_model(model, "best")  
                convert_onnx(model,(IMAGE_SIZE,IMAGE_SIZE),DEVICE,"model")
                mlflow.onnx.log_model(onnx.load_model("model.onnx"),"model_onnx_best") 


            mlflow.log_metric("training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("training accuracy", f"{train_epoch_acc:3f}", step=epoch)

            mlflow.log_metric("validation loss", f"{valid_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("validation accuracy", f"{valid_epoch_acc:3f}", step=epoch)
            mlflow.pytorch.log_model(model, f"{epoch}")  
            convert_onnx(model,(IMAGE_SIZE,IMAGE_SIZE),DEVICE,"model")
            mlflow.onnx.log_model(onnx.load_model("model.onnx"),f"model_onnx_{epoch}") 

            print('estimated time of completion:',(time.time()-epoch_time)*(EPOCHS-epoch-1)/3600,' hrs ')

        import copy
        fp32_model = best_model.clone()
        qat_model = best_model.clone()
        lr = 0.000001


        optimizer = torch.optim.AdamW(qat_model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

        qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        qat_model.train()
        torch.ao.quantization.prepare_qat(qat_model, inplace=True)

        qat_model.apply(torch.ao.quantization.enable_observer)
        qat_model.apply(torch.ao.quantization.enable_fake_quant)

        for epoch in range(int(EPOCHS)):

            epoch_time = time.time()
            print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
            train_epoch_loss, train_epoch_acc = train(qat_model, train_loader, optimizer, criterion,scheduler,DEVICE)
            #print(exp_lr_scheduler.get_last_lr())
            valid_epoch_loss, valid_epoch_acc = validate(qat_model, valid_loader,criterion,DEVICE)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)


            if best_accuracy < valid_epoch_acc:
                
                best_accuracy=valid_epoch_acc
                quantized_model = get_quantized_model_from_weight(qat_model)
                mlflow.pytorch.log_model(quantized_model, "best")  

            try:
               client.create_registered_model(model_name)
            except Exception:
                pass  # Model already exists
    
            model_version = client.create_model_version(
                name=model_name,
                source=str(VERSION),
                run_id=mlflow.active_run().info.run_id,
            )
    
            print(f"Registered Model: {model_name}, Version: {model_version.version}")


            # print('estimated time of completion:',(time.time()-epoch_time)*(epochs-epoch-1)/3600,' hrs ')

            mlflow.log_metric("QAT training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("QAT training accuracy", f"{train_epoch_acc:3f}", step=epoch)

            mlflow.log_metric("QAT validation loss", f"{valid_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("QAT validation accuracy", f"{valid_epoch_acc:3f}", step=epoch)
            quantized_model = get_quantized_model_from_weight(model)
            mlflow.pytorch.log_model(quantized_model, f"{epoch}")  

        qat_model.to("cpu")
        
        quantized_model = get_quantized_model_from_weight(qat_model)

        quantized_model.eval()

        # # Save quantized model.
        save_q_model(quantized_model,model_name+"_q.pt")


        # # Load quantized model.
        quantized_jit_model = load_q_model(model_name+"_q.pt")

        _, fp32_eval_accuracy = validate(model, valid_loader , criterion,DEVICE)

        _, int8_eval_accuracy = validate(quantized_jit_model, valid_loader, criterion,"cpu")

        # # Skip this assertion since the values might deviate a lot.
        # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

        print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
        print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

        fp32_cpu_inference_latency = measure_inference_latency(model=fp32_model, device="cpu", input_size=(1,3,224,224), num_samples=100)
        int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device="cpu", input_size=(1,3,224,224), num_samples=100)
        int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device="cpu", input_size=(1,3,224,224), num_samples=100)
        fp32_gpu_inference_latency = measure_inference_latency(model=fp32_model, device=DEVICE, input_size=(1,3,224,224), num_samples=100)

        print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
        print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
        print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
        print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))




    print('TRAINING COMPLETE')




# kaggle
# https://www.kaggle.com/code/justin900429/quantization-awThe above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, arare-training
# ! TODO : ONNX support, fuse layers,

# https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d
# https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/compressing-models-during-training/quantization-aware-training-pytorch.html




# https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html#prepare-the-model-for-quantization-aware-training