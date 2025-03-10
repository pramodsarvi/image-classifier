import torch
from config import *
from transforms import *
from model import *
from qat import *
from data import *
import mlflow
import time
import copy
from tqdm.auto import tqdm
import argparse
import sys
from sklearn.metrics import f1_score, accuracy_score


"""Kaggle code"""

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    total_predict = []
    total_ground_truth = []
    for data, label in val_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        prediction = output.argmax(dim=-1)

        total_predict.extend(prediction.cpu().tolist())
        total_ground_truth.extend(label.cpu().tolist())

    return accuracy_score(total_ground_truth, total_predict), \
           f1_score(total_ground_truth, total_predict, average="macro")


def quantize_train(model, train_loader, val_loader, criterion, optimizer, lr_sceduler):
    best_f1 = 0
    
    for epoch in range(EPOCHS):

        train_running_loss = 0.0
        counter = 0
        train_running_correct = 0.0
        train_progress_bar = tqdm(
            train_loader, desc=f"Epochs: {epoch + 1}/{EPOCHS}")
        
        model.train()
        for data, label in train_progress_bar:
            counter+=1
            data = data.to(args.device)
            label = label.to(args.device)

            # Send data into the model and compute the loss
            output = model(data)
            loss = criterion(output, label)
            
            # Compute Loss and Accuracy
            train_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            train_running_correct += (preds == label).sum().item()



            # Update the model with back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Obtain the quantization model for evaluation
        quantized_model = torch.ao.quantization.convert(
            copy.deepcopy(model).cpu().eval(), inplace=False)

        # Compute the accuracy ans save the best model
        eval_acc, eval_f1 = evaluate(
            quantized_model, val_loader, "cpu")
        
        train_running_loss = train_running_loss/ counter

        print(f"Validation accuracy: {eval_acc:.8f} f1-score: {eval_f1:.8f}")
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), "best.pt")


def get_quantized_model_from_weight(model, weight="best.pt"):
    new_model = copy.deepcopy(model).cpu().eval()
    new_model.load_state_dict(torch.load(weight, map_location="cpu"))
    quantized_model = torch.quantization.convert(new_model, inplace=False)
    return quantized_model


"""Kaggle Code End"""

def train(model, trainloader, optimizer, criterion,scheduler):
    model.train()
    print('Training')
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


def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
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
            # Forward pass.
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
    return epoch_loss, epoch_acc

def main(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", required=True,help="path to root directory")
    ap.add_argument("-o", "--output", required=True,help="path to output image")
    ap.add_argument("-sz", "--img_sz", required=True,help="image size",type = int)
    ap.add_argument("-b", "--batch", required=True,help="batch size",type=int)
    ap.add_argument("-e", "--epochs", required=True,help="epochs",type=int)
    ap.add_argument("-mn", "--model_name", required=True,help="model name")
    ap.add_argument("-u", "--mlflow_url", required=True,help="mlflow url")
    ap.add_argument("-exp", "--exp_name", required=True,help="experiment name")
    ap.add_argument("-lr", "--learning_rate", required=True,help="learning rate",type=float)
    args = ap.parse_args(args)
    return [args.data+"/train",args.data+"/val",args.output,args.img_sz,args.batch,args.epochs,args.model_name,args.mlflow_url,args.exp_name,args.learning_rate]


if __name__ == '__main__':

    train_dir,val_dir,dest,IMAGE_SIZE,BATCH_SIZE,EPOCHS,model_name,MLFLOW_URL,EXP_NAME,lr= main(sys.argv[1:])

    dataset_train, dataset_valid, dataset_classes = get_datasets(train_dir,val_dir,IMAGE_SIZE,True)

    train_loader, valid_loader = get_data_loaders(BATCH_SIZE,2,dataset_train, dataset_valid)

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = build_model(pretrained=True,fine_tune=True,num_classes=len(dataset_classes)).to(device)
    
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
    best_model=model

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(EXP_NAME)
    experiment = mlflow.get_experiment_by_name(EXP_NAME)



    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=EXP_NAME):

        for epoch in range(EPOCHS):
            epoch_time = time.time()

            quantize_train(model,train_loader,valid_loader,criterion,optimizer, scheduler)



            mlflow.log_metric("training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("training accuracy", f"{train_epoch_acc:3f}", step=epoch)

            mlflow.log_metric("validation loss", f"{valid_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("validation accuracy", f"{valid_epoch_acc:3f}", step=epoch)
            mlflow.pytorch.log_model(model, f"{epoch}")  


           
        model.to("cpu")

        # Make a copy of the model for layer fusion
        fused_model = copy.deepcopy(model)
        model.train()
        # The model has to be switched to training mode before any layer fusion.
        # Otherwise the quantization aware training will not work correctly.
        fused_model.eval()
        # Fuse the model in place rather manually.
        """
        Resnet layer fusion
        """
        # fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        # for module_name, module in fused_model.named_children():
        #     if "layer" in module_name:
        #         for basic_block_name, basic_block in module.named_children():
        #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        #             for sub_block_name, sub_block in basic_block.named_children():
        #                 if sub_block_name == "downsample":
        #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

        """
        Fusing layers for EfficientNet B0 model
        """
        for module_name, module in fused_model.features.named_children():
            if module_name in ["0","8"]:
                torch.quantization.fuse_modules(module, ["0","1"], inplace=True)
            else:
                for basic_block_name, basic_block in module.named_children():
                    if "MBConv" == str(basic_block).split("(")[0]:
                        for sub_block_name, sub_block in basic_block.named_children():
                            for sub_sub_block_name, sub_sub_block in sub_block.named_children():
                                if "Conv2dNormActivation" == str(sub_sub_block).split("(")[0]:
                                    torch.quantization.fuse_modules(sub_sub_block, [["0","1"]], inplace=True)


        fused_model.to("cpu")
        model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=model, model_2=fused_model, device="cpu", rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"


        quantized_model = Quantized_model(fused_model)

        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

        quantized_model.qconfig = quantization_config

        # Print quantization configurations
        # print(quantized_model.qconfig)

        # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
        torch.quantization.prepare_qat(quantized_model, inplace=True)

        # # Use training data for calibration.
        print("Training QAT Model...")
        model.to("cpu")
        quantized_model.to("cuda")
        quantized_model.train()

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        
        best_accuracy=0.0

        for epoch in range(EPOCHS):

            epoch_time = time.time()
            # print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(quantized_model, train_loader, optimizer, criterion,scheduler)
            #print(exp_lr_scheduler.get_last_lr())
            valid_epoch_loss, valid_epoch_acc = validate(quantized_model, valid_loader,criterion)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            # print('-'*50)

            if best_accuracy < valid_epoch_acc:
                
                best_accuracy=valid_epoch_acc
                mlflow.pytorch.log_model(quantized_model, "best")  

            # print('estimated time of completion:',(time.time()-epoch_time)*(epochs-epoch-1)/3600,' hrs ')

            mlflow.log_metric("QAT training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("QAT training accuracy", f"{train_epoch_acc:3f}", step=epoch)

            mlflow.log_metric("QAT validation loss", f"{valid_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("QAT validation accuracy", f"{valid_epoch_acc:3f}", step=epoch)
            mlflow.pytorch.log_model(quantized_model, f"{epoch}")  

        quantized_model.to("cpu")

        # Using high-level static quantization wrapper
        # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
        # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

        quantized_model = torch.quantization.convert(quantized_model, inplace=True)

        quantized_model.eval()


    print('TRAINING COMPLETE')




# kaggle
# https://www.kaggle.com/code/justin900429/quantization-aware-training