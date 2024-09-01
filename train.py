import time
from torch.optim import lr_scheduler
import torch
from config import *
from transforms import *
from datasets import *
from build_model import *

from save import *
from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = webface_r50  
from datasets import *
from qat import quantized_model

import mlflow

# from torch.utils.tensorboard import SummaryWriter

# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

class trainer():

    def __init__(self,model,epochs,gpu,root,save_path,batch_size,train_test_split,input_dims,lr,qat):
        self.model = model
        self.epochs = epochs
        self.gpu = gpu
        self.root = root
        self.save_path = save_path
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.input_dims = input_dims
        self.lr = lr
        self.qat = qat
    
    def train(self):
        pass

    def train_qat(self):
        pass

    def trainfp16(self):
        pass
    def evaluate(self):
        pass

    def benchmark(self):
        pass

    def save_models(self):
        pass

    def log_mlflow_stats(self):
        pass



def start_training(model,epochs,train_loader,valid_loader,optimizer,criterion):
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    
    best_accuracy=0.0
    best_model=model
    for epoch in range(epochs):
        epoch_time = time.time()
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = amp_util_train(model, train_loader,criterion,optimizer)
        #print(exp_lr_scheduler.get_last_lr())
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,criterion)
        # exp_lr_scheduler.step()
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        # time.sleep(5)
        if valid_epoch_acc>best_accuracy:
            best_model=model
            best_accuracy=valid_epoch_acc
            save_model(epoch, model, criterion,is_best=True)
        save_model(epoch, model, criterion, False)
        save_plots(train_acc, valid_acc, train_loss, valid_loss, True)
        print('estimated time of completion:',(time.time()-epoch_time)*(epochs-epoch-1)/3600,' hrs ')


        mlflow.log_metric("training loss", f"{train_epoch_loss:3f}", step=epoch)
        mlflow.log_metric("training accuracy", f"{train_epoch_acc:3f}", step=epoch)
    # Save the trained model weights.
    save_model(epochs, model, criterion, True)
    # Save the loss and accuracy plots.
    # torch.save(model,'/home/shalom/classifier/uniform_classifier/results/uniform_clsfr_efficientnet_pytorch_may6.pt')
    save_plots(train_acc, valid_acc, train_loss, valid_loss, True)
    print('TRAINING COMPLETE')

def main():
    
    # Load the training and validation datasets.
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
    
    if QAT:
        model = quantized_model(model).to(DEVICE)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        model.qconfig = quantization_config
        torch.quantization.prepare_qat(model, inplace=True)

    
    
    # Total parameters and trainable parameters.
    total_params = sum(p.squeeze().numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.squeeze().numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    # Loss function.
    criterion = torch.nn.CrossEntropyLoss()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=70,gamma=0.01)
    # Lists to keep track of losses and accuracies.
    


    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("test2")

    model_name = "model_name_v3"
    # with mlflow.start_run(run_name=model_name):

    # mlflow.set_experiment("experiment name")
    experiment = mlflow.get_experiment_by_name("test")

    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=model_name):

        start_training(model=model,epochs=epochs,train_loader=train_loader,valid_loader=valid_loader,optimizer=optimizer,criterion=criterion)
        mlflow.log_param("model", model_name)
        mlflow.log_metric('accuracy', 90)
        mlflow.log_metric('recall_class_1', 90)
        mlflow.log_metric('recall_class_0', 90)
        mlflow.log_metric('f1_score_macro', 90)        
        
        # if "XGB" in model_name:
        #     mlflow.xgboost.log_model(model, "model")
        # else:
        # mlflow.log_figure(fig1, "time_series_demand.png")
        mlflow.pytorch.log_model(model, "model")  

        



if __name__ == "__main__":
    main()
