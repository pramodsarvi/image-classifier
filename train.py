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
from torch.utils.data import Subset
import mlflow
from torchsampler import ImbalancedDatasetSampler


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
        self.train_dataset, self.validation_dataset, self.classes =  self.prepare_dataset()
        self.train_loader, self.validation_loader =  self.prepare_dataset()

        self.qat = qat
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.fp16 = None
        self.qat()
    
    def prepare_dataset(self):
        dataset = datasets.ImageFolder(self.root,transform=(get_train_transform(IMAGE_DIMS, True))        )
        dataset_size = len(dataset)
        # Calculate the validation dataset size.
        valid_size = int(VALID_SPLIT*dataset_size)
        # Radomize the data indices.
        indices = torch.randperm(len(dataset)).tolist()
        # Training and validation sets.
        dataset_train = Subset(dataset, indices[:-valid_size])
        dataset_valid = Subset(dataset, indices[-valid_size:])
        
        return dataset_train, dataset_valid, dataset_train.classes

    def prepare_data_loaders(self):

        train_loader = DataLoader(
            self.train_dataset, 
            
            batch_size=BATCH_SIZE, 
            # sampler=DistributedSampler(dataset_train),
            shuffle=False, num_workers=NUM_WORKERS
        )
        valid_loader = DataLoader(
            self.validation_dataset,
            # sampler=DistributedSampler(dataset_valid),
            batch_size=BATCH_SIZE, 
            shuffle=False, num_workers=NUM_WORKERS
        )
        return train_loader, valid_loader 

    def train(self):
        self.model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            counter += 1
            image, labels = data
            image = torch.stack([T(img) for img in image])
            # image = normalize_pretrained(image)
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            self.optimizer.zero_grad()
            # Forward pass.
            outputs = self.model(image)

            # Calculate the loss.
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights.
            self.optimizer.step()
        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(self.train_loader.dataset))
        return epoch_loss, epoch_acc


    def prepare_qat(self):
        if self.qat:
            self.model = quantized_model(self.model).to(DEVICE)
            quantization_config = torch.quantization.get_default_qconfig("fbgemm")
            self.model.qconfig = quantization_config
            torch.quantization.prepare_qat(self.model, inplace=True)

    def train_fp16(self):
        scaler = GradScaler()

        self.model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            counter += 1
            image, labels = data
            image = torch.stack([T(img) for img in image])
            
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            self.optimizer.zero_grad()
            # Forward pass.

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(image)
                loss = self.criterion(outputs, labels)

            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            # Calculate the loss.
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()

        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(self.train_loader.dataset))
        return epoch_loss, epoch_acc

    def evaluate(self):
        self.model.eval()
        print('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader)):
                counter += 1
                
                image, labels = data
                # image = torch.stack([T(img) for img in image])
                # image = normalize_pretrained(image)
                image = image.to(DEVICE)
                labels = labels.to(DEVICE)
                # Forward pass.
                outputs = self.model(image)
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
            
        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(self.validation_loader.dataset))
        return epoch_loss, epoch_acc

    def benchmark(self):
        pass

    def save_models(self):
        pass

    def log_mlflow_stats(self):
        pass
    
    def to_onnx(self):
        pass




    def start_training(self,model,epochs,train_loader,valid_loader,optimizer,criterion):

        best_accuracy=-1000.0

        for epoch in range(self.epochs):

            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")

            train_epoch_loss, train_epoch_acc = amp_util_train(model, train_loader,criterion,optimizer)

            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,criterion)
            # exp_lr_scheduler.step()

            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            # time.sleep(5)
            if valid_epoch_acc>best_accuracy:
                # if the model is best then save the model to disk and later log it to mlflow
                best_model=model
                best_accuracy=valid_epoch_acc
                save_model(epoch, model, criterion,is_best=True)
            save_model(epoch, model, criterion, False)


            mlflow.log_metric("training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("training accuracy", f"{train_epoch_acc:3f}", step=epoch)
        # Save the trained model weights.
        print('TRAINING COMPLETE')
        # log best and last epoch accuracies and loss

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
