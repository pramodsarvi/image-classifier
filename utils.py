import time
from config import *
from tqdm.auto import tqdm
from transforms import T
import torch
import os
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


# def train(model,train_loader,test_loader,criterion):
#     train_loss, valid_loss = [], []
#     train_acc, valid_acc = [], []
#     # Start the training.
    
#     best_accuracy=0.0
#     for epoch in range(EPOCHS):
#         epoch_time = time.time()
#         print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        
#         model = model.to(DEVICE)
#         train_epoch_loss, train_epoch_acc = util_train(model,train_loader,criterion)
#         #print(exp_lr_scheduler.get_last_lr())
#         valid_epoch_loss, valid_epoch_acc = validate(model,test_loader,criterion)
#         # lr_scheduler.step()
#         train_loss.append(train_epoch_loss)
#         valid_loss.append(valid_epoch_loss)
#         train_acc.append(train_epoch_acc)
#         valid_acc.append(valid_epoch_acc)
#         print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#         print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#         print('-'*50)
#         if valid_epoch_acc>best_accuracy:
#             best_accuracy=valid_epoch_acc
#             save_model(epoch,is_best=True)
#         save_model(epoch,is_best= False)
#         # save_plot(train_acc, valid_acc, train_loss, valid_loss, True)
#         print('estimated time of completion:',(time.time()-epoch_time)*(EPOCHS-epoch-1)/3600,' hrs ')
#     # Save the trained model weights.
#     save_model(EPOCHS,False)
#     # Save the loss and accuracy plots.
#     # save_plots(train_acc, valid_acc, train_loss, valid_loss, True)
#     print('TRAINING COMPLETE')

def util_train(model,trainloader,criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = torch.stack([T(img) for img in image])
        # image = normalize_pretrained(image)
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        # optimizer.zero_grad()
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
        # optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model,testloader,criterion):
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
            # image = normalize_pretrained(image)
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
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


def save_model(epoch,model,criterion,is_best=False):


    if is_best:
        if QAT:
            for module in model.modules():
                module._forward_hooks.clear()
            torch.jit.save(torch.jit.script(model), os.path.join(SAVE_PATH,"efficient_best.pt"))
            return
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': criterion,
                    }, os.path.join(SAVE_PATH,"efficient_best.pth"))
        # save_gradcam(f"{dest}/efficient_net_best.pth")
    if epoch%SAVE_EPOCH == 0:
        if QAT:
            for module in model.modules():
                module._forward_hooks.clear()
            torch.jit.save(torch.jit.script(model), os.path.join(SAVE_PATH,f"efficient_{epoch}.pt"))
            return
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': criterion,
                    }, os.path.join(SAVE_PATH,f"efficient_net_{epoch}.pth"))
        
    elif epoch==EPOCHS:
        if QAT:
            for module in model.modules():
                module._forward_hooks.clear()
            torch.jit.save(torch.jit.script(model), os.path.join(SAVE_PATH,f"efficient_last.pt"))
            return
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': criterion,
                    }, os.path.join(SAVE_PATH,f"efficient_last.pth"))