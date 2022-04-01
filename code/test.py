
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from code.cnn import resnet56

#import data
from data import MNIST
from data import CIFAR10
from data import Intel


#import datasets
from datasets import MnistDataset
from datasets import Cifar10Dataset
from datasets import IntelDataset


#import CNN
from cnn import CNN_MNIST
from cnn import CNN_CIFAR
from cnn import CNN_INTEL

#import resnets
from cnn import resnet20
from cnn import resnet32
from cnn import resnet56


#Set seed for repoducability
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


#Setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running CNN on {device}')

now = datetime.now()

#Setup Tensorboard
current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')

writer = SummaryWriter(f'runs/intel_{x}')


#Train Function that takes model,model name, epoch count, batch size and patience
#to prevent overfitting

def train_model(model,model_name,max_epochs,batch_size,train,test,patience=7):
    
    #Load Data in Dataloader
    train_loader = torch.utils.dataloader(dataset=train,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.dataloader(dataset=test,batch_size=batch_size,shuffle=False)

    #Setup train test step for Tensorboard
    # and best epoch for early stopping and log best model   
    step_train = 0
    step_test = 0
    best_epoch = 0

    #Set optimizer, loss function
    optimizer = torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    print(f'Current Model: {model_name}')

    #Lists that store accuracy and loss of test and train data
    mean_acc_train = []
    mean_loss_train = []

    mean_acc_test = []
    mean_loss_test = []

    for epoch in tqdm(range(max_epochs)):
                
        model.train()
        losses = []
        train_losses = 0.0
        valid_losses = 0.0

        train_acc = 0.0
        valid_acc = 0.0

    #Load n = batch_size number of images and labels
        for img, targets in train_loader:
            img, targets = img.to(device), targets.to(device)

            #zero gradients
            optimizer.zero_grad()

            #forward+backward
            scores = model(img)
            
            #Compare predicted label and actual label 
            loss = criterion(scores,targets)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            train_losses+= loss.item() * img.size(0)
            _, preds = scores.max(1)
            correct_tensor = preds.eq(targets.data.view_as(preds))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            train_acc += accuracy.item() * img.size(0)
        
            num_correct = (preds==targets).sum()
            running_train_acc = float(num_correct) /float(img.shape[0])

        writer.add_scalar("Training loss", loss, global_step=step_train)
        writer.add_scalar("Training Accuracy", running_train_acc, global_step=step_train)
        step_train +=1
        #Set model to eval
        model.eval()

        with torch.no_grad():
            for data,targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)

                #forward
                scores = model(data)

                #validation loss
                loss = criterion(scores,targets)
                valid_loss += loss.item() * data.size(0)

                #validation acc
                _, preds = scores.max(1)
                correct_tensor = preds.eq(targets.data.view_as(preds))
                accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor)
                )
                valid_acc += accuracy.item() * data.size(0)
            
                num_correct = (preds==targets).sum()
                running_test_acc = float(num_correct) /float(data.shape[0])

            writer.add_scalar("Validation loss", loss, global_step=step_test)
            writer.add_scalar("Validation Accuracy", running_test_acc, global_step=step_test)
            step_test +=1
        #Calculate average losses
        train_losses = train_losses /len(train_loader.dataset)
        valid_losses = valid_losses / len(test_loader.dataset)

        #Calculate average acc
        train_acc = train_acc /len(train_loader.dataset)
        valid_acc = valid_acc /len(test_loader.dataset)

        mean_acc_train.append(train_acc)
        mean_loss_train.append(train_losses)

        mean_acc_test.append(valid_acc)
        mean_loss_test.append(valid_losses)
        writer.add_scalar("Mean Loss Train", train_losses, global_step=epoch)
        writer.add_scalar("Mean Acc Train", train_acc, global_step=epoch)
        writer.add_scalar("Mean Loss Validation", valid_losses, global_step=epoch)
        writer.add_scalar("Mean Accuracy Test", valid_acc, global_step=epoch)

        print(f'Train Loss: {train_losses} Train Accuracy: {train_acc}')
        print(f'Valid Loss: {valid_losses} Valid Accuracy: {valid_acc}')

        if len(mean_acc_test) == 1 or valid_acc > mean_acc_test[best_epoch]:
            best_epoch = epoch
        elif best_epoch <= epoch - patience:
            print('Early Stopping due to patience limit')
            break

    return mean_acc_train,mean_loss_train,mean_acc_test,mean_loss_test
    

#Load datasets
mnist = MNIST()
cifar = CIFAR10()
intel = Intel()



#Load custom dataset
mnist_ds = MnistDataset
cifar_ds = Cifar10Dataset
intel_ds = IntelDataset


#assign CNN
cnn_mnist = CNN_MNIST
cnn_cifar = CNN_CIFAR
cnn_intel = CNN_INTEL

#assign resnet
res50 = resnet56
res32 = resnet32
res20 = resnet20

#assign special net
shuffle = ShuffleNet
xception = XceptionNet
simplenet = simplenetv1

#assign to list
datasets = [mnist_ds,cifar_ds,intel_ds]






        
        

