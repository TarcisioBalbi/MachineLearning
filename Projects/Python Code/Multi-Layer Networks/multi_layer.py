#https://www.youtube.com/watch?v=9suSsTVhYuw
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda



def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

class MnistModel(nn.Module):
    #Feedfoward with 1 hidden layer
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,out_size)

    def forward(self,xb):
        xb = xb.view(xb.size(0),-1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    def training_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss':loss,'val_acc':acc}
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return{'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item()}
    def epoch_end(self,epoch,result):
        print("Epoch[{}],val_loss:{:.4f},val_acc:{:.4f}".format(epoch,result['val_loss'],result['val_acc']))


def evaluate(model,val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_loader,val_loader,opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model,val_loader)
        model.epoch_end(epoch,result)
        history.append(result)
    return history
        
def predict_image(img,model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _,preds = torch.max(yb,dim=1)
    return preds[0].item()

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)

    
device = get_default_device()
print(device)
dataset = MNIST(root='data/',download = True, transform= transforms.ToTensor())

val_size = 10000
train_size = len(dataset) - val_size
  
train_ds,val_ds = random_split(dataset,[train_size, val_size])

batch_size = 128

train_loader = DataLoader(train_ds,batch_size,shuffle=True,num_workers=0,pin_memory=True)
train_loader = DeviceDataLoader(train_loader,device)
val_loader = DataLoader(val_ds, batch_size*2,num_workers=0,pin_memory=True)
val_loader = DeviceDataLoader(val_loader,device)

input_size = 784
hidden_size = 32
num_classes = 10

model = MnistModel(input_size,hidden_size,num_classes).cuda(device)

history = fit(10,0.5,model,train_loader,val_loader)



torch.save(model,'Model/multi_layer_model.pt')
print('salvo')

