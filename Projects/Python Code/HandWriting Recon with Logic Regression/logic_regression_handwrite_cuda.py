
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,num_classes)

    def forward(self,xb):
        xb = xb.reshape(-1,784)
        out = self.linear(xb)
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

    
cuda = torch.device('cuda')

dataset = MNIST(root='data/',download = True)

test_dataset=MNIST(root='data/',train=False)

dataset = MNIST(root='data/',train = True ,transform = transforms.Compose([transforms.ToTensor(), lambda x : x.cuda()]))

test_dataset=MNIST(root='data/',train=False,transform = transforms.Compose([transforms.ToTensor(),lambda x : x.cuda()]))


dataset = [(dataset[i][0],torch.tensor(dataset[i][1]).cuda()) for i in range(0,len(dataset))]

  
train_ds,val_ds = random_split(dataset,[50000, 10000])

batch_size = 128

train_loader = DataLoader(train_ds,batch_size,shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28*28 #numbem of pixels in each image
num_classes = 10

model = MnistModel().cuda()

history1 = fit(100,0.001,model,train_loader,val_loader)

torch.save(model,'Model/num_model.pt')
print('salvo')

print("Examples')

img,label = test_dataset[0]

print('Label:',label,'Predict:',predict_image(img,model))

img,label = test_dataset[10]

print('Label:',label,'Predict:',predict_image(img,model))

img,label = test_dataset[15]

print('Label:',label,'Predict:',predict_image(img,model))

img,label = test_dataset[17]

print('Label:',label,'Predict:',predict_image(img,model))

img,label = test_dataset[155]

print('Label:',label,'Predict:',predict_image(img,model))

img,label = test_dataset[200]

print('Label:',label,'Predict:',predict_image(img,model))

















