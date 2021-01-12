#https://www.youtube.com/watch?v=sJF6PiAjE1M
import os
import torch
import torchvision
import tarfile
import numpy as np
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F



def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
        
        
class ImageClassificationBase(nn.Module):
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
        return{'val_loss':loss.detach(),'val_acc':acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc =torch.stack(batch_accs).mean()
        return{'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item}

    def epoch_end(self,epoch,result):
        print("Epoch[{}],last_lr:{:.5f},train_loss:{:.4f},val_loss: {:.4f},val_acc: {:.4f}".format(
            epoch,result['lrs'][-1],result['train_loss'],result['val_loss'],result['val_acc']))

def conv_block(in_channels,out_channels,pool=False):
    layers =[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
    
class ResNet9(ImageClassificationBase):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels,64)
        self.conv2 = conv_block(64,128)
        self.res1 = nn.Sequential(conv_block(128,128),conv_block(128,128))

        self.conv3 = conv_block(128,256,pool=True)
        self.conv4 = conv_block(256,512,pool=True)
        self.res2 = nn.Sequential(conv_block(512,512),conv_block(512,512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512,num_classes))
        
    def forward(self,xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)+out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)+ out
        out = self.classifier(out)
        return out





    
def apply_kernel(image,kernel):
    ri,ci = image.shape
    rk,ck = kernel.shape
    ro,co = ri-rk+1,ci-ck+1
    output = torch.zeros([ro,co])
    for i in range(ro):
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck]*kernel)
    return output


            

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
        
@torch.no_grad()
def evaluate(model,val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    
    for param_group in optimizer.paramm_group:
        return param_group['lr']
    
def fit_one_cycle(epochs,max_lr,model,train_loader,val_loader,
                  weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs = epochs,
                                                steps_per_epoch = len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model,val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] =lrs
        model.epoch_end(epoch,result)
        history.append(result)
    return history


#dataset_url = "http://files.fast.ai/data/cifar10.tgz"
#download_url(dataset_url,'.')

#with tarfile.open('./cifar10.tgz','r:gz') as tar:
#    tar.extractall(path='./data')

device = get_default_device()


data_dir = './data/cifar10'

dataset = ImageFolder(data_dir+'/train',transform=ToTensor())

stats = ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
train_tfms = tt.Compose([ tt.RandomCrop(32,padding=4,padding_mode = 'reflect'),
                          tt.RandomHorizontalFlip(),
                          tt.ToTensor(),
                          tt.Normalize(*stats,inplace=True)])

valid_tfms = tt.Compose([tt.ToTensor(),tt.Normalize(*stats)])

train_ds = ImageFolder(data_dir+'/train',train_tfms)
val_ds =  ImageFolder(data_dir+'/test',valid_tfms)

batch_size = 400

train_dl = DataLoader(train_ds,batch_size,shuffle=True)
train_dl = DeviceDataLoader(train_dl,device)

val_dl = DataLoader(val_ds,batch_size*2)
val_dl = DeviceDataLoader(val_dl,device)

model = to_device(ResNet9(3,10),device)

num_epochs = 1
max_lr =0.01
grad_clip =0.1
weight_decay =1e-4
opt_func = torch.optim.Adam


history = fit_one_cycle(num_epochs,max_lr,model,train_dl,val_dl,
                        grad_clip=grad_clip,weight_decay=weight_decay,opt_func=opt_func)

#torch.save(model.state_dict(),'cifar10-cnn.pth')

#model2 = to_device(Cifar10nnModel(),device)
#model2.load_state_dict(torch.load('cifar10-cnn.pth'))




