import os
import torch
import torchvision
import tarfile
import cv2 as cv
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


matplotlib.rcParams['figure.facecolor'] = '#ffffff'

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def showImage(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        print(images.shape)
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        plt.show()
        break

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class SVMiningModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        
        for batch in train_loader:
            
            loss = model.training_step(batch)
            
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(img, model):
    
    xb = to_device(img.unsqueeze(0), device)
    
    yb = model(xb)
    
    _, preds  = torch.max(yb, dim=1)
    
    
    return dataset.classes[preds[0].item()]


def sliceImage(img):
    slices = []
    centers = []
    for i in range(0,img.shape[0]-80,40):
        for j in range(0,img.shape[1]-80,40):
            newImage = img[i:i+80,j:j+80]
            slices.append(newImage)
            centers.append([(2*i+80)/2,(2*j+80)/2])
            

    return slices,centers

def findRocks(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB  )
    img = img[350:800,600:1300]


    slices,centers = sliceImage(img)

    labels = []

    for sl in slices:
        
        dsize = (32, 32)
        img_test = cv.resize(sl, dsize)
        img_test = np.rollaxis(img_test,2,0)
    
        img_test = torch.Tensor(img_test)
        lb = predict_image(img_test, model)
        labels.append(lb)
        

    font = cv.FONT_HERSHEY_SIMPLEX 

    org = (50, 50)
    fontScale = 1

    color = (255, 0, 0) 

    thickness = 2

    for i in range(len(centers)):
        
        org = (int(centers[i][1]),int(centers[i][0]))
        if labels[i] == 'rock':
            color = (0,255,0)

        elif labels[i] == 'border':
            color= (0,0,255)
        else:
            color=(255,0,0)
        
        img = cv.putText(img, '.', org, font,  
                   fontScale, color, thickness, cv.LINE_AA) 

  
    plt.imshow(img)
    plt.show()

    
device = get_default_device()
print(torch.cuda.get_device_name())
print(device)


dataDir = './augData/'
classes = os.listdir(dataDir)
print(classes)

data_transforms = transforms.Compose([
        ToTensor()
                ])
dataset = ImageFolder(dataDir, transform=data_transforms)
#print(len(dataset))

#showImage(*dataset[51])


valSize = 400
trainSize = len(dataset) - valSize

trainDs, valDs = random_split(dataset, [trainSize, valSize])
len(trainDs), len(valDs)

batchSize=64

trainDl = DataLoader(trainDs, batchSize, shuffle=True)
valDl = DataLoader(valDs, batchSize*2)

#show_batch(valDl)

model = SVMiningModel()
#print(model)
trainDl = DeviceDataLoader(trainDl, device)
valDl = DeviceDataLoader(valDl, device)
to_device(model, device)

model = to_device(SVMiningModel(), device)
#print(evaluate(model, valDl))


num_epochs = 12
opt_func = torch.optim.Adam
lr = 0.00001
history = fit(num_epochs, lr, model, trainDl, valDl, opt_func)

img, label = dataset[112]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
directory = './test images/'

for filename in os.listdir(directory):
        if filename.endswith(".PNG") or filename.endswith(".png"):
            img = cv.imread(os.path.join(directory, filename))
            findRocks(img)
        else:
            continue

torch.save(model.state_dict(), './Model/Model11')
