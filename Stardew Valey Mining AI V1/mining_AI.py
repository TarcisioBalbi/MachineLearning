import keys as k
from grabscreen import grab_screen
import time 
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
        ax.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
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
            nn.Linear(512, 3))
        
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

def findDistance(img,x,y):
    distX = img.shape[1]/2-x+50
    distY = img.shape[0]/2-y
    dist = np.sqrt(distX**2+distY**2)
    return distX,distY,dist    

def findNextRock(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB  )
    slices,centers = sliceImage(img)

    labels = []

    for sl in slices:
        dsize = (32, 32)
        img_test = cv.resize(sl, dsize)
        img_test = np.rollaxis(img_test,2,0)
    
        img_test = torch.Tensor(img_test)

        labels.append(predict_image(img_test, model))

    font = cv.FONT_HERSHEY_SIMPLEX 

    org = (50, 50)
    fontScale = 1

    color = (255, 0, 0) 

    thickness = 2
    minDist = 1920
    nextCenter = [centers[0][0],centers[0][1]]

    ## ESCOLHE UM CENTRO ALEATÓRIO
##    aux_label = 'floor'
##    while(aux_label!='rock'):
##        
##        i = np.random.randint(0,len(centers))
##        aux_label = labels[i]
##        nextCenter  = [centers[i][0],centers[i][1]]
##
##    for i in range(len(centers)):
##        org = (int(centers[i][1]),int(centers[i][0]))
##        if labels[i] == 'rock':
##            color = (0,255,0)
##                   
##        else:
##            color=(255,0,0)
##    
##        img = cv.putText(img, '.', org, font,  
##                   fontScale, color, thickness, cv.LINE_AA) 
##            
##    
####
    ##ESCOLHE CENTRO MAIS PROXIMO
    for i in range(len(centers)):
        
        org = (int(centers[i][1]),int(centers[i][0]))
        if labels[i] == 'rock':
            color = (0,255,0)
            dx,dy,dist = findDistance(img,centers[i][1],centers[i][0])
            
            if dist<minDist and dist>80:
                
                minDist =dist
                nextCenter = [centers[i][0],centers[i][1]]
                
        else:
            color=(255,0,0)
        
        img = cv.putText(img, '.', org, font,  
                   fontScale, color, thickness, cv.LINE_AA) 

    return img,nextCenter
def randMove():
    moves = ['w','s','d','a']
    i= np.random.randint(0,4)
    print(moves[i])
    keys.directKey(moves[i])
    time.sleep(abs(2))
    keys.directKey(moves[i],keys.key_release)

    
def walk(distX,distY):
    keys = k.Keys()
            
    key = ['','']
    dist = [distX,distY]
    if distX>0:
        key[0] = 'a'
    elif distX<0:
        key[0] = 'd'
    else:
        pass

    if distY>0:
        key[1] = 'w'
    elif distY<0:
        key[1] = 's'
    else:
        pass

    if distX>=distY:

        temp_key = key[0]
        temp_dist = dist[0]        
        
        key[0] = key[1]
        dist[0] = dist[1]

        key[1] = temp_key
        dist[1] = temp_dist
        
        
        
        
    keys.directKey(key[0])
    time.sleep(abs(dist[0]/280))
    keys.directKey(key[0],keys.key_release)
    time.sleep(0.5)
    keys.directKey(key[1])
    time.sleep(abs(dist[1]/280))
    keys.directKey(key[1],keys.key_release)
    
    time.sleep(1)
    
    keys.directKey('c')
    time.sleep(0.05)
    keys.directKey('c',keys.key_release)
    
    time.sleep(0.5)
    
    keys.directKey(key[0])
    time.sleep(0.01)
    keys.directKey(key[0],keys.key_release)
    
    time.sleep(0.5)
    
    keys.directKey('c')
    time.sleep(0.05)
    keys.directKey('c',keys.key_release)
    


dataDir = './augData/'
dataset = ImageFolder(dataDir, transform=ToTensor())
device = get_default_device()

model = SVMiningModel()
model.load_state_dict(torch.load('./Model/Model11'))
to_device(model, device)


img = cv.imread("./raw data/mine_lvl5_2.png")
#plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
#plt.show()
img = img[350:800,600:1300]

out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 2, (img.shape[1],img.shape[0]))
font = cv.FONT_HERSHEY_SIMPLEX 


fontScale = 1
color = (255, 255, 0) 

thickness = 2


for i in range(3):
    time.sleep(1)

keys = k.Keys()
thickness2 = 3
for i in range(10):
    
    screen = grab_screen(region=(0,0,1920,1080))
    screen = cv.cvtColor(screen,cv.COLOR_BGR2RGB)
    screen = screen[350:800,600:1300]
    
    img,center = findNextRock(screen)
    distX,distY,_=findDistance(img,center[1],center[0])
    walk(distX,distY)
    randMove()
    org = (int(center[1]),int(center[0]))
    end_point =(int(center[1]),int(center[0]))
    start_point=  (int(img.shape[1]/2+50),int(img.shape[0]/2))
    
    img = cv.arrowedLine(img, start_point, end_point, color, thickness2) 
    
    out.write(cv.cvtColor(img,cv.COLOR_RGB2BGR))
    
    
out.release() 
