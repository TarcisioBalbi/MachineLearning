import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from random import randint,choice,seed
import pandas as pd
import os
import joblib


white = (255,255,255)
yellow = (255,255,0)
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)
orange = (255,128,0)


def colorCube(cube):
    cCube = np.empty((6,2,2,3),np.uint8)
    
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                #print(cube[i,j,k])
                if cube[i,j,k]== 0:
                    cCube[i,j,k] = white
                    
                elif cube[i,j,k] == 1:
                    cCube[i,j,k] = yellow
                    
                elif cube[i,j,k] == 2:
                    cCube[i,j,k] = green
                    
                elif cube[i,j,k] == 3:
                    cCube[i,j,k] = blue
                    
                elif cube[i,j,k] == 4:
                    cCube[i,j,k] = orange
                    
                    
                elif cube[i,j,k] == 5:
                    cCube[i,j,k] = red
                    
                else:
                    print('Invalid Color!')
   
    return cCube


def cubeView(cube):
    
    view = np.zeros((3,8,6),np.uint8)
    c = colorCube(cube)
   
    view[:,2,0] = c[2,0,0]
    view[:,3,0] = c[2,0,1]
    view[:,2,1] = c[2,1,0]
    view[:,3,1] = c[2,1,1]

    view[:,2,2] = c[0,0,0]
    view[:,3,2] = c[0,0,1]
    view[:,2,3] = c[0,1,0]
    view[:,3,3] = c[0,1,1]

    view[:,2,4] = c[3,0,0]
    view[:,3,4] = c[3,0,1]
    view[:,2,5] = c[3,1,0]
    view[:,3,5] = c[3,1,1]

    view[:,0,2] = c[5,0,0]
    view[:,1,2] = c[5,0,1]
    view[:,0,3] = c[5,1,0]
    view[:,1,3] = c[5,1,1]

    view[:,4,2] = c[4,0,0]
    view[:,5,2] = c[4,0,1]
    view[:,4,3] = c[4,1,0]
    view[:,5,3] = c[4,1,1]
    
    view[:,6,2] = c[1,0,0]
    view[:,7,2] = c[1,0,1]
    view[:,6,3] = c[1,1,0]
    view[:,7,3] = c[1,1,1]
    
    plt.imshow(np.swapaxes(view,2,0))
    
    plt.show(block = False)
    plt.pause(0.5)
    plt.close('all')
       
def sidesMap(side):
    if side == 0:
        return 1,2,3,4,5
    elif side == 1:
        return 0,2,3,5,4
    elif side == 2:
        return 3,1,0,4,5
    elif side == 3:
        return 2,0,1,4,5
    elif side == 4:
        return 5,2,3,1,0
    elif side == 5:
        return 4,2,3,0,1
    else:
        print('Invalid Side!')
   
def movement(cube,side,direction):
    #sides: 0 = front, 1= back, 2 = upper, 3 = botton, 4 = right, 5 = left
    #directions: 0 = 90° clockwise , 2 = 90° anti-clockwise
    direction -=1;
    cube[side] = np.rot90(cube[side],direction)
    
    oposite,upper,botton,right,left = sidesMap(side)

    if side==0:
        if direction == -1:
    
            temp1,temp2 = cube[2,1,0],cube[2,1,1]
            
            cube[2,1,0], cube[2,1,1] = cube[5,0,1], cube[5,1,1]
            
            cube[5,0,1], cube[5,1,1] = cube[3,0,0],cube[3,0,1]
            
            cube[3,0,0],cube[3,0,1] = cube[4,0,0], cube[4,1,0]
            
            cube[4,0,0], cube[4,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,1,0],cube[2,1,1]
            
            cube[2,1,0], cube[2,1,1] = cube[4,0,0], cube[4,1,0]
            
            cube[4,0,0], cube[4,1,0] = cube[3,0,0],cube[3,0,1]
            
            cube[3,0,0],cube[3,0,1] = cube[5,0,1], cube[5,1,1]
            
            cube[5,0,1], cube[5,1,1] = temp1,temp2
            
    elif side==1:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,0],cube[2,0,1]
            
            cube[2,0,0], cube[2,0,1] = cube[4,0,1], cube[4,1,1]
            
            cube[4,0,1], cube[4,1,1] = cube[3,1,1],cube[3,1,0]
            
            cube[3,1,1],cube[3,1,0] = cube[5,0,0], cube[5,1,0]
            
            cube[5,0,0], cube[5,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,0],cube[2,0,1]
            
            cube[2,0,0], cube[2,0,1] = cube[5,0,0], cube[5,1,0]
            
            cube[5,0,0], cube[5,1,0] = cube[3,1,1],cube[3,1,0]
            
            cube[3,1,1],cube[3,1,0] = cube[4,0,1], cube[4,1,1] 
            
            cube[4,0,1], cube[4,1,1] = temp1,temp2
            
    elif side==2:
        if direction == -1:
    
            temp1,temp2 = cube[1,0,0],cube[1,0,1]
            
            cube[1,0,0], cube[1,0,1] = cube[5,0,0], cube[5,0,1]
            
            cube[5,0,0], cube[5,0,1] = cube[0,0,0],cube[0,0,1]
            
            cube[0,0,0],cube[0,0,1] = cube[4,0,0], cube[4,0,1]
            
            cube[4,0,0], cube[4,0,1] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[1,0,0],cube[1,0,1]
            
            cube[1,0,0], cube[1,0,1] = cube[4,0,0], cube[4,0,1]
            
            cube[4,0,0], cube[4,0,1] = cube[0,0,0],cube[0,0,1]
            
            cube[0,0,0], cube[0,0,1] = cube[5,0,0], cube[5,0,1] 
            
            cube[5,0,0], cube[5,0,1] = temp1,temp2
            
    elif side==3:
        if direction == -1:
    
            temp1,temp2 = cube[1,1,0],cube[1,1,1]
            
            cube[1,1,0], cube[1,1,1] = cube[4,1,0], cube[4,1,1]
            
            cube[4,1,0], cube[4,1,1] = cube[0,1,0],cube[0,1,1]
            
            cube[0,1,0],cube[0,1,1] = cube[5,1,0], cube[5,1,1]
            
            cube[5,1,0], cube[5,1,1] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[1,1,0],cube[1,1,1]
            
            cube[1,1,0], cube[1,1,1] = cube[5,1,0], cube[5,1,1]
            
            cube[5,1,0], cube[5,1,1]  = cube[0,1,0],cube[0,1,1]
            
            cube[0,1,0],cube[0,1,1] =  cube[4,1,0], cube[4,1,1]
            
            cube[4,1,0], cube[4,1,1] = temp1,temp2
            
    elif side==4:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,1],cube[2,1,1]
            
            cube[2,0,1],cube[2,1,1] = cube[0,0,1], cube[0,1,1]
            
            cube[0,0,1], cube[0,1,1] = cube[3,0,1],cube[3,1,1]
            
            cube[3,0,1],cube[3,1,1] = cube[1,1,0], cube[1,0,0]
            
            cube[1,1,0], cube[1,0,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,1],cube[2,1,1]
            
            cube[2,0,1],cube[2,1,1] =  cube[1,1,0], cube[1,0,0]
            
            cube[1,1,0], cube[1,0,0] = cube[3,0,1],cube[3,1,1]
            
            cube[3,0,1],cube[3,1,1] =  cube[0,0,1], cube[0,1,1]
            
            cube[0,0,1], cube[0,1,1] = temp1,temp2
    elif side==5:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,0],cube[2,1,0]
            
            cube[2,0,0],cube[2,1,0] = cube[1,1,1], cube[1,0,1]
            
            cube[1,1,1], cube[1,0,1] = cube[3,0,0],cube[3,1,0]
            
            cube[3,0,0],cube[3,1,0] = cube[0,0,0], cube[0,1,0]
            
            cube[0,0,0], cube[0,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,0],cube[2,1,0]
            
            cube[2,0,0],cube[2,1,0] =  cube[0,0,0], cube[0,1,0]
            
            cube[0,0,0], cube[0,1,0] = cube[3,0,0],cube[3,1,0]
            
            cube[3,0,0],cube[3,1,0] = cube[1,1,1], cube[1,0,1]
            
            cube[1,1,1], cube[1,0,1]  = temp1,temp2
            
            
       
    return cube

def moveMap(movement):
    return int(movement[0]),int(movement[2])
    
        
def dataGenerator():
    cube22 = np.array([
                      [[0,0],[0,0]], #front
                      [[1,1],[1,1]], #back
                      [[2,2],[2,2]], #upper
                      [[3,3],[3,3]], #botton
                      [[4,4],[4,4]], #right
                      [[5,5],[5,5]]  #left
                      ])

    
    
    
    positions = [cube22.flatten()]
    moves = ['x']
    for j in range(100000):

        cube22 = np.array([
                          [[0,0],[0,0]], #front
                          [[1,1],[1,1]], #back
                          [[2,2],[2,2]], #upper
                          [[3,3],[3,3]], #botton
                          [[4,4],[4,4]], #right
                          [[5,5],[5,5]]  #left
                          ])
        
        positions.append(cube22.flatten())
        moves.append('x')        
    
        for i in range(20):
            side = randint(0,5)
            ori = choice([0,2])
            #print(side,ori)
            cube22= movement(cube22,side,ori)
            
            if ori==0 and any(cube22.flatten()!= positions[0]):
                moves.append(str(side)+','+'2')
            elif ori==2 and any(cube22.flatten()!=positions[0]):
                moves.append(str(side)+','+'0')
            else:
                moves.append('x')
                
            positions.append(cube22.flatten())
        
   
    
    return positions,moves

def play(cube,model):
    cubeView(cube)
    vector = cube.flatten()
    move = str(model.predict(vector.reshape(1,-1)))
    move = move[2:-2]
    print(move)
    if move != 'x':
        side,ori = moveMap(move)
        cube = movement(cube,side,ori)
        play(cube,model)
    

columns1 = []
for i in range(24):
    columns1.append(str(i))

toSaveData = False
if os.path.exists('cubeDS.csv'):
    cubeDF = pd.read_csv('cubeDS.csv')
else:
    seed(10)
    toSaveData = True
    data,labels = dataGenerator()

    cubeDF = pd.DataFrame(data=data,columns = columns1)
    cubeDF['Move'] = labels

print(cubeDF.head(10))

X = cubeDF.drop('Move',axis=1)
y = cubeDF['Move']

from sklearn.model_selection import train_test_split,GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
fileName = 'model.sav'

#def rfcModel(n_estimators,criterion,)

toSaveModel = False
if os.path.exists(fileName):
    rfc = joblib.load(fileName)
else:
    rfc = RandomForestClassifier()
    parametros = {'n_estimators':[20,50,100],
                  'criterion':['gini','entropy'],
                  'min_samples_split':[2,3]}

    grid_search = GridSearchCV(estimator = rfc,param_grid=parametros,cv = 2)
    grid_search.fit(X_train, y_train)

    melhores_param = grid_search.best_params_
    print(melhores_param)
    toSaveModel = True



##letsPlay = True
##while(letsPlay):
##    cube22 = np.array([
##                          [[0,0],[0,0]], #front
##                          [[1,1],[1,1]], #back
##                          [[2,2],[2,2]], #upper
##                          [[3,3],[3,3]], #botton
##                          [[4,4],[4,4]], #right
##                          [[5,5],[5,5]]  #left
##                          ])
##
##
##    for i in range(7):
##        side = randint(0,5)
##        print(side)
##        ori = choice([0,2])
##        cube22= movement(cube22,side,ori)
##        #cubeView(cube22)
##
##
##    play(cube22,rfc)
##    go = int(input('continuar?'))
##    if go ==1:
##        letsPlay = True
##    else:
##        letsPlay = False

if toSaveData:
    cubeDF.to_csv('cubeDS.csv',index=False)

    
if toSaveModel:
    joblib.dump(rfc, fileName)


