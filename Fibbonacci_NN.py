# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:49:50 2021

@author: divyam
"""
import numpy as np
import torch
import _pickle as pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
def fib(n):

    n = int(n)
    s = {}
    s[1] = 1
    s[2] = 1

    for i in range(3, n+1):
        s[i] = (s[i-1] + s[i-2]) % 1000007

    return np.float64(s[n]) 


def sin(x):
    return np.sin(np.radians(x))

X = np.arange(1, 100000, dtype=np.float64)
Y = np.array([sin(x) for x in X])

total_data = len(X)
train_data_size = int(total_data * 0.8)
test_data_size = total_data - train_data_size

train_X = X[:train_data_size]
train_Y = Y[:train_data_size]

test_X = X[train_data_size:]
test_Y = Y[train_data_size:]


for j in range(10):
    print(train_X[j], train_Y[j])
  

'''
Network - A MLP classifier to classify the features of the superpixels into given class-labels

'''
# Model definition
class FibonacciFunctionNetwork(nn.Module):
    def __init__(self, num_inputs=100, num_outputs=100):  
      # Your code 
        super(FibonacciFunctionNetwork, self).__init__()
      
        self.model = nn.Sequential( nn.Linear(num_inputs, 1024), 
                                  nn.ReLU(inplace=True), 
                                  #nn.Dropout(0.5),
                                  nn.Linear(1024, 1024), 
                                  nn.ReLU(inplace=True), 
                                  #nn.Dropout(0.5),
                                  nn.Linear(1024, num_outputs)
                                 
                              )
      
      #self.model.to(device)
      
    def forward(self, x):
        x = self.model(x)
        return x
  
    
## hyperparams
learning_rate = 0.0001
NUM_EPOCH = 20
BATCH_ITER = 20
BATCH_SIZE = train_data_size // BATCH_ITER


model = FibonacciFunctionNetwork(BATCH_SIZE, BATCH_SIZE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import time
model.train()
start = time.clock()
for epoch in range(NUM_EPOCH):
      losses = []
      running_loss = 0
      running_corrects = 0
      num_samples = 0
      
      for idx in range(BATCH_ITER):
          

          trainX = torch.from_numpy(train_X[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]).float()
          trainY = torch.from_numpy(train_Y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]).float()
        
          #image = image.to(device=device)
          #print(image[0].size)
          #labels = labels.to(device=device)

          # forward
          predicted_output = model(trainX)
          #_, preds = torch.max(predicted_labels, 1)

          loss = criterion(predicted_output, trainY)

          #losses.append(loss.item())

          #running_corrects += torch.sum(preds == labels)
          #num_samples += predicted_labels.size(0)


          # backward
          optimizer.zero_grad()
          loss.backward()


          #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
          clipping_value = 0.1 # arbitrary value of your choosing
          torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
          # gradient descent or adam step
          optimizer.step()

          #batch_loop.set_description(f'Epoch [{epoch}/{NUM_EPOCH}]')
          #batch_loop.set_postfix(loss=loss.item(), acc=(running_corrects/num_samples).item())

          #self.running_loss.append(sum(losses)/len(losses))
      
      #avg_loss = sum(losses)/len(losses)
      #avg_acc = running_corrects/num_samples
      #self.scheduler.step(avg_loss)

      #self.epoch_acc.append(avg_acc)
      #self.epoch_loss.append(avg_loss)

      time_elapsed = time.clock() - start
      print('Epoch: [{}/{}] Time: {}min:{}sec'.format(epoch, NUM_EPOCH, time_elapsed//60, time_elapsed%60))


      #self.scheduler()
      
testX = torch.from_numpy(test_X[:BATCH_SIZE]).float()   


output = model(testX)

for j in range(2):
    print(testX[j].item(), output[j].item(), test_Y[j])
  
import matplotlib.pyplot as plt

plt.plot(testX.detach().numpy()[:1000], output.detach().numpy()[:1000])
plt.show()

a = list(map(lambda x:x*x, [1,2,3,4]))



def add(str1, str2, n1, n2, idx, sum, carry):
    if (n1-idx <= 0 and n2-idx <= 0):
        return sum
    if (n1-idx <= 0):
        s = (int(str2[n2-1-idx]) + carry)
        carry = s // 10
        sum = str(s%10) + sum
        return add(str1, str2, n1, n2, idx+1, sum, carry)
    if (n2-idx <= 0):
        s = (int(str1[n1-1-idx]) + carry)
        carry = s // 10
        sum = str(s%10) + sum
        return add(str1, str2, n1, n2, idx+1, sum, carry)

    s = (int(str1[n1-1-idx]) + int(str2[n2-1-idx]) + carry)
    carry = s // 10
    sum = str(s%10) + sum
    return add(str1, str2, n1, n2, idx+1, sum, carry)
  


