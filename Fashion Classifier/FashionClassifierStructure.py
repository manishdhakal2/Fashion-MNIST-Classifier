#Actual model code
import pandas as pd
import torch
from torch import nn
import numpy as np
import torch.optim  as optim
import matplotlib.pyplot as plt
device="cuda" if torch.cuda.is_available() else "cpu"

class FashionClassifier(nn.Module):
    def __init__(self,lr,epochs,neurons):
        super().__init__()
        self.lr=lr
        self.epochs=epochs
        self.neuron_no=neurons
        self.l1=nn.Linear(784,self.neuron_no,device=device)
        self.l2=nn.Linear(self.neuron_no,self.neuron_no,device=device)
        self.l3=nn.Linear(self.neuron_no,10,device=device)
        self.relu=nn.ReLU()
        self.optimizer=optim.SGD(params=self.parameters(),momentum=0.9,lr=self.lr)
        nn.init.kaiming_uniform_(self.l1.weight,nonlinearity='relu')
        nn.init.kaiming_uniform_(self.l2.weight,nonlinearity='relu')
        nn.init.kaiming_uniform_(self.l3.weight,nonlinearity='relu')
        

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        return self.l3(x)

    def fit(self,x:torch.Tensor,y:torch.Tensor)->None:
        self.train()
        loss_fn=nn.CrossEntropyLoss()
        for i in range(self.epochs):
            print(f"Iteration {i}")
            y_pred=self.forward(x)
            loss=loss_fn(y_pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.eval()
            with torch.inference_mode():
                if i%10==0:
                    print(f"Loss={loss}")
            self.train()
    