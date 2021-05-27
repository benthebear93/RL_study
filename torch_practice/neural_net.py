import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
			nn.ReLU()
		)
	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

#init model
model = NeuralNetwork().to(device) #send to device
print(model)

#send data
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#model layer
input_image = torch.rand(3,28,28) #bring 28x28 size 3 image minibatch 
# see what happend when they go throu the NN. 
print(input_image.size())

#flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#linear layer for linear transformation with weight and bias
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#nonlinear activation makes complecate mapping between input and output
#help NN to learn different env
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#softmax, return logits which represent class prediction probability.
#scale t [0,1]
#dim means 차원 that sum is 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#model parameterize 
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
