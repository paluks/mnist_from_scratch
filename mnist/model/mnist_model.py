import torch.nn as nn
import torch

class MnistModel(nn.Module):
	def __init__(self): 
		super().__init__()


		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2)

		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=2)

		self.fc1 = nn.Linear(in_features=32*5*5, out_features=128)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=128, out_features=10)

	
	def forward(self, x): 
		x = self.conv1(x) 
		x = self.relu1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		x = self.relu3(x)
		x = self.fc2(x)

		

		return x


