import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy
import os

NEpoches = 500
BatchSize = 1000

TrainSet = torchvision.datasets.MNIST("./data/train",
									  train = True,
									  transform = torchvision.transforms.ToTensor(),
									  download = True)

TestSet = torchvision.datasets.MNIST("./data/test",
									  train = False,
									  transform = torchvision.transforms.ToTensor(),
									  download = True)

TrainData = torch.utils.data.DataLoader(TrainSet,
										batch_size = BatchSize,
										num_workers = 0)
TestData = torch.utils.data.DataLoader(TestSet,
									   batch_size = BatchSize,
									   num_workers = 0)

class MLP(torch.nn.Module):

	def __init__(self):
		super(MLP,self).__init__()
		self.layer1 = torch.nn.Linear(28 * 28,512)
		self.layer2 = torch.nn.Linear(512,128)
		self.layer3 = torch.nn.Linear(128,10)
		
	def forward(self,a):
		b = a.view(-1,28 * 28)
		b = torch.nn.functional.relu(self.layer1(b))
		b = torch.nn.functional.relu(self.layer2(b))
		return torch.nn.functional.softmax(self.layer3(b),dim = 1)


model = MLP().cuda()
LossFunction = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

def main(): 
	print(model)
	sepoch = -1
	MinLoss = 100

	RestartTraining = True
	if RestartTraining:
		checkpoint = torch.load('./model/ckpt723.pth') # fill in the path
		sepoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		MinLoss = checkpoint['loss']
	print(MinLoss)

	for epoch in range(sepoch + 1,NEpoches + sepoch + 1):
		TestLoss = train(epoch)
		test()
		print('====> Cur Loss: {:.6f} Best Loss: {:.6f}'.format(TestLoss,MinLoss))
		if epoch > int((sepoch * 2 + 2 + NEpoches) / 2) and TestLoss < MinLoss:
			print('Saving checkpoint...')
			MinLoss = TestLoss
			checkpoint = {
				'epoch':epoch,
				'model':model.state_dict(),
				'optimizer':optimizer.state_dict(),
				'loss':TestLoss}
			if not os.path.isdir('./model'):
				os.mkdir('./model')
			torch.save(checkpoint,'./model/ckpt{}.pth'.format(epoch + 1))


def train(epoch):
	totloss = 0.0
	for data in TrainData:
		optimizer.zero_grad()

		images,labels = data
		images = torch.autograd.Variable(images).cuda()
		labels = torch.autograd.Variable(labels).cuda()

		out = model(images)
		loss = LossFunction(out,labels)
		loss.backward()

		optimizer.step()
		totloss += loss.item() * images.size(0)
	totloss /= len(TrainData.dataset)
	print('====> Epoch: {} Loss: {:.6f}'.format(epoch + 1,totloss))
	return totloss

def test():
	acc = 0
	tot = 0
	with torch.no_grad():
		for data in TestData:
			images,labels = data
			images = torch.autograd.Variable(images).cuda()
			labels = torch.autograd.Variable(labels).cuda()
			_,out = torch.max(model(images).data,1)
			tot += labels.size(0)
			acc += (out == labels).sum().item()
	print('====> Accuracy: {:.6f}'.format(100.0 * acc / float(tot)))
	return (100.0 * acc / float(tot))


if __name__ == '__main__':
	assert torch.cuda.is_available(), "CPU training is not allowed."
	main()