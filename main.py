import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy
import gradio

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
checkpoint = torch.load('./model/ckptBest.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

def recognition(image):
	image = torch.tensor(image)
	image = image.float()
	image.div_(255)
	with torch.no_grad():
		image = torch.autograd.Variable(image).cuda()
		out = model(image).tolist()[0]
	print(out)
	return {str(i): out[i] for i in range(10)}

def GradioMain():
	label = gradio.outputs.Label(num_top_classes=4)
	ui = gradio.Interface(
		fn = recognition,
		inputs="sketchpad",
		outputs=label,
		live = True,
		title = 'Hand Written Digit Recognizer'
	)
	ui.launch()

if __name__ == '__main__':
	GradioMain()