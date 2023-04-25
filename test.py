import torch.nn as nn
import torch
from truncate import ETC, NormTruncate
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt



class TestModel(nn.Module):
    def __init__(self) -> None:
        super(TestModel,self).__init__()
        self.conv1 = nn.Conv2d(3,2,2)
        self.fc1 = nn.Linear(18,1)
        # nn.init.normal_(self.conv1)
        # nn.init.normal_(self.fc1)

    
    def forward(self,x):
        conv = self.conv1(x)
        conv = conv.reshape(1,-1)
        fc = self.fc1(conv)

        return fc
    
def print_grad(m):
    print(50*"=")
    for p in m.parameters():
        print(p.grad)
def main():
    sw = SummaryWriter(log_dir="test")
    model = TestModel()
    etc = NormTruncate(model,0.1)


    for epoch in range(5):
        for step in range(5):
            x = torch.ones((3,4,4))
            y = model(x)
            loss = (torch.tensor(1)-y)
            loss.backward()
            # print("\n"*5)
            print(epoch,step)
            # print("\n"*5)
            # print_grad(model)
            sparse = etc.step(model)
            print(sparse)
            # print_grad(etc.model)
            # print_grad(model)
            model.zero_grad()
        etc.reset()

if __name__=="__main__":

    main()