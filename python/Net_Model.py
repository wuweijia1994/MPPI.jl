import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, action_dm, state_dm):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dm, state_dm)
        self.fc2 = nn.Linear(state_dm, state_dm)
        self.fc3 = nn.Linear(state_dm, action_dm)

        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y):
        #import pdb; pdb.set_trace()
        y_pred = self.forward(x)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        print(loss.item())

if __name__ == "__main__":
    N, D_in, H, D_out = 64, 100, 100, 10

    net = Net(D_out, D_in)
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    #x = np.random.randn(N, D_in)
    #y = np.random.randn(N, D_out)

    for i in range(10):
        net.train(x, y)

