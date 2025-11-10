import torch



class DQN(torch.nn.Module):
    def __init__(self, n_actions, dense_hidden_size=32*9*9,device= "cpu"):
        super().__init__()
        self.device = device

        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.relu2 = torch.nn.ReLU()

        self.dense = torch.nn.Linear(in_features=dense_hidden_size, out_features=256)
        self.relu3 = torch.nn.ReLU()
        self.output = torch.nn.Linear(in_features=256, out_features=n_actions)

        self.to(device)


    def forward(self,x):
        x = x.to(self.device)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.dense(x))
        return self.output(x)
