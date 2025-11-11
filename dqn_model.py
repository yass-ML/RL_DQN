import torch
import os


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

        #self.dropout = torch.nn.Dropout(p=0.2)

        self.to(device)


    def forward(self,x):
        x = x.to(self.device)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.dense(x))
        return self.output(x)
    

    def save(self, save_path="models/latest.pth"):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(self.state_dict(), save_path)

    def load(self, load_path="models/lastest.pth"):
        try:
            self.load_state_dict(torch.load(load_path))
            print(f"Sucessfully loaded saved DQN model {load_path}")
        except:
            print(f"No saved model to load at {load_path}")
