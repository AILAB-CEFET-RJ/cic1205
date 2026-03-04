class Net(nn.Module):
    def __init__(self, in_channels, input_length):
        super(Net, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, 64, kernel_size=3)
        self.relu = nn.ReLU()
        
        # Compute the output length after the Conv1d layer
        # Assuming no padding and stride of 1
        conv_out_length = input_length - 3 + 1  # Length after applying kernel_size=3
        self.fc1 = nn.Linear(64 * conv_out_length, 50)  # Adjusted the input size for fc1
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)  # Flattening the output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x