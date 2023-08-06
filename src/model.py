import torch
import torch.nn as nn
import torch.optim as optim
import config
import numpy as np

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class CustomModel(nn.Module):
    def __init__(self, num_residual_blocks=config.NUM_HIDDEN_LAYERS):
        super(CustomModel, self).__init__()
        self.conv_layer = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(num_residual_blocks)])

        # Head 1: Outputs a constant float
        self.head1 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # Head 2: Outputs a (73, 8, 8) tensor
        self.head2 = nn.Sequential(
            nn.Conv2d(64, 73, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if type(x) == np.ndarray:
            x = x.reshape(19, 8, 8)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        out = self.conv_layer(x)
        out = self.bn(out)
        out = self.relu(out)

        for block in self.residual_blocks:
            out = block(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor for the fully connected layer

        output1 = self.head1(out)
        output2 = self.head2(out.view(out.size(0), 64, 8, 8))
        return output2, output1
    
    def train_model(self, input_array, label_array1, label_array2, num_epochs=10, batch_size=32, learning_rate=0.001):
        criterion1 = nn.MSELoss()  # Mean Squared Error loss for the constant float output
        criterion2 = nn.MSELoss()  # Mean Squared Error loss for the (73, 8, 8) output
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Convert numpy arrays to PyTorch tensors
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        label_tensor1 = torch.tensor(label_array1, dtype=torch.float32)
        label_tensor2 = torch.tensor(label_array2, dtype=torch.float32)

        num_samples = input_tensor.size(0)
        num_batches = num_samples // batch_size

        for epoch in range(num_epochs):
            running_loss1 = 0.0
            running_loss2 = 0.0

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                inputs = input_tensor[start_idx:end_idx]
                labels1 = label_tensor1[start_idx:end_idx]
                labels2 = label_tensor2[start_idx:end_idx]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output1, output2 = self(inputs)

                # Compute the loss for each output
                loss1 = criterion1(output1.view(-1), labels1.view(-1))
                loss2 = criterion2(output2, labels2)

                # Compute the total loss by summing the two losses
                total_loss = loss1 + loss2

                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()

                # Update running losses
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()

            # Print the average loss for each epoch
            avg_loss1 = running_loss1 / num_batches
            avg_loss2 = running_loss2 / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}")
