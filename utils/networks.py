import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    """Residual Block for deep networks (ResNet-style)."""
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        return F.relu(x)
    
class policyNetCNN(nn.Module):
    def __init__(self, board_size = 8, input_channels = 2, num_filters=256):
        """
        Inspired by DeepMind's AlphaGo Zero architecture.
        board_size: Size of the Go board (e.g., 19 for 19x19).
        input_channels: Number of input feature planes (e.g., history of previous moves, legal moves, captured stones, 17 in DeepMind's AlphaGo.)
                        In our's, at least 1 for current observation, 1 for legal moves, 1 for current mover. N for history of previous moves. Overall, N+3.
        num_filters: Number of convolutional filters per layer (256 in AlphaGo).
        """
        super(policyNetCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=3, padding=1)
        
        # Residual Blocks (AlphaGo Zero uses 19 residual blocks)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_filters) for _ in range(10)  # Can be adjusted
        ])

        # Final policy head
        self.policy_conv = nn.Conv2d(in_channels=num_filters, out_channels=2, kernel_size=1)  # Reduce channels
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)  # Output probabilities

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)  # Pass through residual blocks
        x = F.relu(self.policy_conv(x))  # Reduce channels
        x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layer
        x = self.policy_fc(x)
        # x = F.log_softmax(x, dim=1)  # Log probability output (use CrossEntropyLoss)
        return x

    def predict(self, x):
        """
        Predict the next move given the current state.
        x: 4D tensor of shape (batch_size, input_channels, board_size, board_size).
        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            return torch.argmax(x, dim=1)
            

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
