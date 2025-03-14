# TODO

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
    
class AlphaZeroNet(nn.Module):
    def __init__(self, board_size = 8, input_channels = 5, num_filters=256):
        """
        Inspired by DeepMind's AlphaGo Zero architecture.
        board_size: Size of the Go board (e.g., 19 for 19x19).
        input_channels: Number of input feature planes (e.g., history of previous moves, legal moves, captured stones, 17 in DeepMind's AlphaGo.)
                        In our's, at least 1 for current observation, 1 for legal moves, 1 for current mover. N for history of previous moves. Overall, N+3.
        num_filters: Number of convolutional filters per layer (256 in AlphaGo).
        """
        super(AlphaZeroNet, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=3, padding=1)
        
        # Residual Blocks (AlphaGo Zero uses 19 residual blocks)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_filters) for _ in range(4)  # Can be adjusted
        ])

        # Final policy head
        self.policy_conv = nn.Conv2d(in_channels=num_filters, out_channels=2, kernel_size=1)  # Reduce channels
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)  # Output probabilities

        ## --- Value Head ---
        self.value_conv = nn.Conv2d(in_channels=num_filters, out_channels=1, kernel_size=1)  # Reduce to 1 channel
        self.value_fc1 = nn.Linear(board_size * board_size, 64)  # Hidden layer
        self.value_fc2 = nn.Linear(64, 1) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)  # Pass through residual blocks
        ## policy output
        p = F.relu(self.policy_conv(x))  # Reduce channels
        p = torch.flatten(p, start_dim=1)  # Flatten before fully connected layer
        logits = self.policy_fc(p)

        ## value output
        v = F.relu(self.value_conv(x))
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return value, logits


  #   State-Value function
    def inference(self, image):
      return self.forward(image)


    def get_weights(self):
        # Returns the weights of this network.
        return []
    
# class AlphaZeroNet(nn.Module):
#     """policy-value network module"""
#     def __init__(self, board_size = 8):
#         super(AlphaZeroNet, self).__init__()

#         self.board_width = board_size
#         self.board_height = board_size
#         # common layers
#         self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         # action policy layers
#         self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
#         self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
#                                  self.board_width*self.board_height)
#         # state value layers
#         self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
#         self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
#         self.val_fc2 = nn.Linear(64, 1)

#     def forward(self, state_input):
#         # common layers
#         x = F.relu(self.conv1(state_input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         # action policy layers
#         x_act = F.relu(self.act_conv1(x))
#         x_act = x_act.view(-1, 4*self.board_width*self.board_height)
#         x_act = F.log_softmax(self.act_fc1(x_act))
#         # state value layers
#         x_val = F.relu(self.val_conv1(x))
#         x_val = x_val.view(-1, 2*self.board_width*self.board_height)
#         x_val = F.relu(self.val_fc1(x_val))
#         x_val = F.tanh(self.val_fc2(x_val))
#         return x_val, x_act