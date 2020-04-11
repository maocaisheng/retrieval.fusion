""" The closer the feature is to the center, the better
"""

import os

import torch
import torch.nn as nn


class CenterWoParamsLoss(nn.Module):
    def __init__(self, center_path='./data'):
        super(CenterWoParamsLoss, self).__init__()

        self.centers = torch.load(
            os.path.join(center_path, 'centers.pth')
        )['centers'].data

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim)
            labels with shape (batch_size)
        """
        batch_size = x.size(0)

        loss = torch.pow(x[0] - self.centers[labels[0]], 2).sum()

        for batch_idx in range(1, batch_size):
            loss += torch.pow(x[batch_idx] - self.centers[labels[batch_idx]], 2).sum()

        loss /= 2.

        return loss / batch_size
