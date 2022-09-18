import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gmflow.geometry import forward_backward_consistency_check


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10):
        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        metric0 = self.metric_net(torch.cat((img0, fwd_occ.unsqueeze(1)), 1))
        metric1 = self.metric_net(torch.cat((img1, bwd_occ.unsqueeze(1)), 1))

        return metric0, metric1
