import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.gmflow.gmflow import GMFlow
from model.MetricNet import MetricNet
from model.FusionNet import AnimeInterp

device = torch.device("cuda")
    
class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.fusionnet = AnimeInterp()
        self.version = 3.9

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
            self.metricnet.load_state_dict(convert(torch.load('{}/metric.pkl'.format(path))))
            self.fusionnet.load_state_dict(convert(torch.load('{}/fusionnet.pkl'.format(path))))

    def reuse(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)
        if scale != 1.0:
            imgf0 = F.interpolate(imgs[:, :3], scale_factor = scale * 0.5, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(imgs[:, 3:6], scale_factor = scale * 0.5, mode="bilinear", align_corners=False)
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
            flow10 = F.interpolate(flow10, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)
        return flow01, flow10, metric0, metric1

    def inference(self, img0, img1, flow01, flow10, metric0, metric1, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)
        F0t = timestep * flow01
        F1t = (1 - timestep) * flow10
        out = self.fusionnet(imgs, img0, img1, F0t, F1t, metric0, metric1)
        return out
