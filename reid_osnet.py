from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    # OSNet 的基础卷积模块：卷积 + 归一化 + ReLU。
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, use_in=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if use_in else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    # 1x1 卷积常用来调整通道数，计算量小。
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    # 线性 1x1 卷积，不带 ReLU，常用于残差分支或输出映射。
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    # 轻量 3x3 卷积，参数量比普通卷积更小，适合做人 ReID 的轻量网络。
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelGate(nn.Module):
    # 通道注意力：让网络自动学习“哪些通道更重要”。
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        gates = self.global_avgpool(x)
        gates = self.fc1(gates)
        gates = self.relu(gates)
        gates = self.fc2(gates)
        gates = self.gate(gates)
        return x * gates


class OSBlock(nn.Module):
    # OSNet 的核心模块。
    # 它会并行提取不同尺度的特征，再通过门控机制融合，兼顾局部纹理和整体外观。
    def __init__(self, in_channels, out_channels, use_in=False, bottleneck_reduction=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True) if use_in else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1))
        x2 = x2 + self.gate(self.conv2b(x1))
        x2 = x2 + self.gate(self.conv2c(x1))
        x2 = x2 + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = x3 + identity
        if self.inst_norm is not None:
            out = self.inst_norm(out)
        return F.relu(out)


class OSNet(nn.Module):
    # 这里实现的是一个最小可用的 OSNet 主干网络。
    # 我们只拿它做特征提取，不关心最终分类类别。
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, use_in=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, use_in=use_in)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, use_in=use_in)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, use_in=False):
        layers = [block(in_channels, out_channels, use_in=use_in)]
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, use_in=use_in))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        # 经过主干网络后得到高层语义特征图。
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def forward(self, x):
        # 输出的是特征向量 embedding，而不是分类结果。
        x = self.featuremaps(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        return x


def osnet_x0_5(num_classes=1000):
    # x0.5 是一个比较轻量、速度和效果比较平衡的 OSNet 版本。
    return OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
    )


class PersonReIDEncoder:
    # 对外提供一个简单接口：输入一张人物裁剪图，输出一个归一化的 512 维特征。
    def __init__(self, weights_path, device=None, input_size=(128, 256)):
        self.weights_path = Path(weights_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = self._build_model().to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_model(self):
        # 这里加载 OSNet 权重。
        # strict=False 的原因是我们主要用特征提取部分，分类头不一定和当前任务完全一致。
        model = osnet_x0_5(num_classes=4101)
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def preprocess(self, image_bgr):
        # ReID 模型要求固定输入大小和标准化方式，这里把 OpenCV 的 BGR 图像转成模型可用的张量。
        resized = cv2.resize(image_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = (image_rgb - self.mean) / self.std
        tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def encode(self, image_bgr):
        # encode 是最核心的对外方法：
        # 输入人物裁剪图，输出单位长度的 embedding，方便后续直接做余弦相似度比较。
        tensor = self.preprocess(image_bgr)
        embedding = self.model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).detach().cpu().numpy()
