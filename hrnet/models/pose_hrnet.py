import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method):
        super().__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.blocks = blocks
        self.num_inchannels = num_channels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for _ in range(num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i],
                                  kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    ops = []
                    for k in range(i - j):
                        inch = self.num_inchannels[j]
                        outch = self.num_inchannels[j] if k != i - j - 1 else self.num_inchannels[i]
                        ops.append(nn.Sequential(
                            nn.Conv2d(inch, outch, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(outch, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*ops))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        if self.num_branches == 1:
            return x
        x_fuse = []
        for i in range(self.num_branches):
            y = x[i]
            for j in range(self.num_branches):
                if j == i:
                    continue
                if self.fuse_layers[i][j] is not None:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HRNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)  # output: 256

        # Stage 2
        self.transition1, self.stage2_channels = self._make_transition_layer([256], [32, 64])
        self.stage2 = self._make_stage(HighResolutionModule, BasicBlock, [4, 4], [32, 64])

        # Stage 3
        self.transition2, self.stage3_channels = self._make_transition_layer([32, 64], [32, 64, 128])
        self.stage3 = self._make_stage(HighResolutionModule, BasicBlock, [4, 4, 4], [32, 64, 128])

        # Stage 4
        self.transition3, self.stage4_channels = self._make_transition_layer([32, 64, 128], [32, 64, 128, 256])
        self.stage4 = self._make_stage(HighResolutionModule, BasicBlock, [4, 4, 4, 4], [32, 64, 128, 256])

        # Final keypoint head (combine all resolutions)
        last_inp_channels = sum(self.stage4_channels)
        self.final_layer = nn.Conv2d(last_inp_channels, num_keypoints, kernel_size=1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(inplanes, planes, downsample=downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        layers = []
        for i in range(len(num_channels_cur)):
            if i < len(num_channels_pre):
                if num_channels_cur[i] != num_channels_pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    layers.append(None)
            else:
                convs = []
                inch = num_channels_pre[-1]
                for k in range(i + 1 - len(num_channels_pre)):
                    outch = num_channels_cur[i] if k == i - len(num_channels_pre) else inch
                    convs.append(nn.Sequential(
                        nn.Conv2d(inch, outch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outch, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers), num_channels_cur

    def _make_stage(self, module, block, num_blocks, num_channels):
        num_branches = len(num_blocks)
        return module(num_branches, block, num_blocks, num_channels, 'SUM')

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.layer1(x)
        x_list = []

        for i in range(len(self.transition1)):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)

        x_list = self.stage2(x_list)

        y_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                y_list.append(self.transition2[i](x_list[-1]))
            else:
                y_list.append(x_list[i])
        x_list = self.stage3(y_list)

        y_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                y_list.append(self.transition3[i](x_list[-1]))
            else:
                y_list.append(x_list[i])
        x_list = self.stage4(y_list)

        # Upsample all and concatenate
        x0_h, x0_w = x_list[0].shape[2:]
        upsampled = [x_list[0]] + [F.interpolate(x, size=(x0_h, x0_w), mode='bilinear', align_corners=True) for x in x_list[1:]]
        x = torch.cat(upsampled, 1)

        x = self.final_layer(x)
        return x


def get_pose_net(num_keypoints=17):
    return HRNet(num_keypoints=num_keypoints)
