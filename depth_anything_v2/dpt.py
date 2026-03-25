import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import os
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dinov2_layers.trans import DWT, IWT
from peft import LoraConfig, get_peft_model, TaskType

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

# Half Wavelet Dual Attention Block (HWB)

class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class HWB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
        super(HWB, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        modules_body = \
            [
                conv(n_feat*2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat*2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat*2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    针对内窥镜场景使用小 dilation rate（1/2/4/8），避免大 rate 在小特征图上产生网格效应
    """
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        # 融合：1×1 + 4个空洞卷积 + global pool = 6路
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        res = [self.conv1x1(x)]
        for conv in self.atrous_convs:
            res.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=True)
        res.append(gp)
        return self.project(torch.cat(res, dim=1))


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        hwb_channels=[96, 192],
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        # ASPP：放在 refinenet 融合后，多尺度感受野覆盖内窥镜近远景深度变化
        self.aspp = ASPP(features, features, rates=[1, 2, 4, 8])
        # 中间阶段 HWB：作用于 path_2（特征最丰富，分辨率适中），通道数为 features
        self.hwlayer_mid = HWB(n_feat=features, o_feat=features, kernel_size=3, reduction=8, bias=False, act=nn.GELU())
        # 末端 HWB：作用于 output_conv1 之后（64通道），上采样前做最终细节增强
        self.hwlayer2 = HWB(n_feat=64, o_feat=64, kernel_size=3, reduction=8, bias=False, act=nn.GELU())
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0] #(16,1369,768)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))  #(16,768,37,37)

            x = self.projects[i](x) #(16,96,37,37) #(16,192,37,37)  (16,384,37,37) (16,768,37,37)

            x = self.resize_layers[i](x) #(16,96,148,148） #(16,192,74,74)  (16,384,37,37) (16,768,19,19)
            # if i <=1:
            #     x = self.hwlayer1[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out



        layer_1_rn = self.scratch.layer1_rn(layer_1) #(16,128,148,148) 16,128,37,37
        layer_2_rn = self.scratch.layer2_rn(layer_2) #(16,128,74,74) 16,128,37,37
        layer_3_rn = self.scratch.layer3_rn(layer_3) #(16,128,37,37) 16,128,37,37
        layer_4_rn = self.scratch.layer4_rn(layer_4) #(16,128,19,19) 16,128,37,37

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])    #(16,128,37,37) 16,128,37,37
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:]) #(16,128,74,74) 16,128,37,37
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:]) #(16,128,148,148) 16,128,37,37
        path_2 = self.hwlayer_mid(path_2)  # 中间 HWB：多尺度小波增强，捕获软组织纹理细节
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) #(16,128,296,296) 16,128,74,74
        path_1 = self.aspp(path_1)  # 多尺度感受野聚合，覆盖内窥镜近远景深度变化

        out = self.scratch.output_conv1(path_1) #(16,64,296,296) 16 64,74,74
        out = self.hwlayer2(out)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True) #(16,64,518,518) 16,64,518,518
        out = self.scratch.output_conv2(out) #(16,1,518,518) 16,1,518,518

        return out




class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        lora_type="None",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.max_depth = max_depth

        self.encoder = encoder
        # self.pretrained = DINOv2(model_name=encoder)
        self.pretrained = DINOv2(model_name=encoder)

        #========== PEFT实现：替换FC+QV层 ==========
        if lora_type != "None":

            # load pretrained
            print("load_pretrain_model")
            ckpt = torch.load("checkpoints/depth_anything_v2_metric_hypersim_vitb.pth", map_location='cpu')
            # ckpt = torch.load("./run/seed/lora_free_qkv_fc/40_epotwo_02/latest_model.pth", map_location='cpu')
            # 只保留权重里含'pretrained'的参数，且去掉前缀
            pretrained_weight = {}
            for k, v in ckpt.items():
                if 'pretrained' in k:
                    pretrained_weight[k.replace('pretrained.', '')] = v  # 去掉前缀适配模型
            self.pretrained.load_state_dict(pretrained_weight, strict=False)


            # 定义DoRA配置：
            # - 目标层：仅 attention（qkv + proj），专注几何感知
            # - use_dora=True：幅度+方向分离更新，domain gap 大时比标准 LoRA 更稳定
            print("load_dora")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["attn.qkv", "attn.proj"],
                use_dora=True
            )
            # 应用 PEFT 到 pretrained 模型
            self.pretrained = get_peft_model(self.pretrained, peft_config)

            # 手动冻结浅层（block 0-5）的 LoRA 参数，只训练深层（6-11）
            # 浅层学的是低级特征，与自然图像差异小，无需适配
            for name, param in self.pretrained.named_parameters():
                if 'lora_' in name or 'lora_magnitude' in name:
                    # 提取 block 编号
                    parts = name.split('.')
                    for j, part in enumerate(parts):
                        if part == 'blocks' and j + 1 < len(parts):
                            try:
                                block_idx = int(parts[j + 1])
                                if block_idx < 6:
                                    param.requires_grad = False
                            except ValueError:
                                pass
                            break

        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)

        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
