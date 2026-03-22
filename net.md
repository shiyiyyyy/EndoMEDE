DepthAnythingV2(
  (pretrained): DinoVisionTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
      (norm): Identity()
    )
    (blocks): ModuleList(
      (0-11): 12 x NestedTensorBlock(
        (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (attn): MemEffAttention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
    (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
    (head): Identity()
  )
  (depth_head): DPTHead(
    (projects): ModuleList(
      (0): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      (3): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
    )
    (resize_layers): ModuleList(
      (0): ConvTranspose2d(48, 48, kernel_size=(4, 4), stride=(4, 4))
      (1): ConvTranspose2d(96, 96, kernel_size=(2, 2), stride=(2, 2))
      (2): Identity()
      (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (scratch): Module(
      (layer1_rn): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer2_rn): Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer3_rn): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer4_rn): Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (refinenet1): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet2): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet3): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet4): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (output_conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (output_conv2): Sequential(
        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        (3): Sigmoid()
      )
    )
  )
)

<bound method DepthAnythingV2.infer_image of DepthAnythingV2(
  (pretrained): DinoVisionTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
      (norm): Identity()
    )
    (blocks): ModuleList(
      (0-11): 12 x NestedTensorBlock(
        (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (attn): MemEffAttention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
    (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
    (head): Identity()
  )
  (depth_head): DPTHead(
    (projects): ModuleList(
      (0): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      (3): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
    )
    (resize_layers): ModuleList(
      (0): ConvTranspose2d(48, 48, kernel_size=(4, 4), stride=(4, 4))
      (1): ConvTranspose2d(96, 96, kernel_size=(2, 2), stride=(2, 2))
      (2): Identity()
      (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (scratch): Module(
      (layer1_rn): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer2_rn): Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer3_rn): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer4_rn): Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (refinenet1): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet2): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet3): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet4): FeatureFusionBlock(
        (out_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU()
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (output_conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (output_conv2): Sequential(
        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        (3): Sigmoid()
      )
    )
  )
)

(yzp) conghu@doraemon20:~/code/depth_any/Depth-Anything-V2/metric_depth$ python networkprt.py 
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
DepthAnythingV2                                    [1, 518, 518]             --
├─DinoVisionTransformer: 1-1                       --                        526,848
│    └─PatchEmbed: 2-1                             [1, 1369, 384]            --
│    │    └─Conv2d: 3-1                            [1, 384, 37, 37]          226,176
│    │    └─Identity: 3-2                          [1, 1369, 384]            --
│    └─ModuleList: 2-2                             --                        --
│    │    └─NestedTensorBlock: 3-3                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-4                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-5                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-6                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-7                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-8                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-9                 [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-10                [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-11                [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-12                [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-13                [1, 1370, 384]            1,775,232
│    │    └─NestedTensorBlock: 3-14                [1, 1370, 384]            1,775,232
│    └─LayerNorm: 2-3                              [1, 1370, 384]            768
│    └─LayerNorm: 2-4                              [1, 1370, 384]            (recursive)
│    └─LayerNorm: 2-5                              [1, 1370, 384]            (recursive)
│    └─LayerNorm: 2-6                              [1, 1370, 384]            (recursive)
├─DPTHead: 1-2                                     [1, 1, 518, 518]          --
│    └─ModuleList: 2-13                            --                        (recursive)
│    │    └─Conv2d: 3-15                           [1, 48, 37, 37]           18,480
│    └─ModuleList: 2-14                            --                        (recursive)
│    │    └─ConvTranspose2d: 3-16                  [1, 48, 148, 148]         36,912
│    └─ModuleList: 2-13                            --                        (recursive)
│    │    └─Conv2d: 3-17                           [1, 96, 37, 37]           36,960
│    └─ModuleList: 2-14                            --                        (recursive)
│    │    └─ConvTranspose2d: 3-18                  [1, 96, 74, 74]           36,960
│    └─ModuleList: 2-13                            --                        (recursive)
│    │    └─Conv2d: 3-19                           [1, 192, 37, 37]          73,920
│    └─ModuleList: 2-14                            --                        (recursive)
│    │    └─Identity: 3-20                         [1, 192, 37, 37]          --
│    └─ModuleList: 2-13                            --                        (recursive)
│    │    └─Conv2d: 3-21                           [1, 384, 37, 37]          147,840
│    └─ModuleList: 2-14                            --                        (recursive)
│    │    └─Conv2d: 3-22                           [1, 384, 19, 19]          1,327,488
│    └─Module: 2-15                                --                        --
│    │    └─Conv2d: 3-23                           [1, 64, 148, 148]         27,648
│    │    └─Conv2d: 3-24                           [1, 64, 74, 74]           55,296
│    │    └─Conv2d: 3-25                           [1, 64, 37, 37]           110,592
│    │    └─Conv2d: 3-26                           [1, 64, 19, 19]           221,184
│    │    └─FeatureFusionBlock: 3-27               [1, 64, 37, 37]           151,872
│    │    └─FeatureFusionBlock: 3-28               [1, 64, 74, 74]           151,872
│    │    └─FeatureFusionBlock: 3-29               [1, 64, 148, 148]         151,872
│    │    └─FeatureFusionBlock: 3-30               [1, 64, 296, 296]         151,872
│    │    └─Conv2d: 3-31                           [1, 32, 296, 296]         18,464
│    │    └─Sequential: 3-32                       [1, 1, 518, 518]          9,281
====================================================================================================
Total params: 25,006,657
Trainable params: 25,006,657
Non-trainable params: 0
Total mult-adds (G): 12.21
====================================================================================================
Input size (MB): 3.22
Forward/backward pass size (MB): 926.20
Params size (MB): 96.74
Estimated Total Size (MB): 1026.15
====================================================================================================

