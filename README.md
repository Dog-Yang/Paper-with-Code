# paper-list
- [Classical segmentation methods](#Classical)
- [ViT sementation](#ViT)
- [Open vocabulary segmentation](#Open)
- [Other Technologies](#Other)

<a name="Classical"></a>
# Classical segmentation method
**FCN: Fully convolutional networks (CVPR'2015)**
- Paper: https://arxiv.org/abs/1411.4038
- Code: https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn

**UNet (MICCAI'2016)**
- Paper: https://arxiv.org/pdf/1505.04597

**DeepLabV3：Rethinking atrous convolution for semantic image segmentation (ArXiv'2017)**
- Paper: https://arxiv.org/pdf/1706.05587
- Contribution：Atrous Convolution(Dilated Convolution)

**DeepLabV3+ ：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (CVPR'2018)**
- Paper: https://arxiv.org/pdf/1802.02611
- Contribution：The convolution layer and pooling layer is replaced by a depth-separable convolution

**Semantic FPN ：Panoptic Feature Pyramid Networks(CVPR'2019)**
- Paper: https://arxiv.org/pdf/1901.02446
- Contribution：Panoptic Feature Pyramid Networks


**BiSeNetV2 (IJCV'2021)**
- Paper: 
- Code: 
- Contribution：


**STDC (CVPR'2021)**
- Paper: 
- Code: 
- Contribution：


**SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (CVPR'2021)**
- Paper: https://arxiv.org/pdf/2012.15840
- Code: https://github.com/fudan-zvg/SETR
- Contribution：ViT Encoder + 3 types of CNN decoders


**DPT (ArXiv'2021)**
- Paper: 
- Code: 
- Contribution：

**Segmenter: Segmenter: Transformer for Semantic Segmentation (ICCV'2021)**
- Paper: https://arxiv.org/pdf/2105.05633
- Code: https://github.com/rstrudel/segmenter
- Contribution：Supervised ViT Segmentation，ViT Enconder + ViT Decoder

**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (NeurIPS'2021)**
- Paper: https://arxiv.org/pdf/2105.15203
- Code: https://github.com/NVlabs/SegFormer
- Contribution：positional-encoding-free, Hierarchical Transformer Encoder, All-MLP decoder

**K-Net (NeurIPS'2021)**
- Paper: 
- Code: 
- Contribution：

**DEST (CVPRW'2022)**
- Paper: 
- Code: 
- Contribution：

<a name="ViT"></a>
# ViT sementation
**DeiT: Training data-efficient image transformers & distillation through attention**
- Paper: https://arxiv.org/pdf/2012.12877
- Contribution：Distillation Token；Teacher Model:CNN, Student model: ViT

**Swin Transfomer: Hierarchical Vision Transformer using Shifted Windows**
- Paper: https://arxiv.org/pdf/2103.14030
- Code: https://github.com/microsoft/Swin-Transformer
- Contribution：Sliding Window Attention + Patch Merging

<a name="Open"></a>
# Open vocabulary segmentation


<a name="Other"></a>
# Other Technologies
**pixel shuffle**
- Paper: https://arxiv.org/pdf/1609.05158
