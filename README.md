# paper-list
- [Classical segmentation methods](#Classical)
- [Backbone](#Backbone)
- [Open vocabulary segmentation](#Open)
- [Dataset](#Dataset)
- [Other Technologies](#Other)

<a name="Classical"></a>
# Classical segmentation method（Supervised）
**FCN: Fully convolutional networks (CVPR'2015)**
- Paper: https://arxiv.org/abs/1411.4038
- Code: https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn

**UNet:Convolutional Networks for Biomedical Image Segmentation (MICCAI'2016)**
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

**SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (CVPR'2021)**
- Paper: https://arxiv.org/pdf/2012.15840
- Code: https://github.com/fudan-zvg/SETR
- Contribution：ViT Encoder + 3 types of CNN decoders

**Segmenter: Segmenter: Transformer for Semantic Segmentation (ICCV'2021)**
- Paper: https://arxiv.org/pdf/2105.05633
- Code: https://github.com/rstrudel/segmenter
- Contribution：Supervised ViT Segmentation，ViT Enconder + ViT Decoder

**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (NeurIPS'2021)**
- Paper: https://arxiv.org/pdf/2105.15203
- Code: https://github.com/NVlabs/SegFormer
- Contribution：positional-encoding-free, Hierarchical Transformer Encoder, All-MLP decoder

**MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation (CVPR'2021)**
- Paper: https://arxiv.org/pdf/2107.06278
- Code: https://github.com/facebookresearch/MaskFormer
- Contribution：Consider the semantic segmentation task as Mask Classification(predicts a set of binary masks and predicts masks class); Architecture: Encoder + transformer decoder and pixel decoder

**Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR'2022)**
- Paper: https://arxiv.org/pdf/2112.01527
- Code: https://github.com/facebookresearch/Mask2Former
- Contribution：Pixel Decoder multi-scale feature feed to transformer decoder layer; mask transformer decoder; Architecture: Encoder + mask transformer decoder and pixel decoder

**VPD: Unleashing Text-to-Image Diffusion Models for Visual Perception (ICCV'2023)**
- Paper: https://arxiv.org/pdf/2303.02153
- Code: https://github.com/wl-zhao/VPD
- Contribution：

<a name="Backbone"></a>
# Backbone
**ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR'2021)**
- Paper: https://arxiv.org/pdf/2010.11929
- Code: https://github.com/google-research/vision_transformer

**DeiT: Training data-efficient image transformers & distillation through attention**
- Paper: https://arxiv.org/pdf/2012.12877
- Contribution：Distillation Token；Teacher Model:CNN, Student model: ViT

**Swin Transfomer: Hierarchical Vision Transformer using Shifted Windows (ICCV'2021)**
- Paper: https://arxiv.org/pdf/2103.14030
- Code: https://github.com/microsoft/Swin-Transformer
- Contribution：Sliding Window Attention + Patch Merging

**Twins: Revisiting the Design of Spatial Attention in Vision Transformers [NeurIPS 2021]**
- Paper: https://arxiv.org/pdf/2104.13840
- Code: https://github.com/Meituan-AutoML/Twins
- Contribution：Spatially Separable Self-Attention(SSSA)(Similar to separable convolution), Conditional Positional Encodings(CPE) in every transformer block

**BEiT: BERT Pre-Training of Image Transformers (ICLR'2022)**
- Paper: https://arxiv.org/pdf/2106.08254
- Code: https://github.com/microsoft/unilm/tree/master/beit
- Contribution：Encoder：input mask image patches, output mask visual tokens, by dVAE；Decoder：input all visual tokens, output reconstruct image; loss only compute mask part by Cross-Entropy.

**MAE: Masked Autoencoders Are Scalable Vision Learners (CVPR'2022)**
- Paper: https://arxiv.org/pdf/2111.06377
- Code: https://github.com/facebookresearch/mae
- Contribution：Encoder：input unmask image patches, output unmask Embedding, by AutoEncoder；Decoder：input all Embedding(mask Embedding come from random Init), output reconstruct image; loss compute mask part by MSE(Mean Squared Error).

**PoolFormer:MetaFormer is Actually What You Need for Vision (CVPR'2022)**
- Paper: https://arxiv.org/pdf/2111.11418
- Code: https://github.com/sail-sg/poolformer
- Contribution：The success of ViT is not entirely due to attention mechanism, should be attributed to Transformer architecture. Replace the attention layer with a simple pooling layerin the ViT with a simple pooling layer.

**SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS'2022)**
- Paper: https://arxiv.org/pdf/2209.08575
- Code: https://github.com/Visual-Attention-Network/SegNeXt
- Contribution：Attention by convolution, rather than transformer.

<a name="Open"></a>
# Open vocabulary segmentation
**Review: Towards Open Vocabulary Learning:A Survey (TPAMI'2024)**
- Paper: https://arxiv.org/pdf/2306.15880
- Code: https://github.com/jianzongwu/Awesome-Open-Vocabulary

**Fisrt Open vocabulary object detection**
Open-vocabulary object detection using captions(CVPR, 2021)
- Paper: https://arxiv.org/pdf/2011.10678
- Code: https://github.com/alirezazareian/ovr-cnn

Open-vocabulary object detection via vision and language knowledge distillation(ICLR, 2022)
- Paper: https://arxiv.org/pdf/2104.13921
- Code: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild

**Fisrt Open vocabulary segmentation**
Scaling Open-Vocabulary Image Segmentation with Image-Level Labels (ECCV, 2022)
- Paper: https://arxiv.org/pdf/2112.12143
- Code: N/A

Language-driven semantic segmentation (ICLR, 2022)
- Paper: https://arxiv.org/pdf/2201.03546
- Code: https://github.com/isl-org/lang-seg

**SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation (CVPR'2023)**
- Paper: https://arxiv.org/pdf/2302.12242
- Code: https://github.com/MendelXu/SAN
- Contribution：

<a name="Dataset"></a>
# Datasets:
- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isaid)
- [x] [High quality synthetic face occlusion](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#delving-into-high-quality-synthetic-face-occlusion-segmentation-datasets)
- [x] [ImageNetS](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#imagenets)

<a name="Other"></a>
# Other Technologies
**pixel shuffle**
- Paper: https://arxiv.org/pdf/1609.05158
