# Dense prediction Paper with Code
- [Classical segmentation methods](#Classical_Segmentation)
- [Classical detection method](#Classical_detection)
- [Backbone](#Backbone)
- [Open vocabulary segmentation](#open_segmentation)
- [Open vocabulary detection](#open_detection)
- [Dataset](#Dataset)
- [Other Technologies](#Other)

<a name="Classical_Segmentation"></a>
# Classical segmentation method（Supervised）
[2015 CVPR]**FCN: Fully Convolutional Networks for Semantic Segmentation**
- Paper: https://arxiv.org/abs/1411.4038
- Code: https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn

[2016 MICCAI]**UNet:Convolutional Networks for Biomedical Image Segmentation**
- Paper: https://arxiv.org/pdf/1505.04597

[2017 ArXiv]**DeepLabV3：Rethinking atrous convolution for semantic image segmentation**
- Paper: https://arxiv.org/pdf/1706.05587
- Contribution：Atrous Convolution(Dilated Convolution)

[2018 CVPR]**DeepLabV3+ ：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**
[Paper](https://arxiv.org/pdf/1802.02611)
- Contribution：The convolution layer and pooling layer is replaced by a depth-separable convolution

[2019 CVPR]**Semantic FPN ：Panoptic Feature Pyramid Networks**
- Paper: https://arxiv.org/pdf/1901.02446
- Contribution：Like the title.

[2021 CVPR]**SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**
- Paper: https://arxiv.org/pdf/2012.15840
- Code: https://github.com/fudan-zvg/SETR
- Contribution：ViT Encoder + 3 types of CNN decoders

[2021 ICCV]**Segmenter: Transformer for Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2105.05633
- Code: https://github.com/rstrudel/segmenter
- Contribution：Supervised ViT Segmentation，ViT Enconder + ViT Decoder

[2021 NeurIPS]**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**
- Paper: https://arxiv.org/pdf/2105.15203
- Code: https://github.com/NVlabs/SegFormer
- Contribution：positional-encoding-free, Hierarchical Transformer Encoder, All-MLP decoder

[2021 CVPR]**MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2107.06278
- Code: https://github.com/facebookresearch/MaskFormer
- Contribution：Consider the semantic segmentation task as Mask Classification(predicts a set of binary masks and predicts masks class) = Instance Segmentation + Instance classification; Architecture: Encoder + transformer decoder and pixel decoder

[2022 CVPR]**Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation**
- Paper: https://arxiv.org/pdf/2112.01527
- Code: https://github.com/facebookresearch/Mask2Former
- Contribution：Pixel Decoder multi-scale feature feed to transformer decoder layer; mask transformer decoder; Architecture: Encoder + mask transformer decoder and pixel decoder

<a name="Classical_detection"></a>
# Classical detection method
[2015 NeurIPS]**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**
- Paper: https://arxiv.org/pdf/1506.01497
- Contribution：End-to-End objection detection; Region Proposal Network(RPN)

[2020 ECCV]**DETR: End-to-End Object Detection with Transformers**
- Paper: https://arxiv.org/pdf/2005.12872
- Code: https://github.com/facebookresearch/detr
- Contribution：Like the title;  removing non-maximum suppression; bipartite matching.

[2021 ICLR]**Deformable DETR: Deformable Transformers for End-to-End Object Detection**
- Paper: https://arxiv.org/pdf/2010.04159
- Code: https://github.com/fundamentalvision/Deformable-DETR
- Contribution：Multi-scale feature(Resnet 3block output reshape + concat  + GroupNorm); Attention map by linear layer instead of traditional q*k, Attention map is also smaller because the query only corresponds to a few keys; by linear layer compute dynamic offset(Because linear-layer-weight can be learned), then get sampling value.

[2023 ICLR]**DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection**
- Paper: https://arxiv.org/pdf/2203.03605
- Code: https://github.com/IDEA-Research/DINO
- Contribution：

<a name="Backbone"></a>
# Backbone
[2021 ICLR]**ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
- Paper: https://arxiv.org/pdf/2010.11929
- Code: https://github.com/google-research/vision_transformer

[]**DeiT: Training data-efficient image transformers & distillation through attention**
- Paper: https://arxiv.org/pdf/2012.12877
- Contribution：Distillation Token；Teacher Model:CNN, Student model: ViT

[2021 ICCV]**Swin Transfomer: Hierarchical Vision Transformer using Shifted Windows**
- Paper: https://arxiv.org/pdf/2103.14030
- Code: https://github.com/microsoft/Swin-Transformer
- Contribution：Sliding Window Attention + Patch Merging

[2021 NeurIPS]**Twins: Revisiting the Design of Spatial Attention in Vision Transformers**
- Paper: https://arxiv.org/pdf/2104.13840
- Code: https://github.com/Meituan-AutoML/Twins
- Contribution：Spatially Separable Self-Attention(SSSA)(Similar to separable convolution), Conditional Positional Encodings(CPE) in every transformer block

[2022 ICLR]**BEiT: BERT Pre-Training of Image Transformers**
- Paper: https://arxiv.org/pdf/2106.08254
- Code: https://github.com/microsoft/unilm/tree/master/beit
- Contribution：Encoder：input mask image patches, output mask visual tokens, by dVAE；Decoder：input all visual tokens, output reconstruct image; loss only compute mask part by Cross-Entropy.

[2022 CVPR]**MAE: Masked Autoencoders Are Scalable Vision Learners**
- Paper: https://arxiv.org/pdf/2111.06377
- Code: https://github.com/facebookresearch/mae
- Contribution：Encoder：input unmask image patches, output unmask Embedding, by AutoEncoder；Decoder：input all Embedding(mask Embedding come from random Init), output reconstruct image; loss compute mask part by MSE(Mean Squared Error).

[2022 CVPR]**PoolFormer:MetaFormer is Actually What You Need for Vision**
- Paper: https://arxiv.org/pdf/2111.11418
- Code: https://github.com/sail-sg/poolformer
- Contribution：The success of ViT is not entirely due to attention mechanism, should be attributed to Transformer architecture. Replace the attention layer with a simple pooling layerin the ViT with a simple pooling layer.

[2022 NeurIPS]**SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2209.08575
- Code: https://github.com/Visual-Attention-Network/SegNeXt
- Contribution：Attention by convolution, rather than transformer.

[2023 ICCV]**OpenSeeD: A simple framework for open-vocabulary segmentation and detection**
- Paper: https://arxiv.org/pdf/2303.08131
- Code: https://github.com/IDEA-Research/OpenSeeD
- Contribution：Combined object detection data and panoramic segmentation data training, Before SAM.

[2023 Arxiv]**SAM: Segment Anything**
- Paper: https://arxiv.org/pdf/2304.02643
- Code: https://github.com/facebookresearch/segment-anything
- Demo: https://segment-anything.com/demo
- Contribution：Image Encoder: MAE; Prompt Encoder: text(CLIP), points and boxes(positional encodings), mask(convolutions and summed element-wise); Mask Decoder: Transformer block; Prompt-Encoder and Mask-Decoder are efficient

[2024 Arxiv]**SAM 2: Segment Anything in Images and Videos**
- Paper: https://arxiv.org/pdf/2408.00714
- Code: https://github.com/facebookresearch/sam2
- Demo: https://sam2.metademolab.com/

<a name="open_segmentation"></a>
# Open vocabulary segmentation
[2021 ICML]**CLIP: Learning transferable visual models from natural language supervision**
- Paper: https://arxiv.org/pdf/2103.00020
- Code: https://github.com/OpenAI/CLIP

[2024 TPAMI]**Review: Towards Open Vocabulary Learning:A Survey**
- Paper: https://arxiv.org/pdf/2306.15880
- Code: https://github.com/jianzongwu/Awesome-Open-Vocabulary

[2022 ICLR]**Lseg: Language-driven semantic segmentation(Supervised)**
- Paper: https://arxiv.org/pdf/2201.03546
- Code: https://github.com/isl-org/lang-seg
- Contribution：Visual Encoder = DPT(Vit + Decoder), Text Encoder = transformer; use CLIP pretrain parameters

[2022 CVPR]**ZegFormer: Decoupling Zero-Shot Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2112.07910
- Code: https://github.com/dingjiansw101/ZegFormer
- Contribution：A complex architecture = Maskformer (like Maskformer do mask-classification) + CLIP(Like CLIP zero-shot)

[2022 ECCV]**MaskCLIP+: Extract Free Dense Labels from CLIP**
- Paper: https://arxiv.org/pdf/2112.01071
- Code: https://github.com/chongzhou96/MaskCLIP
- Contribution：1.CLIP generates pseudo-labels to guide target-model(deeplabV2) predictions. 2.Over time, CLIP suboptimal target-model, then target model to generate pseudo-labels to self-training. 3.How does CLIP generate pseudo-labels.

[2023 ICML]**MaskCLIP: Open-Vocabulary Universal Image Segmentation with MaskCLIP**
- Paper: https://arxiv.org/pdf/2208.08984
- Code: https://github.com/mlpc-ucsd/MaskCLIP
- Contribution：

[2022 CVPR]**GroupViT: Semantic Segmentation Emerges from Text Supervision(Open-Vocabulary zero-shot)**
- Paper: https://arxiv.org/pdf/2202.11094
- Code: https://github.com/NVlabs/GroupViT
- Contribution：add segment token into transformer layers; use Grouping Block(attention) to reduce tokens num; training: Using global image features and text features compute loss function, inference like CLIP; From the beginning training, Do not use CLIP pre-training parameters.

[2022 ECCV]**OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels**
- Paper: https://arxiv.org/pdf/2112.12143
- Code: N/A
- Contribution：Like segmentor(ICCV'2021) compute Segmentation-loss; Introduce Region-word grounding loss between Word embedding and mask-based pooling image embedding.

[2023 CVPR]**Freeseg: Unified, universal and open-vocabulary image segmentation**
- Paper: https://arxiv.org/pdf/2303.17225
- Code: https://github.com/bytedance/FreeSeg
- Contribution：A unified Open-Vocabulary Segmentation framework, for combine Semantic, Instance and Panoptic Segmentation; Architecture based on Mask2Former, improvement work.

[2023 ICML]**Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation(Open-Vocabulary zero-shot)**
- Paper: https://arxiv.org/pdf/2211.14813
- Code: https://github.com/ArrowLuo/SegCLIP
- Contribution：GroupViT(CVPR, 2022) improvement work. train: Contrastive loss between Image and text(supervise), MAE reconstruction loss(unsupervise), superpixel based KL loss(between super-patch and mapping matrix from attention map)(unsupervise); test: like CLIP.

[2023 CVPR]**X-Decoder: Generalized decoding for pixel, image, and language**
- Paper: https://arxiv.org/pdf/2212.11270
- Code: https://github.com/microsoft/X-Decoder/tree/main
- Contribution：Generalized decoding framework(include pixel-level image segmentation, image-level retrieval and visual-language tasks); Mask2Former improvement work.

[2023 CVPR]**ODISE: Open-vocabulary panoptic segmentation with text-to-image diffusion models**
- Paper: https://arxiv.org/pdf/2303.04803
- Code: https://github.com/NVlabs/ODISE
- Contribution：

[2022 ECCV]**ViL-Seg: Open-world semantic segmentation via contrasting and clustering vision-language embedding**
- Paper: https://arxiv.org/pdf/2207.08455v2
- Code: N/A
- Contribution：

[2023 CVPR]**SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2302.12242
- Code: https://github.com/MendelXu/SAN
- Contribution：

[2024 CVPR]**SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**
- Paper: https://arxiv.org/pdf/2311.15537
- Code: https://github.com/xb534/SED
- Contribution：

[2024 CVPR]**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**
- Paper: https://arxiv.org/pdf/2311.08046
- Code: https://github.com/PKU-YuanGroup/Chat-UniVi
- Contribution：

<a name="open_detection"></a>
# Open vocabulary object detection
[2021 CVPR]**Open-vocabulary object detection using captions**
- Paper: https://arxiv.org/pdf/2011.10678
- Code: https://github.com/alirezazareian/ovr-cnn
- Contribution：

[2022 ICLR]**ViLD: Open-vocabulary object detection via vision and language knowledge distillation**
- Paper: https://arxiv.org/pdf/2104.13921
- Code: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild
- Contribution：text embedding and ViLD region embedding(for Cropped Regions) compute cross-entropy loss; Vision Knowledge Distillation: ViLD region embedding(from Cropped Regions image) and CLIP image embedding(from Cropped Regions image) compute L1-loss,aims to study novel class from CLIP.

[2022 CVPR]**GLIP: Grounded Language-Image Pre-training**
- Paper: https://arxiv.org/pdf/2112.03857
- Code: https://github.com/microsoft/GLIP
- Contribution：Like CLIP do Language-Image Pre-training object detection;Base Swin Transformer ; Difference: 1. Do self-attention between text and image embedding, then calculate the contrast loss. 2.There is more a Localization loss than classification task.

[2022 NeurIPS]**GLIPv2:Unifying Localization and VL Understanding**
- Paper: https://arxiv.org/pdf/2206.05836
- Code: https://github.com/microsoft/GLIP
- Contribution：A unified framework: Localization tasks(Object Detection; Instance Segmentation) and Understanding tasks(VL Grounding;Visual Question Answering;Image Caption)

GroupVit: https://arxiv.org/pdf/2202.11094
BoQ: https://arxiv.org/pdf/2405.07364
Crica-VPR: https://arxiv.org/pdf/2402.19231

<a name="Dataset"></a>
# Dataset:
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

**DDPM: Denoising Diffusion Probabilistic Models(NeurIPS'2020)**
- Paper: https://arxiv.org/pdf/2006.11239
- Code: https://github.com/hojonathanho/diffusion
- Contribution：First Diffusion Model
