# Dense prediction paper with code
- [Visual Place Recognition](#VPR)
- [Token fusion](#Token_fusion)
- [Training Free](#Training_Free)
- [Classical segmentation methods](#Classical_Segmentation)
- [Classical detection method](#Classical_detection)
- [Backbone](#Backbone)
- [Open vocabulary segmentation](#open_segmentation)
- [Open vocabulary detection](#open_detection)
- [Dataset](#Dataset)
- [Other Technologies](#Other)

<a name="VPR"></a>
# Visual Place Recognition
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2024 CVPR   | **CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition** | [[paper]](https://arxiv.org/pdf/2402.19231) [[code]](https://github.com/Lu-Feng/CricaVPR) |
| 2024 CVPR   | **BoQ: A Place is Worth a Bag of Learnable Queries** | [[paper]](https://arxiv.org/pdf/2405.07364) [[code]](https://github.com/amaralibey/Bag-of-Queries) |
| 2024 NIPS   | **SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition** | [[paper]](https://openreview.net/pdf?id=bZpZMdY1sj) [[code]](https://github.com/Lu-Feng/SuperVLAD) |

<a name="Token_fusion"></a>
# Token Mering, Clustering and Pruning
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2022 CVPR   | **GroupViT: Semantic Segmentation Emerges from Text Supervision** | [[paper]](https://arxiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT) |
| 2024 CVPR   | **Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers** | [[paper]](https://arxiv.org/pdf/2403.10030) [[code]](https://github.com/mlvlab/MCTF) |
| 2021 NIPS   | **TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?** | [[paper]](https://arxiv.org/pdf/2106.11297) [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner) <details><summary>[Summary]</summary> TokenLearner. </details> |
| 2023 ICLR   | **GPVIT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation** | [[paper]](https://arxiv.org/pdf/2212.06795) [[code]](https://github.com/ChenhongyiYang/GPViT) |
| 2024 CVPR   | **Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding** | [[paper]](https://arxiv.org/pdf/2311.08046) [[code]](https://github.com/PKU-YuanGroup/Chat-UniVi) <details><summary>[Summary]</summary> 1.non-parameters token fusion by Density-Peaks-Clustering KNN(DPC-KNN) 2.Primary object with numerous visual tokens and background only one visual token </details> |
| 2023 CVPR   | **SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2302.12242) [[code]](https://github.com/MendelXu/SAN) <details><summary>[Summary]</summary> The mask classification is designed as an end-to-end form, the backbone uses CLIP and freezes, and then extracts the features of CLIP and adds them to the Adapter network with additional training (transformer + learnable query +cls token) </details> |
| 2025 CVPR   | **PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models** | [[paper]](https://arxiv.org/pdf/2504.08966) [[code]](https://github.com/orailix/PACT/tree/main) <details><summary>[Summary]</summary> 1.The pruning module eliminates unimportant tokens. 2.DBDPC algorithm clusters the remaining tokens. 3.Tokens that were initially pruned but are sufficiently close to the constructed clusters are reincorporated, ensuring that valuable information from the pruned tokens is recovered. </details> |
| 2024 ICLR   | **LaVIT: Unified language-vision pretraining in LLM with dynamic discrete visual tokenization** | [[paper]](https://arxiv.org/pdf/2309.04669) [[code]](https://github.com/jy0205/LaVIT) <details><summary>[Summary]</summary> LaVIT used the Gumbel-Softmax to train a mask that selects tokens for retention, merging discarded tokens into retained ones via additional attention layers. </details> |
| 2022 CVPR   | **MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2308.03005) [[code]](https://github.com/xulianuwa/MCTformer) <details><summary>[Summary]</summary> MCTformer+. </details> |
| 2023 CVPR   | **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** |[[paper]](https://arxiv.org/pdf/2301.12597) [[code]](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) <details><summary>[Summary]</summary> Qformer fusion image and text gap. </details>  |
| 2023 ICCV   | **Perceptual Grouping in Contrastive Vision-Language Models** | [[paper]](https://arxiv.org/abs/2210.09996) <details><summary>[Summary]</summary> Self-supervised pretraining how to deal with patch token: **1.max pooling(this paper choise).** 2.average pooling. 3.cls token. </details>  |
| 2024 arXiv  | **TokenPacker: Efficient Visual Projector for Multimodal LLM** |[[paper]](https://arxiv.org/abs/2407.02392) [[code]](https://github.com/CircleRadon/TokenPacker) <details><summary>[Summary]</summary> TokenPacker can achieve an average multimodal performance similar to the original LLaVA-1.5 when compressed to 64 (1/9) tokens, and the inference speed of the model has been significantly improved. TokenPacker can further compress to 32 or even less. </details>  |
| 2024 arXiv  | **DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models** |[[paper]](https://arxiv.org/pdf/2405.20985) [[code]](https://github.com/yaolinli/DeCo) <details><summary>[Summary]</summary> DeCo </details>  |
| 2024 CVPR  | **Grounding Everything: Emerging Localization Properties in Vision-Language Transformers** |[[paper]](https://arxiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM) <details><summary>[Summary]</summary> GEM </details>  |


<a name="Training_Free"></a>
# Training Free
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2024 ECCV   | **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** | [[paper]](https://arxiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP) |
| 2024 WACV   | **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP) |
| 2025 AAAI   | **Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP) |


<a name="Classical_Segmentation"></a>
# Classical segmentation method（Supervised）
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2015 CVPR   | **FCN: Fully Convolutional Networks for Semantic Segmentation** | [[paper]](https://arxiv.org/abs/1411.4038) [[code]](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn) |
| 2016 MICCAI | **UNet: Convolutional Networks for Biomedical Image Segmentation** | [[paper]](https://arxiv.org/pdf/1505.04597) |
| 2017 arXiv  | **DeepLabV3: Rethinking atrous convolution for semantic image segmentation** | [[paper]](https://arxiv.org/pdf/1706.05587) <details><summary>[summary]</summary>Atrous Convolution (Dilated Convolution)</details> |
| 2018 CVPR   | **DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation** | [[paper]](https://arxiv.org/pdf/1802.02611) <details><summary>[summary]</summary>The convolution layer and pooling layer is replaced by a depth-separable convolution</details> |
| 2019 CVPR   | **Semantic FPN: Panoptic Feature Pyramid Networks** | [[paper]](https://arxiv.org/pdf/1901.02446) <details><summary>[summary]</summary>Like the title.</details> |
| 2021 CVPR   | **SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers** | [[paper]](https://arxiv.org/pdf/2012.15840) [[code]](https://github.com/fudan-zvg/SETR) <details><summary>[summary]</summary>ViT Encoder + 3 types of CNN decoders</details> |
| 2021 ICCV   | **Segmenter: Transformer for Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2105.05633) [[code]](https://github.com/rstrudel/segmenter) <details><summary>[summary]</summary>Supervised ViT Segmentation, ViT Encoder + ViT Decoder</details> |
| 2021 NIPS   | **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** | [[paper]](https://arxiv.org/pdf/2105.15203) [[code]](https://github.com/NVlabs/SegFormer) <details><summary>[summary]</summary>Positional-encoding-free, Hierarchical Transformer Encoder, All-MLP decoder</details> |
| 2021 CVPR   | **MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2107.06278) [[code]](https://github.com/facebookresearch/MaskFormer) <details><summary>[summary]</summary>Consider the semantic segmentation task as Mask Classification (predicts a set of binary masks and predicts mask class) = Instance Segmentation + Instance Classification; Architecture: Encoder + Transformer Decoder and Pixel Decoder</details> |
| 2022 CVPR   | **Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation** | [[paper]](https://arxiv.org/pdf/2112.01527) [[code]](https://github.com/facebookresearch/Mask2Former) <details><summary>[summary]</summary>Pixel Decoder multi-scale feature feed to Transformer Decoder layer; Mask Transformer Decoder; Architecture: Encoder + Mask Transformer Decoder and Pixel Decoder</details> |
| 2024 CVPR   | **Rein: Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2312.04265) [[code]](https://github.com/w1oves/Rein) <details><summary>[summary]</summary>  1.Domain Generalized Semantic Segmentation 2.Robust parameter-efficiently fine-tuning approach </details> |



<a name="Classical_detection"></a>
# Classical detection method
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2015 NIPS   | **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** | [[paper]](https://arxiv.org/pdf/1506.01497) <details><summary>[Summary]</summary>End-to-End object detection; Region Proposal Network (RPN)</details> |
| 2020 ECCV   | **DETR: End-to-End Object Detection with Transformers** | [[paper]](https://arxiv.org/pdf/2005.12872) [[code]](https://github.com/facebookresearch/detr) <details><summary>[Summary]</summary>Like the title; Removing non-maximum suppression; Bipartite matching.</details> |
| 2021 ICLR   | **Deformable DETR: Deformable Transformers for End-to-End Object Detection** | [[paper]](https://arxiv.org/pdf/2010.04159) [[code]](https://github.com/fundamentalvision/Deformable-DETR) <details><summary>[Summary]</summary>Multi-scale feature (ResNet 3-block output reshape + concat + GroupNorm); Attention map by linear layer instead of traditional q*k; Attention map is also smaller because the query only corresponds to a few keys; By linear layer compute dynamic offset (Because linear-layer-weight can be learned), then get sampling value.</details> |
| 2023 ICLR   | **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** | [[paper]](https://arxiv.org/pdf/2203.03605) [[code]](https://github.com/IDEA-Research/DINO) <details><summary>[Summary]</summary>To be added.</details> |


<a name="Backbone"></a>
# Backbone
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2017 NIPS   | **transfomer: Attention Is All You Need** | [[paper]](https://arxiv.org/pdf/1706.03762) [[code]](https://github.com/tensorflow/tensor2tensor) |
| 2021 ICLR   | **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** | [[paper]](https://arxiv.org/pdf/2010.11929) [[code]](https://github.com/google-research/vision_transformer) |
| 2021 ICML   | **DeiT: Training data-efficient image transformers & distillation through attention** | [[paper]](https://arxiv.org/pdf/2012.12877) <details><summary>[Summary]</summary>Distillation Token; Teacher Model: CNN, Student Model: ViT</details> |
| 2021 ICCV   | **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** | [[paper]](https://arxiv.org/pdf/2103.14030) [[code]](https://github.com/microsoft/Swin-Transformer) <details><summary>[Summary]</summary>Sliding Window Attention + Patch Merging</details> |
| 2021 NIPS   | **Twins: Revisiting the Design of Spatial Attention in Vision Transformers** | [[paper]](https://arxiv.org/pdf/2104.13840) [[code]](https://github.com/Meituan-AutoML/Twins) <details><summary>[Summary]</summary>Spatially Separable Self-Attention (SSSA) (Similar to separable convolution), Conditional Positional Encodings (CPE) in every transformer block</details> |
| 2022 ICLR   | **BEiT: BERT Pre-Training of Image Transformers** | [[paper]](https://arxiv.org/pdf/2106.08254) [[code]](https://github.com/microsoft/unilm/tree/master/beit) <details><summary>[Summary]</summary>Encoder: input masked image patches, output masked visual tokens via dVAE; Decoder: input all visual tokens, output reconstructed image; Loss computed only on masked parts using Cross-Entropy.</details> |
| 2022 CVPR   | **MAE: Masked Autoencoders Are Scalable Vision Learners** | [[paper]](https://arxiv.org/pdf/2111.06377) [[code]](https://github.com/facebookresearch/mae) <details><summary>[Summary]</summary>Encoder: input unmasked image patches, output unmasked embeddings via AutoEncoder; Decoder: input all embeddings (masked embeddings initialized randomly), output reconstructed image; Loss computed on masked parts using MSE (Mean Squared Error).</details> |
| 2022 CVPR   | **PoolFormer: MetaFormer is Actually What You Need for Vision** | [[paper]](https://arxiv.org/pdf/2111.11418) [[code]](https://github.com/sail-sg/poolformer) <details><summary>[Summary]</summary>The success of ViT is not entirely due to the attention mechanism but rather the Transformer architecture. Replaces the attention layer with a simple pooling layer.</details> |
| 2022 NIPS   | **SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2209.08575) [[code]](https://github.com/Visual-Attention-Network/SegNeXt) <details><summary>[Summary]</summary>Attention by convolution instead of transformer.</details> |
| 2023 ICCV   | **OpenSeeD: A simple framework for open-vocabulary segmentation and detection** | [[paper]](https://arxiv.org/pdf/2303.08131) [[code]](https://github.com/IDEA-Research/OpenSeeD) <details><summary>[Summary]</summary>Combines object detection data and panoramic segmentation data for training, preceding SAM.</details> |
| 2023 arXiv  | **SAM: Segment Anything** | [[paper]](https://arxiv.org/pdf/2304.02643) [[code]](https://github.com/facebookresearch/segment-anything) [[demo]](https://segment-anything.com/demo) <details><summary>[Summary]</summary>Image Encoder: MAE; Prompt Encoder: text (CLIP), points and boxes (positional encodings), mask (convolutions and summed element-wise); Mask Decoder: Transformer block; Prompt-Encoder and Mask-Decoder are efficient.</details> |
| 2024 arXiv  | **SAM 2: Segment Anything in Images and Videos** | [[paper]](https://arxiv.org/pdf/2408.00714) [[code]](https://github.com/facebookresearch/sam2) [[demo]](https://sam2.metademolab.com/) |

<a name="open_segmentation"></a>
# Open vocabulary segmentation
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2021 ICML   | **CLIP: Learning transferable visual models from natural language supervision** | [[paper]](https://arxiv.org/pdf/2103.00020) [[code]](https://github.com/OpenAI/CLIP) |
| 2024 TPAMI  | **Review: Towards Open Vocabulary Learning: A Survey** | [[paper]](https://arxiv.org/pdf/2306.15880) [[code]](https://github.com/jianzongwu/Awesome-Open-Vocabulary) |
| 2022 ICLR   | **Lseg: Language-driven semantic segmentation (Supervised)** | [[paper]](https://arxiv.org/pdf/2201.03546) [[code]](https://github.com/isl-org/lang-seg) <details><summary>[Summary]</summary>Visual Encoder = DPT (ViT + Decoder), Text Encoder = Transformer; uses CLIP pre-trained parameters.</details> |
| 2022 CVPR   | **ZegFormer: Decoupling Zero-Shot Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2112.07910) [[code]](https://github.com/dingjiansw101/ZegFormer) <details><summary>[Summary]</summary>Complex architecture = MaskFormer (mask classification) + CLIP (zero-shot learning).</details> |
| 2022 ECCV   | **MaskCLIP+: Extract Free Dense Labels from CLIP** | [[paper]](https://arxiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP) <details><summary>[Summary]</summary>1. CLIP generates pseudo-labels to guide the target model (DeepLabV2). 2. Over time, CLIP → suboptimal target model → target model self-training. 3. How CLIP generates pseudo-labels.</details> |
| 2023 ICML   | **MaskCLIP: Open-Vocabulary Universal Image Segmentation with MaskCLIP** | [[paper]](https://arxiv.org/pdf/2208.08984) [[code]](https://github.com/mlpc-ucsd/MaskCLIP) |
| 2022 CVPR   | **GroupViT: Semantic Segmentation Emerges from Text Supervision (Open-Vocabulary Zero-Shot)** | [[paper]](https://arxiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT) <details><summary>[Summary]</summary>Adds segment token into transformer layers; uses Grouping Block (attention) to reduce token count; trained from scratch, no CLIP pre-training.</details> |
| 2022 ECCV   | **OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels** | [[paper]](https://arxiv.org/pdf/2112.12143) <details><summary>[Summary]</summary>Similar to Segmentor (ICCV'2021); introduces region-word grounding loss between word embeddings and mask-based pooled image embeddings.</details> |
| 2023 CVPR   | **FreeSeg: Unified, Universal, and Open-Vocabulary Image Segmentation** | [[paper]](https://arxiv.org/pdf/2303.17225) [[code]](https://github.com/bytedance/FreeSeg) <details><summary>[Summary]</summary>A unified open-vocabulary segmentation framework combining semantic, instance, and panoptic segmentation; based on Mask2Former with improvements.</details> |
| 2023 ICML   | **SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation (Zero-Shot)** | [[paper]](https://arxiv.org/pdf/2211.14813) [[code]](https://github.com/ArrowLuo/SegCLIP) <details><summary>[Summary]</summary>Improves upon GroupViT (CVPR 2022). Training: contrastive loss (image-text, supervised), MAE reconstruction loss (unsupervised), superpixel-based KL loss. Testing: CLIP-like inference.</details> |
| 2023 CVPR   | **X-Decoder: Generalized Decoding for Pixel, Image, and Language** | [[paper]](https://arxiv.org/pdf/2212.11270) [[code]](https://github.com/microsoft/X-Decoder/tree/main) <details><summary>[Summary]</summary>Generalized decoding framework covering pixel-level image segmentation, image retrieval, and visual-language tasks; improvement over Mask2Former.</details> |
| 2023 CVPR   | **ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models** | [[paper]](https://arxiv.org/pdf/2303.04803) [[code]](https://github.com/NVlabs/ODISE) |
| 2022 ECCV   | **ViL-Seg: Open-World Semantic Segmentation via Contrasting and Clustering Vision-Language Embeddings** | [[paper]](https://arxiv.org/pdf/2207.08455v2) |
| 2023 CVPR   | **SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2302.12242) [[code]](https://github.com/MendelXu/SAN) <details><summary>[Summary]</summary> The mask classification is designed as an end-to-end form, the backbone uses CLIP and freezes, and then extracts the features of CLIP and adds them to the Adapter network with additional training (transformer + learnable query +cls token) </details> |
| 2024 ECCV   | **CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation** | [[paper]](https://arxiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser) |
| 2024 CVPR   | **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation** | [[paper]](https://arxiv.org/pdf/2311.15537) [[code]](https://github.com/xb534/SED) <details><summary>[Summary]</summary> The design of decoder </details> |



<a name="open_detection"></a>
# Open vocabulary object detection
| Year/Source | Title | Links |
|-------------|------------------------------------------------------------|----------------------------------------------------------------|
| 2021 CVPR   | **Open-Vocabulary Object Detection Using Captions** | [[paper]](https://arxiv.org/pdf/2011.10678) [[code]](https://github.com/alirezazareian/ovr-cnn) |
| 2022 ICLR   | **ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation** | [[paper]](https://arxiv.org/pdf/2104.13921) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) <details><summary>[Summary]</summary>Text embedding and ViLD region embedding (for cropped regions) compute cross-entropy loss. Vision Knowledge Distillation: ViLD region embedding (from cropped image regions) and CLIP image embedding compute L1 loss, enabling novel class learning from CLIP.</details> |
| 2022 CVPR   | **GLIP: Grounded Language-Image Pre-training** | [[paper]](https://arxiv.org/pdf/2112.03857) [[code]](https://github.com/microsoft/GLIP) <details><summary>[Summary]</summary>Similar to CLIP's language-image pretraining but for object detection. Uses Swin Transformer as backbone. Differences: (1) Self-attention between text and image embeddings before contrastive loss calculation. (2) Adds a localization loss beyond classification.</details> |
| 2022 NIPS   | **GLIPv2: Unifying Localization and Vision-Language Understanding** | [[paper]](https://arxiv.org/pdf/2206.05836) [[code]](https://github.com/microsoft/GLIP) <details><summary>[Summary]</summary>A unified framework for both localization tasks (object detection, instance segmentation) and understanding tasks (VL grounding, visual question answering, image captioning).</details> |
  

<a name="Dataset"></a>
# Dataset:
- [x][Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
- [x][PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
- [x][ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k)
- [x][Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [x][COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k)
- [x][COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k)
- [x][CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#chase-db1)
- [x][DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#drive)
- [x][HRF](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#hrf)
- [x][STARE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#stare)
- [x][Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#dark-zurich)
- [x][Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#nighttime-driving)
- [x][LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda)
- [x][Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-potsdam)
- [x][Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-vaihingen)
- [x][iSAID](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isaid)
- [x][High quality synthetic face occlusion](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#delving-into-high-quality-synthetic-face-occlusion-segmentation-datasets)
- [x][ImageNetS](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#imagenets)

<a name="Other"></a>
# Other Technologies
**pixel shuffle**
[[paper]](https://arxiv.org/pdf/1609.05158)

[2020 NIPS] **DDPM: Denoising Diffusion Probabilistic Models**
[[paper]](https://arxiv.org/pdf/2006.11239)
[[code]](https://github.com/hojonathanho/diffusion)
- summary：First Diffusion Model
