# Content
1. [[Remote Sensing](#Remote_Sensing)]      
2. [[Training Free Segmentation](#Training_Free)]            
3. [[Zero shot Classification/Test Time Adaptation](#Zero_shot_classification)]      
4. [[Optimal Transport](#Optimal_Transport)]     
5. [[CLIP](#CLIP)]           
6. [[Visual Place Recognition](#VPR)]         
7. [[Token Mering, Clustering and Pruning](#Token_fusion)]           
8. [[Backbone](#Backbone)]           
9. [[Weakly Supervised Semantic Segmentation](#WSSS)]           
10. [[Open vocabulary](#open_vocabulary)]                 
11. [[segmentation and detection](#Segmentation_and_Detection)]           
12. [[Other](#Other)]      
-----------------------------------------------------------------------------------------------


<a name="Remote_Sensing"></a> 
## Remote Sensing
1. [2025 arXiv] **DynamicEarth: How Far are We from Open-Vocabulary Change Detection?** [[paper]](https://arXiv.org/abs/2501.12931) [[code]](https://github.com/likyoo/DynamicEarth)
2. [2025 TGRS] **A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation.** [[paper]](https://ieeexplore.ieee.org/document/11063320) [[code]](https://github.com/sstary/SSRS)
3. [2025 ICASSP] **Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification.** [[paper]](https://arXiv.org/abs/2409.00698) [[code]](https://github.com/elkhouryk/RS-TransCLIP)
5. [2025 ICCV] **Dynamic Dictionary Learning for Remote Sensing Image Segmentation.** [[paper]](https://arXiv.org/pdf/2503.06683) [[code]](https://github.com/XavierJiezou/D2LS)
6. [2025 ICCV] **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks.** [[paper]](https://arxiv.org/pdf/2411.19325) [[code]](https://github.com/The-AI-Alliance/GEO-Bench-VLM)
7. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation.** [[paper]](https://arXiv.org/abs/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
8. [2025 ICCV] **When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning.** [[paper]](https://arXiv.org/pdf/2503.07588) [[code]](https://github.com/VisionXLab/LRS-VQA)
9. [2025 AAAI] **ZoRI: Towards discriminative zero-shot remote sensing instance segmentation.** [[paper]](https://arXiv.org/abs/2412.12798) [[code]](https://github.com/HuangShiqi128/ZoRI)
10. [2024 NIPS] **Segment Any Change.** [[paper]](https://proceedings.NIPS.cc/paper_files/paper/2024/file/9415416201aa201902d1743c7e65787b-Paper-Conference.pdf) [[code]](https://github.com/Z-Zheng/pytorch-change-models)
11. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images.** [[paper]](https://arXiv.org/abs/2410.01768) [[code]](https://github.com/likyoo/SegEarth-OV)
12. [2025 CVPR] **XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?** [[paper]](https://arXiv.org/abs/2503.23771) [[code]](https://github.com/EvolvingLMMs-Lab/XLRS-Bench)
13. [2025 CVPR] **Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf) [[code]](https://github.com/MiSsU-HH/Exact)
14. [2025 Arxiv] **SegEarth-OV-2: Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images** [[paper]](https://arxiv.org/abs/2508.18067)  [[code]](https://github.com/earth-insights/SegEarth-OV-2)
15. [2025 AAAI] **Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation** [[paper]](https://arxiv.org/abs/2412.19492) [[code]](https://github.com/yecy749/GSNet)
16. [2025 Arxiv] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM)
17. [2025 Arxiv] **DescribeEarth: Describe Anything for Remote Sensing Images** [[paper]](https://arxiv.org/pdf/2509.25654v1) [[code]](https://github.com/earth-insights/DescribeEarth)

<a name="Training_Free"></a>
## Training Free Segmentation
### VLM Only
1. [2023 arXiv] **CLIPSurgery: CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks.** [[paper]](https://arXiv.org/pdf/2304.05653) [[code]](https://github.com/xmed-lab/CLIP_Surgery)
2. [2024 arXiv] **SC-CLIP: Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP)
3. [2025 arXiv] **A Survey on Training-free Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2505.22209)
4. [2022 ECCV] **Maskclip: Extract Free Dense Labels from CLIP.** [[paper]](https://arXiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP)
5. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference.** [[paper]](https://arXiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP)
6. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
7. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference.** [[paper]](https://arXiv.org/pdf/2407.12442) [[code]](https://github.com/mc-lan/ClearCLIP)
8. [2025 AAAI] **Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation.** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32602) [[code]](https://ojs.aaai.org/index.php/AAAI/article/view/32602)
9. [2022 NIPS] **ReCo: Retrieve and Co-segment for Zero-shot Transfer.** [[paper]](https://arXiv.org/pdf/2206.07045) [[code]](https://github.com/NoelShin/reco)
10. [2024 WACV] **NACLIP: Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
11. [2024 ICLR] **Vision Transformers Need Registers.** [[paper]](https://arXiv.org/pdf/2309.16588) [[code]](https://github.com/kyegomez/Vit-RGTS)
12. [2024 ICLR] **Vision Transformers Don't Need Trained Registers.** [[paper]](https://arXiv.org/pdf/2506.08010) [[code]](https://github.com/nickjiang2378/test-time-registers/tree/main)
13. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference.** [[paper]](https://arXiv.org/pdf/2411.15851) [[code]](https://github.com/yvhangyang/ResCLIP)
14. [2024 CVPR] **GEM: Grounding Everything: Emerging Localization Properties in Vision-Language Transformers.** [[paper]](https://arXiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM)
15. [2025 CVPRW] **ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements.** [[paper]](https://arXiv.org/pdf/2411.12044) [[code]](https://github.com/m-arda-aydn/ITACLIP)

***************************************************************
### VLM & VFM & Diffusion & SAM
1. [2024 arXiv] **CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2411.13836) [[code]](https://github.com/linsun449/cliper.code)
2. [2024 ECCV] **ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2408.04883v1) [[code]](https://github.com/mc-lan/ProxyCLIP)
3. [2024 ECCV] **CLIP_Dinoiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation.** [[paper]](https://arXiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser)
4. [2024 WACV] **CLIP-DIY: CLIP Dense Inference Yields Open-Vocabulary Semantic Segmentation For-Free.** [[paper]](https://arXiv.org/pdf/2309.14289) [[code]](https://github.com/wysoczanska/clip-diy)
5. [2024 IJCV] **IPSeg: Towards Training-free Open-world Segmentation via Image Prompting Foundation Models.** [[paper]](https://arXiv.org/pdf/2310.10912) [[code]](https://github.com/luckybird1994/IPSeg)
6. [2025 ICCV] **CorrCLIP: Reconstructing Correlations in CLIP with Off-the-Shelf Foundation Models for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2411.10086) [[code]](https://github.com/zdk258/CorrCLIP/tree/master)
7. [2025 ICCV] **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2411.19331) [[code]](https://github.com/lorebianchi98/Talk2DINO)
8. [2025 ICCV] **ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2506.21233) [[code]](https://github.com/xiweix/ReME)
9. [2025 ICCV] **FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2504.10487) [[code]](https://github.com/yasserben/FLOS
10. [2025 ICCV] **Trident: Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2411.09219) [[code]](https://github.com/YuHengsss/Trident)
11. [2024 NIPS] **DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut.** [[paper]](https://arXiv.org/pdf/2406.02842v2) [[code]](https://github.com/PaulCouairon/DiffCut)
12. [2025 NIPS] **TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models.** [[paper]](https://arXiv.org/pdf/2505.23769) [[code]](https://github.com/avaxiao/TextRegion)
13. [2024 AAAI] **TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training.** [[paper]](https://arXiv.org/pdf/2312.12828) [[code]](https://github.com/linyq2117/TagCLIP)
14. [2024 ICLR] **EmerDiff: Emerging Pixel-level Semantic Knowledge in Diffusion Models.** [[paper]](https://arXiv.org/pdf/2401.11739) [[code]](https://github.com/linyq2117/TagCLIP)
15. [2024 ICML] **Language-driven Cross-modal Classifier for Zero-shot Multi-label Image Recognition.** [[paper]](https://openreview.net/pdf?id=sHswzNWUW2) [[code]](https://github.com/yic20/CoMC)
16. [2025 ICML] **FlexiReID: Adaptive Mixture of Expert for Multi-Modal Person Re-Identification.** [[paper]](https://openreview.net/pdf?id=dewR2augg2)
17. [2025 ICML] **Multi-Modal Object Re-Identification via Sparse Mixture-of-Experts.** [[paper]](https://openreview.net/pdf?id=uvFE58mSnR) [[code]](https://github.com/stone96123/MFRNet)
18. [2024 CVPR] **FreeDA: Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation.** [[paper]](https://arXiv.org/pdf/2404.06542) [[code]](https://github.com/aimagelab/freeda)
19. [2025 CVPR] **GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery.** [[paper]](https://arXiv.org/pdf/2403.09974) [[code]](https://github.com/enguangW/GET)
20. [2025 CVPR] **CCD: Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification.** [[paper]](https://arXiv.org/pdf/2503.16873) [[code]](https://github.com/k0u-id/CCD)
21. [2025 CVPR] **SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2502.16911?) [[code]](https://github.com/kjmillerCURIS/SPARC)
22. [2025 CVPR] **LOPSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS/tree/main)
23. [2025 arXiv] **One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework** [[paper]](https://arxiv.org/pdf/2510.02898v1) [[code]](https://paciosoft.com/Patch-ioner/)

***************************************************************
1. [2024 WACV] **FOSSIL: Free Open-Vocabulary Semantic Segmentation through Synthetic References Retrieval.** [[paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Barsellotti_FOSSIL_Free_Open-Vocabulary_Semantic_Segmentation_Through_Synthetic_References_Retrieval_WACV_2024_paper.pdf)
2. [2024 AAAI] **TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training.** [[paper]](https://arXiv.org/pdf/2312.12828) [[code]](https://github.com/linyq2117/TagCLIP)
3. [2024 CVPR] **Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2311.17095) [[code]](https://github.com/letitiabanana/PnP-OVSS)
4. [2024 CVPR] **CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor.** [[paper]](https://arXiv.org/pdf/2312.07661) [[code]](https://github.com/kevin-ssy/CLIP_as_RNN)
5. [2024 CVPR] **Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2404.00262) [[code]](https://github.com/072jiajia/image-text-co-decomposition)
6. [2024 ECCV] **In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2408.04961) [[code]](https://github.com/dahyun-kang/lavg)
7. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2411.17150) [[code]](https://micv-yonsei.github.io/cass/)
8. [2025 ICCV] **LUDVIG: Learning-free uplifting of 2D visual features to gaussuan splatting scenes.** [[paper]](https://arXiv.org/pdf/2410.14462#page=17.85) [[code]](https://github.com/naver/ludvig)
9. [2025 CVPR] **MOS: Modeling Object-Scene Associations in Generalized Category Discovery.** [[paper]](https://arXiv.org/pdf/2503.12035) [[code]](https://github.com/JethroPeng/MOS)
10. [2024 NIPS] **Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels** [[paper]](https://arxiv.org/pdf/2409.19846)  [[code]](https://github.com/cvlab-kaist/PixelCLIP)
11. [2024 NIPS] **Renovating Names in Open-Vocabulary Segmentation Benchmarks** [[paper]](https://arxiv.org/pdf/2403.09593)
12. [2024 NIPS] **Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts** [[paper]](https://arxiv.org/pdf/2410.05963)
13. [2025 ICML] **Unlocking the Power of SAM 2 for Few-Shot Segmentation** [[paper]](https://arxiv.org/pdf/2505.14100) [[code]](https://github.com/Sam1224/FSSAM)
14. [2025 ICCV] **ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation.** [[paper]](https://arXiv.org/pdf/2506.21233) [[code]](https://github.com/xiweix/ReME)


<a name="Zero_shot_classification"></a>
## Zero shot Classification / Test Time Adaptation
1. [2024 NIPS] **SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings.** [[paper]](https://proceedings.NIPS.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf) [[code]](https://github.com/AI4LIFE-GROUP/SpLiCE)
2. [2024 NIPS] **Transclip: Boosting Vision-Language Models with Transduction.** [[paper]](https://arXiv.org/pdf/2406.01837) [[code]](https://github.com/MaxZanella/transduction-for-vlms)
3. [2025 CVPR] **Realistic Test-Time Adaptation of Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2501.03729) [[code]](https://github.com/MaxZanella/StatA)
4. [2023 AAAI] **CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention.** [[paper]](https://arXiv.org/pdf/2209.14169) [[code]](https://github.com/ZiyuGuo99/CALIP)
5. [2025 AAAI] **TIMO: Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP.** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32534) [[code]](https://github.com/lyymuwu/TIMO)
6. [2025 CVPR] **COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation.** [[paper]](https://arXiv.org/pdf/2503.23388) [[code]](https://github.com/hf618/COSMIC)
7. [2024 CVPR] **Transductive Zero-Shot and Few-Shot CLIP.** [[paper]](https://arXiv.org/pdf/2405.18437) [[code]](https://github.com/SegoleneMartin/transductive-CLIP)
8. [2023 CVPR] **Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling.** [[paper]](https://arXiv.org/pdf/2111.03930) [[code]](https://github.com/gaopengcuhk/Tip-Adapter)
9. [2024 ICLR] **GDA-CLIP: A hard-to-beat baseline for training-free clip-based adaptation.** [[paper]](https://arXiv.org/pdf/2402.04087) [[code]](https://github.com/mrflogs/ICLR24)
10. [2023 ICLR] **DCLIP: Visual Classification via Description from Large Language Models.** [[paper]](https://arXiv.org/pdf/2210.07183) [[code]](https://github.com/sachit-menon/classify_by_description_release)
11. [2023 ICCV] **CuPL: What does a platypus look like? Generating customized prompts for zero-shot image classification.** [[paper]](https://arXiv.org/pdf/2209.03320) [[code]](https://github.com/sarahpratt/CuPL)
12. [2024 CVPR] **On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?** [[paper]](https://arXiv.org/pdf/2405.02266) [[code]](https://github.com/MaxZanella/MTA)
13. [2024 NIPS] **Frustratingly Easy Test-Time Adaptation of Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2405.18330) [[code]](https://github.com/FarinaMatteo/zero)
14. [2024 NIPS] **BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping.** [[paper]](https://arXiv.org/pdf/2410.15430) [[code]](https://github.com/taolinzhang/BoostAdapter)
15. [2024 CVPR] **DMN: Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2403.17589) [[code]](https://github.com/YBZh/DMN)
16. [2023 ICCV] **Zero-Shot Composed Image Retrieval with Textual Inversion.** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf) [[code]](https://github.com/miccunifi/SEARLE.)
17. [2025 CVPR] **Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval.** [[paper]](https://arXiv.org/pdf/2412.11077) [[code]](https://github.com/Pter61/osrcir)
18. [2024 NIPS] **Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting.** [[paper]](https://arXiv.org/pdf/2410.19294) [[code]](https://github.com/zhuhsingyuu/Frolic)
19. [2025 ICML] **From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection.** [[paper]](https://arXiv.org/pdf/2505.13233) [[code]](https://github.com/BIT-DA/ABS)
20. [2025 CVPR] **PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts.** [[paper]](https://arXiv.org/pdf/2308.01313) [[code]](https://github.com/umd-huang-lab/perceptionCLIP)
21. [2024 CVPR] **ZLaP: Label Propagation for Zero-shot Classification with Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2404.04072) [[code]](https://github.com/vladan-stojnic/ZLaP)
22. [2025 CVPRW] **TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification.** [[paper]](https://arXiv.org/pdf/2503.12206) [[code]](https://github.com/ans92/TLAC)
23. [2023 NIPS] **Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP.** [[paper]](https://arXiv.org/pdf/2310.19752) [[code]](https://github.com/idstcv/InMaP)
24. [2024 ICML] **Let Go of Your Labels with Unsupervised Transfer.** [[paper]](https://arXiv.org/pdf/2406.07236?) [[code]](https://github.com/mlbio-epfl/turtle)
25. [2025 CVPR] **ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf) [[code]](https://github.com/ybendou/ProKeR)
26. [2024 CVPR] **TDA: Efficient Test-Time Adaptation of Vision-Language Model.** [[paper]](https://arXiv.org/pdf/2403.18293) [[code]](https://github.com/kdiAAA/TDA)
27. [2024 arXiv] **DOTA: Distributional test-time adaptation of Vision-Language Models** [[paper]](https://arxiv.org/pdf/2409.19375)
28. [2023 ICCV] **Black Box Few-Shot Adaptation for Vision-Language models** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf) [[code]](https://github.com/saic-fi/LFA)
29. [2025 ICCV] **Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks** [[paper]](https://arxiv.org/pdf/2509.16163v1)) [[code]](https://github.com/HettyPatel/TensorDefenseVLM)
30. [2025 arXiv] **Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models via Region, Token, and Instruction-Guided Importance** [[paper]](https://arxiv.org/pdf/2509.15704v1)
31. [2025 arXiv] **Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models** [[paper]](https://arxiv.org/pdf/2509.20751v1)
32. [2009 ICML] **Nearest Neighbors in High-Dimensional Data: The Emergence and Influence of Hubs** [[paper]](https://icml.cc/Conferences/2009/papers/360.pdf)
33. [2010 JMLR] **Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data** [[paper]](https://jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) 
34. [2023 CVPR] **noHub: Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning with Hyperspherical Embeddings** [[paper]](https://arxiv.org/pdf/2303.09352) [[code]](https://github.com/uitml/noHub)
35. [2025 CVPR] **AHubness Perspective on Representation Learning for Graph-Based Multi-View Clustering** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_A_Hubness_Perspective_on_Representation_Learning_for_Graph-Based_Multi-View_Clustering_CVPR_2025_paper.pdf) [[code]](https://github.com/zmxu196/hubREP)
36. [2025 CVPR] **NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval** [[paper]](https://arxiv.org/pdf/2503.10526) [[code]](https://github.com/zzezze/NeighborRetr/tree/main)
37. [2025 arXiv] **SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP** [[paper]](https://arxiv.org/pdf/2509.26036v1) [[code]](https://github.com/christti98/semobridge)
38. [2023 ICCV] **Not All Features Matter: Enhancing Few-shot CLIP with Adaptive Prior Refinement** [[paper]](https://arxiv.org/pdf/2304.01195) [[code]](https://github.com/yangyangyang127/APE)
39. [2025 arXiv] **SQUARE: Semantic Query-Augmented Fusion and Efficient Batch Reranking for Training-free Zero-Shot Composed Image Retrieval** [[paper]](https://arxiv.org/pdf/2509.26330v1)
40. [2025 arXiv] **Skip-It? Theoretical Conditions for Layer Skipping in Vision-Language Models** [[paper]](https://arxiv.org/pdf/2509.25584v1)
41. [2025 arXiv] **VLOD-TTA: Test-Time Adaptation of Vision-Language Object Detectors** [[paper]](https://arxiv.org/pdf/2510.00458v1) [[code]](https://github.com/imatif17/VLOD-TTA)
42. [2025 arXiv] **Bayesian Test-time Adaptation for Object Recognition and Detection with Vision-language Models** [[paper]](https://arxiv.org/pdf/2510.02750v1)



<a name="Optimal_Transport"></a>
## Optimal Transport
1. [2022 AISTATS] **Sinkformers: Transformers with Doubly Stochastic Attention.** [[paper]](https://arXiv.org/pdf/2110.11773) [[code]](https://github.com/michaelsdr/sinkformers)
2. [2024 ECCV] **OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2403.14183) [[code]](https://github.com/cubeyoung/OTSeg)
3. [2025 CVPR] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/jianwang91/POT)
4. [2025 CVPR] **RAM: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Transport.** [[paper]](https://arXiv.org/pdf/2503.15337) [[code]](https://github.com/EricTan7/RAM)
5. [2022 NIPS] **SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.** [[paper]](https://arXiv.org/pdf/2006.09882) [[code]](https://github.com/facebookresearch/swav)
6. [2023 ICRL] **PLOT: Prompt Learning with Optimal Transport for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2210.01253) [[code]](https://github.com/CHENGY12/PLOT)
7. [2024 NIPS] **OTTER: Effortless Label Distribution Adaptation of Zero-shot Models.** [[paper]](https://arXiv.org/pdf/2404.08461) [[code]](https://github.com/sprocketlab/otter)
8. [2025 ICCV] **LaZSL: Intrepretable Zero-Shot Learning with Locally-Aligned Vision-Language Model.** [[paper]](https://arXiv.org/pdf/2506.23822) [[code]](https://github.com/shiming-chen/LaZSL)
9. [2025 CVPR] **Conformal Prediction for Zero-Shot Models.** [[paper]](https://arXiv.org/pdf/2505.24693) [[code]](https://github.com/jusiro/CLIP-Conformal)
10. [2025 ICML] **ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence.** [[paper]](https://arXiv.org/abs/2505.04560) [[code]](https://github.com/ghwang-s/abkd)
11. [2024 ICLR] **EMO: Earth mover distance optimization for auto-regessive language modeling.** [[paper]](https://arXiv.org/pdf/2310.04691) [[code]](https://github.com/DRSY/EMO)
12. [2025 CVPR] **RAM: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport.** [[paper]](https://arXiv.org/abs/2503.15337) [[code]](https://github.com/EricTan7/RAM)
13. [2025 ICCV] **Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation.** [[paper]](https://iccv.thecvf.com/virtual/2025/poster/1933)
14. [2025 AAAI] **Training-free Open-Vocabulary Semantic Segmentation via Diverse Prototype Construction and Sub-region Matching** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33137)


<a name="WSSS"></a>
## Weakly Supervised Semantic Segmentation
1. [2022 CVPR] **Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers.** [[paper]](https://arXiv.org/pdf/2203.02664) [[code]](https://github.com/rulixiang/afa)
2. [2022 CVPR] **MCTFormer:Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2203.02891) [[code]](https://github.com/xulianuwa/MCTformer)
3. [2023 CVPR] **Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization.** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf) [[code]](https://github.com/xulianuwa/MMCST)
4. [2023 ICCV] **Spatial-Aware Token for Weakly Supervised Object Localization.** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf) [[code]](https://github.com/wpy1999/SAT)
5. [2023 CVPR] **Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf) [[code]](https://github.com/ShenghaiRong/BECO?tab=readme-ov-file)
6. [2023 CVPR] **ToCo:Token Contrast for Weakly-Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2303.01267) [[code]](https://github.com/rulixiang/ToCo)
7. [2023 arXiv] **MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2308.03005) [[code]](https://github.com/xulianuwa/MCTformer)
8. [2024 CVPR] **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2406.11189v1) [[code]](https://github.com/zbf1991/WeCLIP)
9. [2024 CVPR] **DuPL: Dual Student with Trustworthy Progressive Learning for RobustWeakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2403.11184) [[code]](https://github.com/Wu0409/DuPL)
10. [2024 CVPR] **Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2403.07630) [[code]](https://github.com/Barrett-python/CPAL)
11. [2024 ECCV] **DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2409.15801)
12. [2024 CVPR] **Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2402.18467) [[code]](https://github.com/zwyang6/SeCo)
13. [2024 ECCV] **CoSa:Weakly Supervised Co-training with Swapping Assignments for Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2402.17891) [[code]](https://github.com/youshyee/CoSA)
14. [2024 IEEE] **SSC:Spatial Structure Constraints for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2401.11122) [[code]](https://github.com/NUST-Machine-Intelligence-Laboratory/SSC)
15. [2024 AAAI] **Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2312.08916) [[code]](https://github.com/Jessie459/feature-self-reinforcement)
16. [2024 CVPR] **Class Tokens Infusion for Weakly Supervised Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/yoon307/CTI)
17. [2024 CVPR] **SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2401.11719) [[code]](https://github.com/Barrett-python/SFC)
18. [2024 CVPR] **PSDPM:Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_PSDPM_Prototype-based_Secondary_Discriminative_Pixels_Mining_for_Weakly_Supervised_Semantic_CVPR_2024_paper.pdf) [[code]](https://github.com/xinqiaozhao/PSDPM)
19. [2024 arXiv] **A Realistic Protocol for Evaluation of Weakly Supervised Object Localization.** [[paper]](https://arXiv.org/pdf/2404.10034) [[code]](https://github.com/shakeebmurtaza/wsol_model_selection)
20. [2025 AAAI] **MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2412.11076) [[code]](https://github.com/zwyang6/MoRe)
21. [2025 CVPR] **PROMPT-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis.** [[paper]](https://arXiv.org/pdf/2501.09333) [[code]](https://github.com/Imageomics/Prompt_CAM)
22. [2025 CVPR] **Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2503.20826) [[code]](https://github.com/zwyang6/ExCEL)
23. [2025 CVPR] **GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery.** [[paper]](https://arXiv.org/abs/2403.09974) [[code]](https://github.com/enguangW/GET)
24. [2025 arXiv] **TeD-Loc: Text Distillation for Weakly Supervised Object Localization.** [[paper]](https://arXiv.org/pdf/2501.12632) [[code]](https://github.com/shakeebmurtaza/TeDLOC)
25. [2025 arXiv] **Image Augmentation Agent for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2412.20439)
26. [2025 CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
27. [2025 CVPRW] **Prompt Categories Cluster for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2412.13823)
28. [2025 arXiv] **No time to train! Training-Free Reference-Based Instance Segmentation.** [[paper]](https://arXiv.org/pdf/2507.02798) [[code]](https://github.com/miquel-espinosa/no-time-to-train)


<a name="Graph_Structure"></a>
## Graph Structure
1. [2016 AAAI] **The Constrained Laplacian Rank Algorithm for Graph-Based Clustering.** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/10302) [[code]](https://github.com/convexfi/spectralGraphTopology/blob/97eee40c8aa30cd62fa01b6e9e01e20197fc7940/R/constrLaplacianRank.R)
2. [2016 IJCAI] **Parameter-Free Auto-Weighted Multiple Graph Learning: A Framework for Multiview Clustering and Semi-Supervised Classification.** [[paper]](https://www.ijcai.org/Proceedings/16/Papers/269.pdf)
3. [2023 NIPS] **GSLB: The Graph Structure Learning Benchmark.** [[paper]](https://arXiv.org/pdf/2310.05174) [[code]](https://github.com/GSL-Benchmark/GSLB)
4. [2024 AAAI] **Catalyst for Clustering-based Unsupervised Object Re-Identification: Feature Calibration.** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28092) [[code]](https://github.com/lhf12278/FCM-ReID)
5. [2025 ICLR] **Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model.** [[paper]](https://arXiv.org/pdf/2412.18303) [[code]](https://github.com/Yushu-Li/ECALP)
6. [network] **.** [[paper]](https://www.gnn.club/?p=2170)


<a name="VPR"></a>
## Visual Place Recognition
1. [2022 CVPR] **CosPlace: Rethinking Visual Geo-localization for Large-Scale Applications.** [[paper]](https://arXiv.org/pdf/2204.02287) [[code]](https://github.com/gmberton/CosPlace)
2. [2024 CVPR] **CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition.** [[paper]](https://arXiv.org/pdf/2402.19231) [[code]](https://github.com/Lu-Feng/CricaVPR)
3. [2024 CVPR] **BoQ: A Place is Worth a Bag of Learnable Queries.** [[paper]](https://arXiv.org/pdf/2405.07364) [[code]](https://github.com/amaralibey/Bag-of-Queries)
4. [2024 NIPS] **SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition.** [[paper]](https://openreview.net/pdf?id=bZpZMdY1sj) [[code]](https://github.com/Lu-Feng/SuperVLAD)
5. [2024 ECCV] **Revisit Anything: Visual Place Recognition via Image Segment Retrieval.** [[paper]](https://arXiv.org/pdf/2409.18049) [[code]](https://github.com/AnyLoc/Revisit-Anything)
6. [2025 arXiv] **HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition.** [[paper]](https://arXiv.org/pdf/2506.04764) [[code]](https://github.com/suhan-woo/HypeVPR)
7. [2023 IROS] **Training-Free Attentive-Patch Selection for Visual Place Recognition.** [[paper]](https://ieeexplore.ieee.org/abstract/document/10342347)


<a name="Token_fusion"></a>
## Token Mering, Clustering and Pruning
1. [2021 NIPS] **TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?** [[paper]](https://arXiv.org/pdf/2106.11297) [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)
2. [2022 CVPR] **GroupViT: Semantic Segmentation Emerges from Text Supervision.** [[paper]](https://arXiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT)
3. [2022 CVPR] **MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2308.03005) [[code]](https://github.com/xulianuwa/MCTformer)
4. [2023 CVPR] **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** [[paper]](https://arXiv.org/pdf/2301.12597) [[code]](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
5. [2023 ICCV] **Perceptual Grouping in Contrastive Vision-Language Models.** [[paper]](https://arXiv.org/abs/2210.09996)
6. [2023 ICLR] **GPVIT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation.** [[paper]](https://arXiv.org/pdf/2212.06795) [[code]](https://github.com/ChenhongyiYang/GPViT)
7. [2023 CVPR] **SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2302.12242) [[code]](https://github.com/MendelXu/SAN)
8. [2024 CVPR] **Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers.** [[paper]](https://arXiv.org/pdf/2403.10030) [[code]](https://github.com/mlvlab/MCTF)
9. [2024 CVPR] **Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding.** [[paper]](https://arXiv.org/pdf/2311.08046) [[code]](https://github.com/PKU-YuanGroup/Chat-UniVi)
10. [2024 ICLR] **LaVIT: Unified language-vision pretraining in LLM with dynamic discrete visual tokenization.** [[paper]](https://arXiv.org/pdf/2309.04669) [[code]](https://github.com/jy0205/LaVIT)
11. [2024 arXiv] **TokenPacker: Efficient Visual Projector for Multimodal LLM.** [[paper]](https://arXiv.org/abs/2407.02392) [[code]](https://github.com/CircleRadon/TokenPacker)
12. [2024 arXiv] **DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models.** [[paper]](https://arXiv.org/pdf/2405.20985) [[code]](https://github.com/yaolinli/DeCo)
13. [2024 CVPR] **Grounding Everything: Emerging Localization Properties in Vision-Language Transformers.** [[paper]](https://arXiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM)
14. [2025 CVPR] **PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models.** [[paper]](https://arXiv.org/pdf/2504.08966) [[code]](https://github.com/orailix/PACT/tree/main)
15. 


<a name="Segmentation_and_Detection"></a>
## Segmentation and Detection
1. [2015 CVPR] **FCN: Fully Convolutional Networks for Semantic Segmentation.** [[paper]](https://arXiv.org/abs/1411.4038) [[code]](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)
2. [2016 MICCAI] **UNet: Convolutional Networks for Biomedical Image Segmentation.** [[paper]](https://arXiv.org/pdf/1505.04597)
3. [2017 arXiv] **DeepLabV3: Rethinking atrous convolution for semantic image segmentation.** [[paper]](https://arXiv.org/pdf/1706.05587)
4. [2018 CVPR] **DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.** [[paper]](https://arXiv.org/pdf/1802.02611)
5. [2019 CVPR] **Semantic FPN: Panoptic Feature Pyramid Networks.** [[paper]](https://arXiv.org/pdf/1901.02446)
6. [2021 CVPR] **SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers.** [[paper]](https://arXiv.org/pdf/2012.15840) [[code]](https://github.com/fudan-zvg/SETR)
7. [2021 ICCV] **Segmenter: Transformer for Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2105.05633) [[code]](https://github.com/rstrudel/segmenter)
8. [2021 NIPS] **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** [[paper]](https://arXiv.org/pdf/2105.15203) [[code]](https://github.com/NVlabs/SegFormer)
9. [2021 CVPR] **MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2107.06278) [[code]](https://github.com/facebookresearch/MaskFormer)
10. [2022 CVPR] **Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation.** [[paper]](https://arXiv.org/pdf/2112.01527) [[code]](https://github.com/facebookresearch/Mask2Former)
11. [2024 CVPR] **Rein: Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2312.04265) [[code]](https://github.com/w1oves/Rein)
12. [2015 NIPS] **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.** [[paper]](https://arXiv.org/pdf/1506.01497)
13. [2020 ECCV] **DETR: End-to-End Object Detection with Transformers.** [[paper]](https://arXiv.org/pdf/2005.12872) [[code]](https://github.com/facebookresearch/detr)
14. [2021 ICLR] **Deformable DETR: Deformable Transformers for End-to-End Object Detection.** [[paper]](https://arXiv.org/pdf/2010.04159) [[code]](https://github.com/fundamentalvision/Deformable-DETR)
15. [2023 ICLR] **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.** [[paper]](https://arXiv.org/pdf/2203.03605) [[code]](https://github.com/IDEA-Research/DINO)


<a name="Backbone"></a>
## Backbone
1. [2017 NIPS] **transfomer: Attention Is All You Need.** [[paper]](https://arXiv.org/pdf/1706.03762) [[code]](https://github.com/tensorflow/tensor2tensor)
2. [2021 ICLR] **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** [[paper]](https://arXiv.org/pdf/2010.11929) [[code]](https://github.com/google-research/vision_transformer)
3. [2021 ICML] **DeiT: Training data-efficient image transformers & distillation through attention.** [[paper]](https://arXiv.org/pdf/2012.12877)
4. [2021 ICCV] **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** [[paper]](https://arXiv.org/pdf/2103.14030) [[code]](https://github.com/microsoft/Swin-Transformer)
5. [2021 NIPS] **Twins: Revisiting the Design of Spatial Attention in Vision Transformers.** [[paper]](https://arXiv.org/pdf/2104.13840) [[code]](https://github.com/Meituan-AutoML/Twins)
6. [2022 CVPR] **Hyperbolic Vision Transformers: Combining Improvements in Metric Learning.** [[paper]](https://arXiv.org/pdf/2203.10833) [[code]](https://github.com/htdt/hyp_metric)
7. [2022 ICLR] **BEiT: BERT Pre-Training of Image Transformers.** [[paper]](https://arXiv.org/pdf/2106.08254) [[code]](https://github.com/microsoft/unilm/tree/master/beit)
8. [2022 CVPR] **MAE: Masked Autoencoders Are Scalable Vision Learners.** [[paper]](https://arXiv.org/pdf/2111.06377) [[code]](https://github.com/facebookresearch/mae)
9. [2022 CVPR] **PoolFormer: MetaFormer is Actually What You Need for Vision.** [[paper]](https://arXiv.org/pdf/2111.11418) [[code]](https://github.com/sail-sg/poolformer)
10. [2022 NIPS] **SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2209.08575) [[code]](https://github.com/Visual-Attention-Network/SegNeXt)
11. [2023 ICCV] **OpenSeeD: A simple framework for open-vocabulary segmentation and detection.** [[paper]](https://arXiv.org/pdf/2303.08131) [[code]](https://github.com/IDEA-Research/OpenSeeD)
12. [2023 arXiv] **SAM: Segment Anything.** [[paper]](https://arXiv.org/pdf/2304.02643) [[code]](https://github.com/facebookresearch/segment-anything) [[demo]](https://segment-anything.com/demo)
13. [2024 arXiv] **SAM 2: Segment Anything in Images and Videos.** [[paper]](https://arXiv.org/pdf/2408.00714) [[code]](https://github.com/facebookresearch/sam2) [[demo]](https://sam2.metademolab.com/)


<a name="CLIP"></a>
## CLIP
1. [2021 ICML] **CLIP: Learning transferable visual models from natural language supervision.** [[paper]](https://arXiv.org/pdf/2103.00020) [[code]](https://github.com/OpenAI/CLIP)
2. [2022 IJCV] **CoOp: Learning to Prompt for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp)
3. [2022 ECCV] **VPT: Visual Prompt Tuning.** [[paper]](https://arXiv.org/pdf/2203.12119) [[code]](https://github.com/kmnp/vpt)
4. [2022 ICLR] **LoRA: Low-Rank Adaptation of Large Language Models.** [[paper]](https://arXiv.org/pdf/2106.09685) [[code]](https://github.com/microsoft/LoRA)
5. [2022 NIPS] **TPT: Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2209.07511) [[code]](https://github.com/azshue/TPT)
6. [2022 arXiv] **UPL: Unsupervised Prompt Learning for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2204.03649) [[code]](https://github.com/tonyhuang2022/UPL)
7. [2022 arXiv] **CLIPPR: Improving Zero-Shot Models with Label Distribution Priors.** [[paper]](https://arXiv.org/pdf/2212.00784) [[code]](https://github.com/jonkahana/CLIPPR)
8. [2022 CVPR] **CoCoOp: Conditional Prompt Learning for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2203.05557) [[code]](https://github.com/KaiyangZhou/CoOp)
9. [2023 CVPR] **TaskRes: Task Residual for Tuning Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2211.10277) [[code]](https://github.com/geekyutao/TaskRe)
10. [2023 ICML] **POUF: Prompt-Oriented Unsupervised Fine-tuning for Large Pre-trained Models.** [[paper]](https://arXiv.org/pdf/2305.00350) [[code]](https://github.com/korawat-tanwisuth/POUF)
11. [2023 NIPS] **Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning.** [[paper]](https://arXiv.org/pdf/2306.01669) [[code]](https://github.com/BatsResearch/menghini-NIPS23-code)
12. [2023 NIPS] **LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections.** [[paper]](https://arXiv.org/pdf/2305.18287) [[code]](https://github.com/jmiemirza/LaFTer)
13. [2023 PRCV] **Unsupervised Prototype Adapter for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2308.11507)
14. [2023 IJCV] **CLIP-Adapter: Better Vision-Language Models with Feature Adapters.** [[paper]](https://arXiv.org/pdf/2110.04544) [[code]](https://github.com/gaopengcuhk/CLIP-Adapter)
15. [2024 CVPR] **CODER: Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification.** [[paper]](https://arXiv.org/pdf/2404.17753) [[code]](https://github.com/YCaigogogo/CODER)
16. [2024 CVPR] **LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP.** [[paper]](https://arXiv.org/pdf/2404.02285) [[code]](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline)
17. [2024 CVPR] **PromptKD: Unsupervised Prompt Distillation for Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2403.02781) [[code]](https://github.com/zhengli97/PromptKD)
18. [2024 CVPR] **Transfer CLIP for Generalizable Image Denoising.** [[paper]](https://arXiv.org/pdf/2403.15132) [[code]](https://github.com/alwaysuu/CLIPDenoising)
19. [2024 ECCV] **BRAVE: Broadening the visual encoding of vision-language model.** [[paper]](https://arXiv.org/pdf/2404.07204)
20. [2024 ICML] **Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data.** [[paper]](https://arXiv.org/pdf/2406.10502) [[code]](https://github.com/vanillaer/CPL-ICML2024)
21. [2024 CVPR] **CLIP-KD: An Empirical Study of CLIP Model Distillation.** [[paper]](https://arXiv.org/pdf/2307.12732) [[code]](https://github.com/winycg/CLIP-KD)
22. [2025 WACV] **DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2408.08855) [[code]](https://github.com/Externalhappy/DPA)
23. [2025 WACV] **Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models.** [[paper]](https://arXiv.org/pdf/2403.12952) [[code]](https://github.com/elaine-sui/TPS)
24. [2025 WACV] **LATTECLIP: Unsupervised CLIP Fine-Tuning via LMM-Synthetic Texts.** [[paper]](https://arXiv.org/pdf/2410.08211) [[code]](https://github.com/astra-vision/LatteCLIP)
25. [2025 ICLR] **CROSS THE GAP: EXPOSING THE INTRA-MODAL MISALIGNMENT IN CLIP VIA MODALITY INVERSION.** [[paper]](https://arXiv.org/pdf/2502.04263) [[code]](https://github.com/miccunifi/Cross-the-Gap)
26. [2025 ICLR] **CLIP’s Visual Embedding Projector is a Few-shot Cornucopia.** [[paper]](https://arXiv.org/pdf/2410.05270) [[code]](https://github.com/astra-vision/ProLIP)
27. [2025 CVPR] **DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers.** [[paper]](https://arXiv.org/pdf/2505.23694) [[code]](https://github.com/Noahsark/DA-VPT)
28. [2025 ICML] **Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models.** [[paper]](https://arXiv.org/pdf/2506.02557) [[code]](https://github.com/peterant330/KUEA)
29. [2025 CVPR] **Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification.** [[paper]](https://arXiv.org/pdf/2503.16873) [[code]](https://github.com/k0u-id/CCD)
30. [2024 CVPR] **Multi-Modal Adapter for Vision-Language Models.** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf) [[code]](https://github.com/ZjjConan/VLM-MultiModalAdapter)
31. [2025 arXiv] **DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning** [[paper]](https://arxiv.org/pdf/2509.20912v1) [[code]](https://github.com/tinnel123666888/defacto)
32. [2025 arXiv] **Hierarchical representation matching for clip-based class-incremental learning** [[paper]](https://arxiv.org/pdf/2509.22645v1)
33. [2025 EMNLP]**From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-LENS** [[paper]](https://arxiv.org/pdf/2510.02292v1) [[code]](https://github.com/compling-wat/vlm-lens)


<a name="open_vocabulary"></a>
## Open vocabulary
### segmentation
1. [2022 ICLR] **Lseg: Language-driven semantic segmentation (Supervised).** [[paper]](https://arXiv.org/pdf/2201.03546) [[code]](https://github.com/isl-org/lang-seg)
2. [2022 CVPR] **ZegFormer: Decoupling Zero-Shot Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2112.07910) [[code]](https://github.com/dingjiansw101/ZegFormer)
3. [2022 ECCV] **MaskCLIP+: Extract Free Dense Labels from CLIP.** [[paper]](https://arXiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP)
4. [2022 ECCV] **ViL-Seg: Open-World Semantic Segmentation via Contrasting and Clustering Vision-Language Embeddings.** [[paper]](https://arXiv.org/pdf/2207.08455v2)
5. [2022 CVPR] **GroupViT: Semantic Segmentation Emerges from Text Supervision (Open-Vocabulary Zero-Shot).** [[paper]](https://arXiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT)
6. [2022 ECCV] **OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels.** [[paper]](https://arXiv.org/pdf/2112.12143)
7. [2023 CVPR] **FreeSeg: Unified, Universal, and Open-Vocabulary Image Segmentation.** [[paper]](https://arXiv.org/pdf/2303.17225) [[code]](https://github.com/bytedance/FreeSeg)
8. [2023 ICML] **SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation (Zero-Shot).** [[paper]](https://arXiv.org/pdf/2211.14813) [[code]](https://github.com/ArrowLuo/SegCLIP)
9. [2023 CVPR] **ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2212.03588) [[code]](https://github.com/ZiqinZhou66/ZegCLIP)
10. [2023 CVPR] **X-Decoder: Generalized Decoding for Pixel, Image, and Language.** [[paper]](https://arXiv.org/pdf/2212.11270) [[code]](https://github.com/microsoft/X-Decoder/tree/main)
11. [2023 CVPR] **ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models.** [[paper]](https://arXiv.org/pdf/2303.04803) [[code]](https://github.com/NVlabs/ODISE)
12. [2023 ICML] **MaskCLIP: Open-Vocabulary Universal Image Segmentation with MaskCLIP.** [[paper]](https://arXiv.org/pdf/2208.08984) [[code]](https://github.com/mlpc-ucsd/MaskCLIP)
13. [2023 CVPR] **SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2302.12242) [[code]](https://github.com/MendelXu/SAN)
14. [2024 ECCV] **CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation.** [[paper]](https://arXiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser)
15. [2024 CVPR] **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation.** [[paper]](https://arXiv.org/pdf/2311.15537) [[code]](https://github.com/xb534/SED)
16. [2024 TPAMI] **Review: Towards Open Vocabulary Learning: A Survey.** [[paper]](https://arXiv.org/pdf/2306.15880) [[code]](https://github.com/jianzongwu/Awesome-Open-Vocabulary)
17. [2025 ICCV] **Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction.** [[paper]](https://arXiv.org/abs/2412.06244) [[code]](https://github.com/HVision-NKU/DenseVLM)
18. [2024 CVPR] **Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Exploring_Regional_Clues_in_CLIP_for_Zero-Shot_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/Jittor/JSeg)
19. [2025 CVPR] **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception.** [[paper]](https://arXiv.org/pdf/2505.04410) [[code]](https://github.com/xiaomoguhz/DeCLIP)
20. [2025 arXiv] **REFAM: Attention magnets for zero-shot referral segmentaion** [[paper]](https://arxiv.org/pdf/2509.22650v1)


### object detection
1. [2021 CVPR] **Open-Vocabulary Object Detection Using Captions.** [[paper]](https://arXiv.org/pdf/2011.10678) [[code]](https://github.com/alirezazareian/ovr-cnn)
2. [2022 ICLR] **ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation.** [[paper]](https://arXiv.org/pdf/2104.13921) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
3. [2022 CVPR] **GLIP: Grounded Language-Image Pre-training.** [[paper]](https://arXiv.org/pdf/2112.03857) [[code]](https://github.com/microsoft/GLIP)
4. [2022 NIPS] **GLIPv2: Unifying Localization and Vision-Language Understanding.** [[paper]](https://arXiv.org/pdf/2206.05836) [[code]](https://github.com/microsoft/GLIP)
5. [2024 ICCV] **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** [[paper]](https://arxiv.org/pdf/2303.05499) [[code]](https://github.com/IDEA-Research/GroundingDINO)

### Active learning/ Data Selection
1. [2025 arXiv] **Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories** [[paper]](https://arxiv.org/pdf/2510.01454v1) [[code]](https://bigml-cs-ucla.github.io/XMAS-project-page/)
2. [2025 arXiv] **AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding** [[paper]](https://arxiv.org/pdf/2510.02778v1) [[code]](https://github.com/Xian867/AdaRD-Key)

<a name="Other"></a>
## Other Technologies
1. [2016 CVPRW] **pixel shuffle.** [[paper]](https://arXiv.org/pdf/1609.05158)
2. [2020 NIPS] **DDPM: Denoising Diffusion Probabilistic Models.** [[paper]](https://arXiv.org/pdf/2006.11239) [[code]](https://github.com/hojonathanho/diffusion)
