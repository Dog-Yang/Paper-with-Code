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
10. [[Open vocabulary segmentation and detection](#open_segmentation)]                
11. [[segmentation and detection](#Segmentation_and_Detection)]           
12. [[Other](#Other)]      
-----------------------------------------------------------------------------------------------


<a name="Remote_Sensing"></a>
## Remote Sensing

 1. [2025 TGRS]      A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation. [[code](https://github.com/sstary/SSRS)] [[paper](https://ieeexplore.ieee.org/document/11063320)]
 2. [2025 ICCV]      Dynamic Dictionary Learning for Remote Sensing Image Segmentation. [[code](https://github.com/XavierJiezou/D2LS)] [[paper](https://arxiv.org/pdf/2503.06683)]
 3. [2025 ICCV]      GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks. [[code](https://github.com/The-AI-Alliance/GEO-Bench-VLM)] [[paper](https://arxiv.org/abs/2411.15789)]
 4. [2024 NeurIPS]   Segment Any Change. [[code](https://github.com/Z-Zheng/pytorch-change-models)] [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9415416201aa201902d1743c7e65787b-Paper-Conference.pdf)]
 5. [2025 Arxiv]     DynamicEarth: How Far are We from Open-Vocabulary Change Detection? [[code](https://github.com/likyoo/DynamicEarth)] [[paper](https://arxiv.org/abs/2501.12931)]
 6. [2025 CVPR]      SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images. [[code](https://github.com/likyoo/SegEarth-OV)] [[paper](https://arxiv.org/abs/2410.01768)]
 7. [2025 ICCV]      SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation. [[code](https://github.com/HuangShiqi128/SCORE)] [[paper](https://arxiv.org/abs/2507.12857)]
 8. [2025 ICCV]      When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning. [[code](https://github.com/VisionXLab/LRS-VQA)] [[paper](https://arxiv.org/pdf/2503.07588)]
 9. [2025 CVPR]      XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery? [[code](https://github.com/EvolvingLMMs-Lab/XLRS-Bench)] [[paper](https://arxiv.org/abs/2503.23771)]
10. [2025 CVPR]      Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation. [[code](https://github.com/MiSsU-HH/Exact)] [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf)]
11. [2025 ICASSP]   Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification. [[code](https://github.com/elkhouryk/RS-TransCLIP)] [[paper](https://arxiv.org/abs/2409.00698)]
12. [2023 ICCV]      Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning. [[code](https://github.com/chunmeifeng/DiffTPT)] [[paper](https://arxiv.org/abs/2308.05569)]
13. [2025 AAAI]      ZoRI: Towards discriminative zero-shot remote sensing instance segmentation. [[code](https://github.com/HuangShiqi128/ZoRI)] [[paper](https://arxiv.org/abs/2412.12798)]


<a name="Training_Free"></a>
## Training Free Segmentation
### VLM Only

 1. [2022 ECCV]       Maskclip: Extract Free Dense Labels from CLIP. [[code](https://github.com/chongzhou96/MaskCLIP)] [[paper](https://arxiv.org/pdf/2112.01071)]
 2. [2022 NeurIPS]     ReCo: Retrieve and Co-segment for Zero-shot Transfer. [[code](https://github.com/NoelShin/reco)] [[paper](https://arxiv.org/pdf/2206.07045)]
 3. [2023 Arxiv]       CLIPSurgery: CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks. [[code](https://github.com/xmed-lab/CLIP_Surgery)] [[paper](https://arxiv.org/pdf/2304.05653)]
 4. [2024 ECCV]       SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference. [[code](https://github.com/wangf3014/SCLIP)] [[paper](https://arxiv.org/pdf/2312.01597)]
 5. [2024 WACV]       NACLIP: Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation. [[code](https://github.com/sinahmr/NACLIP)] [[paper](https://arxiv.org/pdf/2404.08181)]
 6. [2024 Arxiv]       SC-CLIP: Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation. [[code](https://github.com/SuleBai/SC-CLIP)] [[paper](https://arxiv.org/pdf/2411.15869)]
 7. [2025 CVPR]       ResCLIP: Residual Attention for Training-free Dense Vision-language Inference. [[code](https://github.com/yvhangyang/ResCLIP)] [[paper](https://arxiv.org/pdf/2411.15851)]
 8. [2024 ECCV]       ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference. [[code](https://github.com/mc-lan/ClearCLIP)] [[paper](https://arxiv.org/pdf/2407.12442)]
 9. [2024 CVPR]       GEM: Grounding Everything: Emerging Localization Properties in Vision-Language Transformers. [[code](https://github.com/WalBouss/GEM)] [[paper](https://arxiv.org/pdf/2312.00878)]
10. [2025 AAAI]       Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation. [[code](https://ojs.aaai.org/index.php/AAAI/article/view/32602)] [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32602)]
11. [2025 CVPRW]      ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements. [[code](https://github.com/m-arda-aydn/ITACLIP)] [[paper](https://arxiv.org/pdf/2411.12044)]
12. [2024 ECCV]       Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation. [[code](https://github.com/leaves162/CLIPtrase)] [[paper](https://arxiv.org/pdf/2407.08268)]
13. [2025.05 Arxiv]   A Survey on Training-free Open-Vocabulary Semantic Segmentation. [[paper](https://arxiv.org/pdf/2505.22209)]
14. [2024 ICLR]       Vision Transformers Need Registers. [[code](https://github.com/kyegomez/Vit-RGTS)] [[paper](https://arxiv.org/pdf/2309.16588)]
15. [2024 ICLR]       Vision Transformers Don't Need Trained Registers. [[code](https://github.com/nickjiang2378/test-time-registers/tree/main)] [[paper](https://arxiv.org/pdf/2506.08010)]

***************************************************************
### VLM & VFM & Diffusion & SAM

 1. [2024 ECCV]      ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation. [[code](https://github.com/mc-lan/ProxyCLIP)] [[paper](https://arxiv.org/pdf/2408.04883v1)]
 2. [2024 ECCV]      CLIP_Dinoiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation. [[code](https://github.com/wysoczanska/clip_dinoiser)] [[paper](https://arxiv.org/pdf/2312.12359)]
 3. [2024 Arxiv]     CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/linsun449/cliper.code)] [[paper](https://arxiv.org/pdf/2411.13836)]
 4. [2024 WACV]      CLIP-DIY: CLIP Dense Inference Yields Open-Vocabulary Semantic Segmentation For-Free. [[code](https://github.com/wysoczanska/clip-diy)] [[paper](https://arxiv.org/pdf/2309.14289)]
 5. [2024 NeurIPS]   DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut. [[code](https://github.com/PaulCouairon/DiffCut)] [[paper](https://arxiv.org/pdf/2406.02842v2)]
 6. [2024 CVPR]      FreeDA: Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation. [[code](https://github.com/aimagelab/freeda)] [[paper](https://arxiv.org/pdf/2404.06542)]
 7. [2024 IJCV]      IPSeg: Towards Training-free Open-world Segmentation via Image Prompting Foundation Models. [[code](https://github.com/luckybird1994/IPSeg)] [[paper](https://arxiv.org/pdf/2310.10912)]
 8. [2025 CVPR]      LOPSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation. [[code](https://github.com/vladan-stojnic/LPOSS/tree/main)] [[paper](https://arxiv.org/pdf/2503.19777)]
 9. [2025 ICCV]      FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation. [[code](https://github.com/yasserben/FLOSS)] [[paper](https://arxiv.org/pdf/2504.10487)]
10. [2025 ICCV]      Trident: Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation. [[code](https://github.com/YuHengsss/Trident)] [[paper](https://arxiv.org/pdf/2411.09219)]
11. [2025 NeurIPS]   TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models. [[code](https://github.com/avaxiao/TextRegion)] [[paper](https://arxiv.org/pdf/2505.23769)]
12. [2025 ICCV]      CorrCLIP: Reconstructing Correlations in CLIP with Off-the-Shelf Foundation Models for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/zdk258/CorrCLIP/tree/master)] [[paper](https://arxiv.org/pdf/2411.10086)]
13. [2025 ICCV]      Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation. [[code](https://github.com/lorebianchi98/Talk2DINO)] [[paper](https://arxiv.org/pdf/2411.19331)]
14. [2024 ICLR]      EmerDiff: Emerging Pixel-level Semantic Knowledge in Diffusion Models. [[code](https://github.com/linyq2117/TagCLIP)] [[paper](https://arxiv.org/pdf/2401.11739)]
15. [2025 ICCV]      ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation. [[code](https://github.com/xiweix/ReME)] [[paper](https://arxiv.org/pdf/2506.21233)]
16. [2025 CVPR]      GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery. [[code](https://github.com/enguangW/GET)] [[paper](https://arxiv.org/pdf/2403.09974)]
17. [2025 ICML]      Multi-Modal Object Re-Identification via Sparse Mixture-of-Experts. [[code](https://github.com/stone96123/MFRNet)] [[paper](https://openreview.net/pdf?id=uvFE58mSnR)]
18. [2025 ICML]      FlexiReID: Adaptive Mixture of Expert for Multi-Modal Person Re-Identification. [[paper](https://openreview.net/pdf?id=dewR2augg2)]
19. [2025 CVPR]      CCD: Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification. [[code](https://github.com/k0u-id/CCD)] [[paper](https://arxiv.org/pdf/2503.16873)]
20. [2024 ICML]      Language-driven Cross-modal Classifier for Zero-shot Multi-label Image Recognition. [[code](https://github.com/yic20/CoMC)] [[paper](https://openreview.net/pdf?id=sHswzNWUW2)]
21. [2024 AAAI]      TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training. [[code](https://github.com/linyq2117/TagCLIP)] [[paper](https://arxiv.org/pdf/2312.12828)]
22. [2025 CVPR]      SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models. [[code](https://github.com/kjmillerCURIS/SPARC)] [[paper](https://arxiv.org/pdf/2502.16911?)]

***************************************************************

 1. [2024 WACV]   FOSSIL: Free Open-Vocabulary Semantic Segmentation through Synthetic References Retrieval. [[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Barsellotti_FOSSIL_Free_Open-Vocabulary_Semantic_Segmentation_Through_Synthetic_References_Retrieval_WACV_2024_paper.pdf)]
 2. [2024 AAAI]   TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training. [[code](https://github.com/linyq2117/TagCLIP)] [[paper](https://arxiv.org/pdf/2312.12828)]
 3. [2024 CVPR]   Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models. [[code](https://github.com/letitiabanana/PnP-OVSS)] [[paper](https://arxiv.org/pdf/2311.17095)]
 4. [2024 CVPR]   CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor. [[code](https://github.com/kevin-ssy/CLIP_as_RNN)] [[paper](https://arxiv.org/pdf/2312.07661)]
 5. [2024 CVPR]   Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/072jiajia/image-text-co-decomposition)] [[paper](https://arxiv.org/pdf/2404.00262)]
 6. [2024 ECCV]   In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/dahyun-kang/lavg)] [[paper](https://arxiv.org/pdf/2408.04961)]
 7. [2025 CVPR]   Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation. [[code](https://micv-yonsei.github.io/cass/)] [[paper](https://arxiv.org/pdf/2411.17150)]
 8. [2025 ICCV]   LUDVIG: Learning-free uplifting of 2D visual features to gaussuan splatting scenes. [[code](https://github.com/naver/ludvig)] [[paper](https://arxiv.org/pdf/2410.14462#page=17.85)]
 9. [2025 CVPR]   MOS: Modeling Object-Scene Associations in Generalized Category Discovery. [[code](https://github.com/JethroPeng/MOS)] [[paper](https://arxiv.org/pdf/2503.12035)]


<a name="Zero_shot_classification"></a>
## Zero shot Classification / Test Time Adaptation
### parameter free

 1. [2024 NeurIPS]   SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings. [[code](https://github.com/AI4LIFE-GROUP/SpLiCE)] [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf)]
 2. [2024 NeurIPS]   Transclip: Boosting Vision-Language Models with Transduction. [[code](https://github.com/MaxZanella/transduction-for-vlms)] [[paper](https://arxiv.org/pdf/2406.01837)]
 3. [2025 CVPR]      Realistic Test-Time Adaptation of Vision-Language Models. [[code](https://github.com/MaxZanella/StatA)] [[paper](https://arxiv.org/pdf/2501.03729)]
 4. [2023 AAAI]      CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention. [[code](https://github.com/ZiyuGuo99/CALIP)] [[paper](https://arxiv.org/pdf/2209.14169)]
 5. [2025 AAAI]      TIMO: Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP. [[code](https://github.com/lyymuwu/TIMO)] [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32534)]
 6. [2025 CVPR]      COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation. [[code](https://github.com/hf618/COSMIC)] [[paper](https://arxiv.org/pdf/2503.23388)]
 7. [2024 CVPR]      Transductive Zero-Shot and Few-Shot CLIP. [[code](https://github.com/SegoleneMartin/transductive-CLIP)] [[paper](https://arxiv.org/pdf/2405.18437)]
 8. [2023 CVPR]      Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. [[code](https://github.com/gaopengcuhk/Tip-Adapter)] [[paper](https://arxiv.org/pdf/2111.03930)]
 9. [2024 ICLR]      GDA-CLIP: A hard-to-beat baseline for training-free clip-based adaptation. [[code](https://github.com/mrflogs/ICLR24)] [[paper](https://arxiv.org/pdf/2402.04087)]
10. [2023 ICLR]      DCLIP: Visual Classification via Description from Large Language Models. [[code](https://github.com/sachit-menon/classify_by_description_release)] [[paper](https://arxiv.org/pdf/2210.07183)]
11. [2023 ICCV]      CuPL: What does a platypus look like? Generating customized prompts for zero-shot image classification. [[code](https://github.com/sarahpratt/CuPL)] [[paper](https://arxiv.org/pdf/2209.03320)]

***************************************************************
### parameter fitting

 1. [2024 CVPR]      On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? [[code](https://github.com/MaxZanella/MTA)] [[paper](https://arxiv.org/pdf/2405.02266)]
 2. [2024 NeurIPS]   Frustratingly Easy Test-Time Adaptation of Vision-Language Models. [[code](https://github.com/FarinaMatteo/zero)] [[paper](https://arxiv.org/pdf/2405.18330)]
 3. [2024 CVPR]      TDA: Efficient Test-Time Adaptation of Vision-Language Model. [[code](https://github.com/kdiAAA/TDA)] [[paper](https://arxiv.org/pdf/2403.18293)]
 4. [2024 NeurIPS]   BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping. [[code](https://github.com/taolinzhang/BoostAdapter)] [[paper](https://arxiv.org/pdf/2410.15430)]
 5. [2024 CVPR]      DMN: Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models. [[code](https://github.com/YBZh/DMN)] [[paper](https://arxiv.org/pdf/2403.17589)]
 6. [2023 ICCV]      Zero-Shot Composed Image Retrieval with Textual Inversion. [[code](https://github.com/miccunifi/SEARLE.)] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf)]
 7. [2025 CVPR]      Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval. [[code](https://github.com/Pter61/osrcir)] [[paper](https://arxiv.org/pdf/2412.11077)]
 8. [2024 NeurIPS]   Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting. [[code](https://github.com/zhuhsingyuu/Frolic)] [[paper](https://arxiv.org/pdf/2410.19294)]
 9. [2025 ICML]      From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection. [[code](https://github.com/BIT-DA/ABS)] [[paper](https://arxiv.org/pdf/2505.13233)]
10. [2025 CVPR]      PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts. [[code](https://github.com/umd-huang-lab/perceptionCLIP)] [[paper](https://arxiv.org/pdf/2308.01313)]
11. [2024 CVPR]      ZLaP: Label Propagation for Zero-shot Classification with Vision-Language Models. [[code](https://github.com/vladan-stojnic/ZLaP)] [[paper](https://arxiv.org/pdf/2404.04072)]
12. [2025 CVPRW]     TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification. [[code](https://github.com/ans92/TLAC)] [[paper](https://arxiv.org/pdf/2503.12206)]
13. [2023 NeurIPS]   Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP. [[code](https://github.com/idstcv/InMaP)] [[paper](https://arxiv.org/pdf/2310.19752)]
14. [2024 ICML]      Let Go of Your Labels with Unsupervised Transfer. [[code](https://github.com/mlbio-epfl/turtle)] [[paper](https://arxiv.org/pdf/2406.07236?)]
15. [2025 CVPR]      ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models. [[code](https://github.com/ybendou/ProKeR)] [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf)]


<a name="Optimal_Transport"></a>
## Optimal Transport

 1. [2022 AISTATS]   Sinkformers: Transformers with Doubly Stochastic Attention. [[code](https://github.com/michaelsdr/sinkformers)] [[paper](https://arxiv.org/pdf/2110.11773)]
 2. [2024 ECCV]      OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation. [[code](https://github.com/cubeyoung/OTSeg)] [[paper](https://arxiv.org/pdf/2403.14183)]
 3. [2025 CVPR]      POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation. [[code](https://github.com/jianwang91/POT)] [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)]
 4. [2025 CVPR]      RAM: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Transport. [[code](https://github.com/EricTan7/RAM)] [[paper](https://arxiv.org/pdf/2503.15337)]
 5. [2022 NeurIPS]   SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. [[code](https://github.com/facebookresearch/swav)] [[paper](https://arxiv.org/pdf/2006.09882)]
 6. [2023 ICLR]      PLOT: Prompt Learning with Optimal Transport for Vision-Language Models. [[code](https://github.com/CHENGY12/PLOT)] [[paper](https://arxiv.org/pdf/2210.01253)]
 7. [2024 NeurIPS]   OTTER: Effortless Label Distribution Adaptation of Zero-shot Models. [[code](https://github.com/sprocketlab/otter)] [[paper](https://arxiv.org/pdf/2404.08461)]
 8. [2025 ICCV]      LaZSL: Intrepretable Zero-Shot Learning with Locally-Aligned Vision-Language Model. [[code](https://github.com/shiming-chen/LaZSL)] [[paper](https://arxiv.org/pdf/2506.23822)]
 9. [2025 CVPR]      Conformal Prediction for Zero-Shot Models. [[code](https://github.com/jusiro/CLIP-Conformal)] [[paper](https://arxiv.org/pdf/2505.24693)]
10. [2025 ICML]      ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence. [[code](https://github.com/ghwang-s/abkd)] [[paper](https://arxiv.org/abs/2505.04560)]
11. [2024 ICLR]      EMO: Earth mover distance optimization for auto-regessive language modeling. [[code](https://github.com/DRSY/EMO)] [[paper](https://arxiv.org/pdf/2310.04691)]
12. [2025 CVPR]      RAM: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport. [[code](https://github.com/EricTan7/RAM)] [[paper](https://arxiv.org/abs/2503.15337)]
13. [2025 ICCV]      Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation. [[paper](https://iccv.thecvf.com/virtual/2025/poster/1933)]


<a name="WSSS"></a>
## Weakly Supervised Semantic Segmentation

 1. [2022 CVPR]   Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers. [[code](https://github.com/rulixiang/afa)] [[paper](https://arxiv.org/pdf/2203.02664)]
 2. [2022 CVPR]   MCTFormer:Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation. [[code](https://github.com/xulianuwa/MCTformer)] [[paper](https://arxiv.org/pdf/2203.02891)]
 3. [2023 CVPR]   Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization. [[code](https://github.com/xulianuwa/MMCST)] [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf)]
 4. [2023 ICCV]   Spatial-Aware Token for Weakly Supervised Object Localization. [[code](https://github.com/wpy1999/SAT)] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf)]
 5. [2023 CVPR]   Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation. [[code](https://github.com/ShenghaiRong/BECO?tab=readme-ov-file)] [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf)]
 6. [2023 CVPR]   ToCo:Token Contrast for Weakly-Supervised Semantic Segmentation. [[code](https://github.com/rulixiang/ToCo)] [[paper](https://arxiv.org/pdf/2303.01267)]
 7. [2023 arXiv]  MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation. [[code](https://github.com/xulianuwa/MCTformer)] [[paper](https://arxiv.org/pdf/2308.03005)]
 8. [2024 CVPR]   Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation. [[code](https://github.com/zbf1991/WeCLIP)] [[paper](https://arxiv.org/pdf/2406.11189v1)]
 9. [2024 CVPR]   DuPL: Dual Student with Trustworthy Progressive Learning for RobustWeakly Supervised Semantic Segmentation. [[code](https://github.com/Wu0409/DuPL)] [[paper](https://arxiv.org/pdf/2403.11184)]
10. [2024 CVPR]   Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation. [[code](https://github.com/Barrett-python/CPAL)] [[paper](https://arxiv.org/pdf/2403.07630)]
11. [2024 ECCV]   DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation. [[paper](https://arxiv.org/pdf/2409.15801)]
12. [2024 CVPR]   Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation. [[code](https://github.com/zwyang6/SeCo)] [[paper](https://arxiv.org/pdf/2402.18467)]
13. [2024 ECCV]   CoSa:Weakly Supervised Co-training with Swapping Assignments for Semantic Segmentation. [[code](https://github.com/youshyee/CoSA)] [[paper](https://arxiv.org/pdf/2402.17891)]
14. [2024 IEEE]   SSC:Spatial Structure Constraints for Weakly Supervised Semantic Segmentation. [[code](https://github.com/NUST-Machine-Intelligence-Laboratory/SSC)] [[paper](https://arxiv.org/pdf/2401.11122)]
15. [2024 AAAI]   Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation. [[code](https://github.com/Jessie459/feature-self-reinforcement)] [[paper](https://arxiv.org/pdf/2312.08916)]
16. [2024 CVPR]   Class Tokens Infusion for Weakly Supervised Semantic Segmentation. [[code](https://github.com/yoon307/CTI)] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)]
17. [2024 CVPR]   SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation. [[code](https://github.com/Barrett-python/SFC)] [[paper](https://arxiv.org/pdf/2401.11719)]
18. [2024 CVPR]   PSDPM:Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation. [[code](https://github.com/xinqiaozhao/PSDPM)] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_PSDPM_Prototype-based_Secondary_Discriminative_Pixels_Mining_for_Weakly_Supervised_Semantic_CVPR_2024_paper.pdf)]
19. [2024 arXiv]  A Realistic Protocol for Evaluation of Weakly Supervised Object Localization. [[code](https://github.com/shakeebmurtaza/wsol_model_selection)] [[paper](https://arxiv.org/pdf/2404.10034)]
20. [2025 AAAI]   MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation. [[code](https://github.com/zwyang6/MoRe)] [[paper](https://arxiv.org/pdf/2412.11076)]
21. [2025 CVPR]   PROMPT-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis. [[code](https://github.com/Imageomics/Prompt_CAM)] [[paper](https://arxiv.org/pdf/2501.09333)]
22. [2025 CVPR]   Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation. [[code](https://github.com/zwyang6/ExCEL)] [[paper](https://arxiv.org/pdf/2503.20826)]
23. [2025 CVPR]   GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery. [[code](https://github.com/enguangW/GET)] [[paper](https://arxiv.org/abs/2403.09974)]
24. [2025 arXiv]  TeD-Loc: Text Distillation for Weakly Supervised Object Localization. [[code](https://github.com/shakeebmurtaza/TeDLOC)] [[paper](https://arxiv.org/pdf/2501.12632)]
25. [2025 arXiv]  Image Augmentation Agent for Weakly Supervised Semantic Segmentation. [[paper](https://arxiv.org/pdf/2412.20439)]
26. [2025 CVPR]   Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation. [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)]
27. [2025 CVPRW]  Prompt Categories Cluster for Weakly Supervised Semantic Segmentation. [[paper](https://arxiv.org/pdf/2412.13823)]
28. [2025 Arxiv]  No time to train! Training-Free Reference-Based Instance Segmentation. [[code](https://github.com/miquel-espinosa/no-time-to-train)] [[paper](https://arxiv.org/pdf/2507.02798)]


<a name="Graph_Structure"></a>
## Graph Structure

 1. [2016 AAAI]      The Constrained Laplacian Rank Algorithm for Graph-Based Clustering. [[code](https://github.com/convexfi/spectralGraphTopology/blob/97eee40c8aa30cd62fa01b6e9e01e20197fc7940/R/constrLaplacianRank.R)] [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10302)]
 2. [2016 IJCAI]     Parameter-Free Auto-Weighted Multiple Graph Learning: A Framework for Multiview Clustering and Semi-Supervised Classification. [[paper](https://www.ijcai.org/Proceedings/16/Papers/269.pdf)]
 3. [2023 NeurIPS]   GSLB: The Graph Structure Learning Benchmark. [[code](https://github.com/GSL-Benchmark/GSLB)] [[paper](https://arxiv.org/pdf/2310.05174)]
 4. [2024 AAAI]      Catalyst for Clustering-based Unsupervised Object Re-Identification: Feature Calibration. [[code](https://github.com/lhf12278/FCM-ReID)] [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28092)]
 5. [2025 ICLR]      Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model. [[code](https://github.com/Yushu-Li/ECALP)] [[paper](https://arxiv.org/pdf/2412.18303)]
 6. [network]       [[paper](https://www.gnn.club/?p=2170)]


<a name="VPR"></a>
## Visual Place Recognition

 1. [2022 CVPR]      CosPlace: Rethinking Visual Geo-localization for Large-Scale Applications. [[code](https://github.com/gmberton/CosPlace)] [[paper](https://arxiv.org/pdf/2204.02287)]
 2. [2024 CVPR]      CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition. [[code](https://github.com/Lu-Feng/CricaVPR)] [[paper](https://arxiv.org/pdf/2402.19231)]
 3. [2024 CVPR]      BoQ: A Place is Worth a Bag of Learnable Queries. [[code](https://github.com/amaralibey/Bag-of-Queries)] [[paper](https://arxiv.org/pdf/2405.07364)]
 4. [2024 NeurIPS]   SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition. [[code](https://github.com/Lu-Feng/SuperVLAD)] [[paper](https://openreview.net/pdf?id=bZpZMdY1sj)]
 5. [2024 ECCV]      Revisit Anything: Visual Place Recognition via Image Segment Retrieval. [[code](https://github.com/AnyLoc/Revisit-Anything)] [[paper](https://arxiv.org/pdf/2409.18049)]
 6. [2025 Arxiv]     HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition. [[code](https://github.com/suhan-woo/HypeVPR)] [[paper](https://arxiv.org/pdf/2506.04764)]
 7. [2023 IROS]      Training-Free Attentive-Patch Selection for Visual Place Recognition. [[paper](https://ieeexplore.ieee.org/abstract/document/10342347)]


<a name="Token_fusion"></a>
## Token Mering, Clustering and Pruning

 1. [2021 NeurIPS]   TokenLearner: What Can 8 Learned Tokens Do for Images and Videos? [[code](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)] [[paper](https://arxiv.org/pdf/2106.11297)]
 2. [2022 CVPR]      GroupViT: Semantic Segmentation Emerges from Text Supervision. [[code](https://github.com/NVlabs/GroupViT)] [[paper](https://arxiv.org/pdf/2202.11094)]
 3. [2022 CVPR]      MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation. [[code](https://github.com/xulianuwa/MCTformer)] [[paper](https://arxiv.org/pdf/2308.03005)]
 4. [2023 CVPR]      BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. [[code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)] [[paper](https://arxiv.org/pdf/2301.12597)]
 5. [2023 ICCV]      Perceptual Grouping in Contrastive Vision-Language Models. [[paper](https://arxiv.org/abs/2210.09996)]
 6. [2023 ICLR]      GPVIT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation. [[code](https://github.com/ChenhongyiYang/GPViT)] [[paper](https://arxiv.org/pdf/2212.06795)]
 7. [2023 CVPR]      SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/MendelXu/SAN)] [[paper](https://arxiv.org/pdf/2302.12242)]
 8. [2024 CVPR]      Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers. [[code](https://github.com/mlvlab/MCTF)] [[paper](https://arxiv.org/pdf/2403.10030)]
 9. [2024 CVPR]      Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding. [[code](https://github.com/PKU-YuanGroup/Chat-UniVi)] [[paper](https://arxiv.org/pdf/2311.08046)]
10. [2024 ICLR]      LaVIT: Unified language-vision pretraining in LLM with dynamic discrete visual tokenization. [[code](https://github.com/jy0205/LaVIT)] [[paper](https://arxiv.org/pdf/2309.04669)]
11. [2024 arXiv]     TokenPacker: Efficient Visual Projector for Multimodal LLM. [[code](https://github.com/CircleRadon/TokenPacker)] [[paper](https://arxiv.org/abs/2407.02392)]
12. [2024 arXiv]     DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models. [[code](https://github.com/yaolinli/DeCo)] [[paper](https://arxiv.org/pdf/2405.20985)]
13. [2024 CVPR]      Grounding Everything: Emerging Localization Properties in Vision-Language Transformers. [[code](https://github.com/WalBouss/GEM)] [[paper](https://arxiv.org/pdf/2312.00878)]
14. [2025 CVPR]      PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models. [[code](https://github.com/orailix/PACT/tree/main)] [[paper](https://arxiv.org/pdf/2504.08966)]


<a name="Segmentation_and_Detection"></a>
## Segmentation and Detection

 1. [2015 CVPR]      FCN: Fully Convolutional Networks for Semantic Segmentation. [[code](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)] [[paper](https://arxiv.org/abs/1411.4038)]
 2. [2016 MICCAI]    UNet: Convolutional Networks for Biomedical Image Segmentation. [[paper](https://arxiv.org/pdf/1505.04597)]
 3. [2017 arXiv]     DeepLabV3: Rethinking atrous convolution for semantic image segmentation. [[paper](https://arxiv.org/pdf/1706.05587)]
 4. [2018 CVPR]      DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. [[paper](https://arxiv.org/pdf/1802.02611)]
 5. [2019 CVPR]      Semantic FPN: Panoptic Feature Pyramid Networks. [[paper](https://arxiv.org/pdf/1901.02446)]
 6. [2021 CVPR]      SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers. [[code](https://github.com/fudan-zvg/SETR)] [[paper](https://arxiv.org/pdf/2012.15840)]
 7. [2021 ICCV]      Segmenter: Transformer for Semantic Segmentation. [[code](https://github.com/rstrudel/segmenter)] [[paper](https://arxiv.org/pdf/2105.05633)]
 8. [2021 NeurIPS]   SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. [[code](https://github.com/NVlabs/SegFormer)] [[paper](https://arxiv.org/pdf/2105.15203)]
 9. [2021 CVPR]      MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation. [[code](https://github.com/facebookresearch/MaskFormer)] [[paper](https://arxiv.org/pdf/2107.06278)]
10. [2022 CVPR]      Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation. [[code](https://github.com/facebookresearch/Mask2Former)] [[paper](https://arxiv.org/pdf/2112.01527)]
11. [2024 CVPR]      Rein: Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation. [[code](https://github.com/w1oves/Rein)] [[paper](https://arxiv.org/pdf/2312.04265)]
12. [2015 NeurIPS]   Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. [[paper](https://arxiv.org/pdf/1506.01497)]
13. [2020 ECCV]      DETR: End-to-End Object Detection with Transformers. [[code](https://github.com/facebookresearch/detr)] [[paper](https://arxiv.org/pdf/2005.12872)]
14. [2021 ICLR]      Deformable DETR: Deformable Transformers for End-to-End Object Detection. [[code](https://github.com/fundamentalvision/Deformable-DETR)] [[paper](https://arxiv.org/pdf/2010.04159)]
15. [2023 ICLR]      DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. [[code](https://github.com/IDEA-Research/DINO)] [[paper](https://arxiv.org/pdf/2203.03605)]


<a name="Backbone"></a>
## Backbone

 1. [2017 NeurIPS]   transfomer: Attention Is All You Need. [[code](https://github.com/tensorflow/tensor2tensor)] [[paper](https://arxiv.org/pdf/1706.03762)]
 2. [2021 ICLR]      ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. [[code](https://github.com/google-research/vision_transformer)] [[paper](https://arxiv.org/pdf/2010.11929)]
 3. [2021 ICML]      DeiT: Training data-efficient image transformers & distillation through attention. [[paper](https://arxiv.org/pdf/2012.12877)]
 4. [2021 ICCV]      Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. [[code](https://github.com/microsoft/Swin-Transformer)] [[paper](https://arxiv.org/pdf/2103.14030)]
 5. [2021 NeurIPS]   Twins: Revisiting the Design of Spatial Attention in Vision Transformers. [[code](https://github.com/Meituan-AutoML/Twins)] [[paper](https://arxiv.org/pdf/2104.13840)]
 6. [2022 CVPR]      Hyperbolic Vision Transformers: Combining Improvements in Metric Learning. [[code](https://github.com/htdt/hyp_metric)] [[paper](https://arxiv.org/pdf/2203.10833)]
 7. [2022 ICLR]      BEiT: BERT Pre-Training of Image Transformers. [[code](https://github.com/microsoft/unilm/tree/master/beit)] [[paper](https://arxiv.org/pdf/2106.08254)]
 8. [2022 CVPR]      MAE: Masked Autoencoders Are Scalable Vision Learners. [[code](https://github.com/facebookresearch/mae)] [[paper](https://arxiv.org/pdf/2111.06377)]
 9. [2022 CVPR]      PoolFormer: MetaFormer is Actually What You Need for Vision. [[code](https://github.com/sail-sg/poolformer)] [[paper](https://arxiv.org/pdf/2111.11418)]
10. [2022 NeurIPS]   SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation. [[code](https://github.com/Visual-Attention-Network/SegNeXt)] [[paper](https://arxiv.org/pdf/2209.08575)]
11. [2023 ICCV]      OpenSeeD: A simple framework for open-vocabulary segmentation and detection. [[code](https://github.com/IDEA-Research/OpenSeeD)] [[paper](https://arxiv.org/pdf/2303.08131)]
12. [2023 arXiv]     SAM: Segment Anything. [[code](https://github.com/facebookresearch/segment-anything)] [[demo](https://segment-anything.com/demo)] [[paper](https://arxiv.org/pdf/2304.02643)]
13. [2024 arXiv]     SAM 2: Segment Anything in Images and Videos. [[code](https://github.com/facebookresearch/sam2)] [[demo](https://sam2.metademolab.com/)] [[paper](https://arxiv.org/pdf/2408.00714)]


<a name="CLIP"></a>
## CLIP

 1. [2021 ICML]      CLIP: Learning transferable visual models from natural language supervision. [[code](https://github.com/OpenAI/CLIP)] [[paper](https://arxiv.org/pdf/2103.00020)]
 2. [2022 IJCV]      CoOp: Learning to Prompt for Vision-Language Models. [[code](https://github.com/KaiyangZhou/CoOp)] [[paper](https://arxiv.org/pdf/2109.01134)]
 3. [2022 ECCV]      VPT: Visual Prompt Tuning. [[code](https://github.com/kmnp/vpt)] [[paper](https://arxiv.org/pdf/2203.12119)]
 4. [2022 ICLR]      LoRA: Low-Rank Adaptation of Large Language Models. [[code](https://github.com/microsoft/LoRA)] [[paper](https://arxiv.org/pdf/2106.09685)]
 5. [2022 NeurIPS]   TPT: Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models. [[code](https://github.com/azshue/TPT)] [[paper](https://arxiv.org/pdf/2209.07511)]
 6. [2022 Arxiv]     UPL: Unsupervised Prompt Learning for Vision-Language Models. [[code](https://github.com/tonyhuang2022/UPL)] [[paper](https://arxiv.org/pdf/2204.03649)]
 7. [2022 Arxiv]     CLIPPR: Improving Zero-Shot Models with Label Distribution Priors. [[code](https://github.com/jonkahana/CLIPPR)] [[paper](https://arxiv.org/pdf/2212.00784)]
 8. [2022 CVPR]      CoCoOp: Conditional Prompt Learning for Vision-Language Models. [[code](https://github.com/KaiyangZhou/CoOp)] [[paper](https://arxiv.org/pdf/2203.05557)]
 9. [2023 CVPR]      TaskRes: Task Residual for Tuning Vision-Language Models. [[code](https://github.com/geekyutao/TaskRe)] [[paper](https://arxiv.org/pdf/2211.10277)]
10. [2023 ICML]      POUF: Prompt-Oriented Unsupervised Fine-tuning for Large Pre-trained Models. [[code](https://github.com/korawat-tanwisuth/POUF)] [[paper](https://arxiv.org/pdf/2305.00350)]
11. [2023 NeurIPS]   Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning. [[code](https://github.com/BatsResearch/menghini-neurips23-code)] [[paper](https://arxiv.org/pdf/2306.01669)]
12. [2023 NeurIPS]   LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections. [[code](https://github.com/jmiemirza/LaFTer)] [[paper](https://arxiv.org/pdf/2305.18287)]
13. [2023 PRCV]      Unsupervised Prototype Adapter for Vision-Language Models. [[paper](https://arxiv.org/pdf/2308.11507)]
14. [2023 IJCV]      CLIP-Adapter: Better Vision-Language Models with Feature Adapters. [[code](https://github.com/gaopengcuhk/CLIP-Adapter)] [[paper](https://arxiv.org/pdf/2110.04544)]
15. [2024 CVPR]      CODER: Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification. [[code](https://github.com/YCaigogogo/CODER)] [[paper](https://arxiv.org/pdf/2404.17753)]
16. [2024 CVPR]      LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP. [[code](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline)] [[paper](https://arxiv.org/pdf/2404.02285)]
17. [2024 CVPR]      PromptKD: Unsupervised Prompt Distillation for Vision-Language Models. [[code](https://github.com/zhengli97/PromptKD)] [[paper](https://arxiv.org/pdf/2403.02781)]
18. [2024 CVPR]      Transfer CLIP for Generalizable Image Denoising. [[code](https://github.com/alwaysuu/CLIPDenoising)] [[paper](https://arxiv.org/pdf/2403.15132)]
19. [2024 ECCV]      BRAVE: Broadening the visual encoding of vision-language model. [[paper](https://arxiv.org/pdf/2404.07204)]
20. [2024 ICML]      Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data. [[code](https://github.com/vanillaer/CPL-ICML2024)] [[paper](https://arxiv.org/pdf/2406.10502)]
21. [2024 CVPR]      CLIP-KD: An Empirical Study of CLIP Model Distillation. [[code](https://github.com/winycg/CLIP-KD)] [[paper](https://arxiv.org/pdf/2307.12732)]
22. [2025 WACV]      DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models. [[code](https://github.com/Externalhappy/DPA)] [[paper](https://arxiv.org/pdf/2408.08855)]
23. [2025 WACV]      Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models. [[code](https://github.com/elaine-sui/TPS)] [[paper](https://arxiv.org/pdf/2403.12952)]
24. [2025 WACV]      LATTECLIP: Unsupervised CLIP Fine-Tuning via LMM-Synthetic Texts. [[code](https://github.com/astra-vision/LatteCLIP)] [[paper](https://arxiv.org/pdf/2410.08211)]
25. [2025 ICLR]      CROSS THE GAP: EXPOSING THE INTRA-MODAL MISALIGNMENT IN CLIP VIA MODALITY INVERSION. [[code](https://github.com/miccunifi/Cross-the-Gap)] [[paper](https://arxiv.org/pdf/2502.04263)]
26. [2025 ICLR]      CLIP’s Visual Embedding Projector is a Few-shot Cornucopia. [[code](https://github.com/astra-vision/ProLIP)] [[paper](https://arxiv.org/pdf/2410.05270)]
27. [2025 CVPR]      DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers. [[code](https://github.com/Noahsark/DA-VPT)] [[paper](https://arxiv.org/pdf/2505.23694)]
28. [2025 ICML]      Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models. [[code](https://github.com/peterant330/KUEA)] [[paper](https://arxiv.org/pdf/2506.02557)]
29. [2025 CVPR]      Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification. [[code](https://github.com/k0u-id/CCD)] [[paper](https://arxiv.org/pdf/2503.16873)]
30. [2024 CVPR]      Multi-Modal Adapter for Vision-Language Models. [[code](https://github.com/ZjjConan/VLM-MultiModalAdapter)] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf)]


<a name="open_segmentation"></a>
## Open vocabulary segmentation
### segmentation

 1. [2022 ICLR]   Lseg: Language-driven semantic segmentation (Supervised). [[code](https://github.com/isl-org/lang-seg)] [[paper](https://arxiv.org/pdf/2201.03546)]
 2. [2022 CVPR]   ZegFormer: Decoupling Zero-Shot Semantic Segmentation. [[code](https://github.com/dingjiansw101/ZegFormer)] [[paper](https://arxiv.org/pdf/2112.07910)]
 3. [2022 ECCV]   MaskCLIP+: Extract Free Dense Labels from CLIP. [[code](https://github.com/chongzhou96/MaskCLIP)] [[paper](https://arxiv.org/pdf/2112.01071)]
 4. [2022 ECCV]   ViL-Seg: Open-World Semantic Segmentation via Contrasting and Clustering Vision-Language Embeddings. [[paper](https://arxiv.org/pdf/2207.08455v2)]
 5. [2022 CVPR]   GroupViT: Semantic Segmentation Emerges from Text Supervision (Open-Vocabulary Zero-Shot). [[code](https://github.com/NVlabs/GroupViT)] [[paper](https://arxiv.org/pdf/2202.11094)]
 6. [2022 ECCV]   OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels. [[paper](https://arxiv.org/pdf/2112.12143)]
 7. [2023 CVPR]   FreeSeg: Unified, Universal, and Open-Vocabulary Image Segmentation. [[code](https://github.com/bytedance/FreeSeg)] [[paper](https://arxiv.org/pdf/2303.17225)]
 8. [2023 ICML]   SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation (Zero-Shot). [[code](https://github.com/ArrowLuo/SegCLIP)] [[paper](https://arxiv.org/pdf/2211.14813)]
 9. [2023 CVPR]   ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation. [[code](https://github.com/ZiqinZhou66/ZegCLIP)] [[paper](https://arxiv.org/pdf/2212.03588)]
10. [2023 CVPR]   X-Decoder: Generalized Decoding for Pixel, Image, and Language. [[code](https://github.com/microsoft/X-Decoder/tree/main)] [[paper](https://arxiv.org/pdf/2212.11270)]
11. [2023 CVPR]   ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models. [[code](https://github.com/NVlabs/ODISE)] [[paper](https://arxiv.org/pdf/2303.04803)]
12. [2023 ICML]   MaskCLIP: Open-Vocabulary Universal Image Segmentation with MaskCLIP. [[code](https://github.com/mlpc-ucsd/MaskCLIP)] [[paper](https://arxiv.org/pdf/2208.08984)]
13. [2023 CVPR]   SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/MendelXu/SAN)] [[paper](https://arxiv.org/pdf/2302.12242)]
14. [2024 ECCV]   CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation. [[code](https://github.com/wysoczanska/clip_dinoiser)] [[paper](https://arxiv.org/pdf/2312.12359)]
15. [2024 CVPR]   SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation. [[code](https://github.com/xb534/SED)] [[paper](https://arxiv.org/pdf/2311.15537)]
16. [2024 TPAMI]  Review: Towards Open Vocabulary Learning: A Survey. [[code](https://github.com/jianzongwu/Awesome-Open-Vocabulary)] [[paper](https://arxiv.org/pdf/2306.15880)]
17. [2025 ICCV]   Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction. [[code](https://github.com/HVision-NKU/DenseVLM)] [[paper](https://arxiv.org/abs/2412.06244)]
18. [2024 CVPR]   Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation. [[code](https://github.com/Jittor/JSeg)] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Exploring_Regional_Clues_in_CLIP_for_Zero-Shot_Semantic_Segmentation_CVPR_2024_paper.pdf)]
19. [2025 CVPR]   DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception. [[code](https://github.com/xiaomoguhz/DeCLIP)] [[paper](https://arxiv.org/pdf/2505.04410)]

### object detection

 1. [2021 CVPR]      Open-Vocabulary Object Detection Using Captions. [[code](https://github.com/alirezazareian/ovr-cnn)] [[paper](https://arxiv.org/pdf/2011.10678)]
 2. [2022 ICLR]      ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation. [[code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)] [[paper](https://arxiv.org/pdf/2104.13921)]
 3. [2022 CVPR]      GLIP: Grounded Language-Image Pre-training. [[code](https://github.com/microsoft/GLIP)] [[paper](https://arxiv.org/pdf/2112.03857)]
 4. [2022 NeurIPS]   GLIPv2: Unifying Localization and Vision-Language Understanding. [[code](https://github.com/microsoft/GLIP)] [[paper](https://arxiv.org/pdf/2206.05836)]


<a name="Other"></a>
## Other Technologies

 1. [2016 CVPRW]    pixel shuffle. [[paper](https://arxiv.org/pdf/1609.05158)]
 2. [2020 NeurIPS]   DDPM: Denoising Diffusion Probabilistic Models. [[code](https://github.com/hojonathanho/diffusion)] [[paper](https://arxiv.org/pdf/2006.11239)]
