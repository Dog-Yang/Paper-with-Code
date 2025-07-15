# Content
[[Training Free Segmentation](#Training_Free)]            
[[Zero shot Classification/Test Time Adaptation](#Zero_shot_classification)]      
[[Optimal Transport](#Optimal_Transport)]   

[[CLIP](#CLIP)]           
[[Visual Place Recognition](#VPR)]         
[[Token Mering, Clustering and Pruning](#Token_fusion)]           
[[Backbone](#Backbone)]           

[[Weakly Supervised Semantic Segmentation](#Weakly_Supervised_Semantic_Segmentation)]           
[[Open vocabulary segmentation](#open_segmentation)]           
[[Open vocabulary detection](#open_detection)]           
[[Classical segmentation](#Classical_Segmentation)]           
[[Classical detection](#Model-Classical_detection)]     
[[Other](#Other)]          


<a name="Training_Free"></a>
# Training Free Segmentation
VLM Only    
[[2022 ECCV](https://arxiv.org/pdf/2112.01071)]  [[code](https://github.com/chongzhou96/MaskCLIP)] Maskclip: Extract Free Dense Labels from CLIP    
[[2022 NeurIPS](https://arxiv.org/pdf/2206.07045)] [[code](https://github.com/NoelShin/reco)] ReCo: Retrieve and Co-segment for Zero-shot Transfer    
[[2023 Arxiv](https://arxiv.org/pdf/2304.05653)] [[code](https://github.com/xmed-lab/CLIP_Surgery)] CLIPSurgery: CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks       
[[2024 ECCV](https://arxiv.org/pdf/2312.01597)]  [[code](https://github.com/wangf3014/SCLIP)] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference    
[[2024 WACV](https://arxiv.org/pdf/2404.08181)]  [[code](https://github.com/sinahmr/NACLIP)] NACLIP: Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation       
[[2024 Arxiv](https://arxiv.org/pdf/2411.15869)] [[code](https://github.com/SuleBai/SC-CLIP)] SC-CLIP: Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation    
[[2025 CVPR](https://arxiv.org/pdf/2411.15851)]  [[code](https://github.com/yvhangyang/ResCLIP)] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference    
[[2024 ECCV](https://arxiv.org/pdf/2407.12442)]  [[code](https://github.com/mc-lan/ClearCLIP)] ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference         
[[2025 AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32602)] [[code](https://ojs.aaai.org/index.php/AAAI/article/view/32602)] Unveiling the Knowledge of CLIP for Training-Free Open-Vocabulary Semantic Segmentation      
[[2025 CVPRW](https://arxiv.org/pdf/2411.12044)] [[code](https://github.com/m-arda-aydn/ITACLIP)] ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements    
[[2024 ECCV](https://arxiv.org/pdf/2407.08268)]  [[code](https://github.com/leaves162/CLIPtrase)] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation    
[[2025.05 Arxiv](https://arxiv.org/pdf/2505.22209)] A Survey on Training-free Open-Vocabulary Semantic Segmentation         

***************************************************************
VLM & VFM & Diffusion & SAM     
[[2024 ECCV](https://arxiv.org/pdf/2408.04883v1)] [[code](https://github.com/mc-lan/ProxyCLIP)] ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation    
[[2024 ECCV](https://arxiv.org/pdf/2312.12359)] [[code](https://github.com/wysoczanska/clip_dinoiser)] CLIP_Dinoiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation    
[[2025 CVPR](https://arxiv.org/pdf/2503.19777)] [[code](https://github.com/vladan-stojnic/LPOSS/tree/main)] LOPSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation    
[[2025 ICCV](https://arxiv.org/pdf/2504.10487)] [[code](https://github.com/yasserben/FLOSS)] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation    
[[2024 Arxiv](https://arxiv.org/pdf/2411.13836)] [[code](https://github.com/linsun449/cliper.code)] CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation    
[[2024 WACV](https://arxiv.org/pdf/2309.14289)] [[code](https://github.com/wysoczanska/clip-diy)] CLIP-DIY: CLIP Dense Inference Yields Open-Vocabulary Semantic Segmentation For-Free       
[[2024 NeurIPS](https://arxiv.org/pdf/2406.02842v2)] [[code](https://github.com/PaulCouairon/DiffCut)] DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut      
[[2025 NeurIPS](https://arxiv.org/pdf/2505.23769)] [[code](https://github.com/avaxiao/TextRegion)] TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models         
[[2025 ICCV](https://arxiv.org/pdf/2411.09219)] [[code](https://github.com/YuHengsss/Trident)] Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation         
[[2024 CVPR](https://arxiv.org/pdf/2404.06542)] [[code](https://github.com/aimagelab/freeda)] Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation         
[[2025 CVPR]()] [[code]()]          

***************************************************************

[[2024 IJCV](https://arxiv.org/pdf/2310.10912)] [[code](https://github.com/luckybird1994/IPSeg)] Towards Training-free Open-world Segmentation via Image Prompting Foundation Models       
[[2024 CVPR](https://arxiv.org/pdf/2312.00878)] [[code](https://github.com/WalBouss/GEM)] Grounding Everything: Emerging Localization Properties in Vision-Language Transformers       
[[2024 WACV](https://openaccess.thecvf.com/content/WACV2024/papers/Barsellotti_FOSSIL_Free_Open-Vocabulary_Semantic_Segmentation_Through_Synthetic_References_Retrieval_WACV_2024_paper.pdf)] FOSSIL: Free Open-Vocabulary Semantic Segmentation through Synthetic References Retrieval       
[[2024 AAAI](https://arxiv.org/pdf/2312.12828)] [[code](https://github.com/linyq2117/TagCLIP)] TagCLIP: A Local-to-Global Framework to Enhance Open-VocabularyMulti-Label Classification of CLIP Without Training       
[[2024 ICLR](https://arxiv.org/pdf/2401.11739)] [[code](https://github.com/linyq2117/TagCLIP)] EmerDiff: Emerging Pixel-level Semantic Knowledge in Diffusion Models           
[[2024 CVPR](https://arxiv.org/pdf/2311.17095)] [[code](https://github.com/letitiabanana/PnP-OVSS)] Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models       
[[2024 CVPR](https://arxiv.org/pdf/2312.07661)] [[code](https://github.com/kevin-ssy/CLIP_as_RNN)] CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor       
[[2024 CVPR](https://arxiv.org/pdf/2404.00262)] [[code](https://github.com/072jiajia/image-text-co-decomposition)] Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation       
[[2024 ECCV](https://arxiv.org/pdf/2407.08268)] [[code](https://github.com/leaves162/CLIPtrase)] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation       
[[2024 ECCV](https://arxiv.org/pdf/2408.04961)] [[code](https://github.com/dahyun-kang/lavg)] In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation        
[[2025 ICCV](https://arxiv.org/pdf/2411.10086)] [[code](https://github.com/zdk258/CorrCLIP/tree/master)] CorrCLIP: Reconstructing Correlations in CLIP with Off-the-Shelf Foundation Models for Open-Vocabulary Semantic Segmentation       
[[2025 ICCV](https://arxiv.org/pdf/2408.04883)] [[code](https://github.com/lorebianchi98/Talk2DINO)] Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation       
[[2025 CVPR](https://arxiv.org/pdf/2411.17150)] [[code](https://micv-yonsei.github.io/cass/)] Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation       

<a name="Zero_shot_classification"></a>
# Zero shot Classification / Test Time Adaptation
[[2024 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf)] [[code](https://github.com/AI4LIFE-GROUP/SpLiCE)] SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings    
[[2024 NeurIPS](https://arxiv.org/pdf/2406.01837)] [[code](https://github.com/MaxZanella/transduction-for-vlms)] Transclip: Boosting Vision-Language Models with Transduction     
[[2025 CVPR](https://arxiv.org/pdf/2501.03729)] [[code](https://github.com/MaxZanella/StatA)] Realistic Test-Time Adaptation of Vision-Language Models     
[[2023 AAAI](https://arxiv.org/pdf/2209.14169)] [[code](https://github.com/ZiyuGuo99/CALIP)] CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention      
[[2025 AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32534)] [[code](https://github.com/lyymuwu/TIMO)] TIMO: Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP     
[[2025 CVPR](https://arxiv.org/pdf/2503.23388)] [[code](https://github.com/hf618/COSMIC)] COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation   
[[2024 CVPR](https://arxiv.org/pdf/2405.18437)] [[code](https://github.com/SegoleneMartin/transductive-CLIP)] Transductive Zero-Shot and Few-Shot CLIP      
[[2023 CVPR](https://arxiv.org/pdf/2111.03930)] [[code](https://github.com/gaopengcuhk/Tip-Adapter)] Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling      
[[2024 ICLR](https://arxiv.org/pdf/2402.04087)] [[code](https://github.com/mrflogs/ICLR24)] GDA-CLIP: A hard-to-beat baseline for training-free clip-based adaptation     
***************************************************************
[[2024 CVPR](https://arxiv.org/pdf/2405.02266)] [[code](https://github.com/MaxZanella/MTA)] On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?   
[[2024 NeurIPS](https://arxiv.org/pdf/2405.18330)] [[code](https://github.com/FarinaMatteo/zero)] Frustratingly Easy Test-Time Adaptation of Vision-Language Models      
[[2024 CVPR](https://arxiv.org/pdf/2403.18293)] [[code](https://github.com/kdiAAA/TDA)] TDA: Efficient Test-Time Adaptation of Vision-Language Model   
[[2024 NeurIPS](https://arxiv.org/pdf/2410.15430)] [[code](https://github.com/taolinzhang/BoostAdapter)] BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping      
[[2024 CVPR](https://arxiv.org/pdf/2403.17589)] [[code](https://github.com/YBZh/DMN)] DMN: Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models     
[[2023 ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf)] [[code](https://github.com/miccunifi/SEARLE.)] Zero-Shot Composed Image Retrieval with Textual Inversion   
[[2025 CVPR](https://arxiv.org/pdf/2412.11077)] [[code](https://github.com/Pter61/osrcir)] Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval   
[[2024 NeurIPS](https://arxiv.org/pdf/2410.19294)] [[code](https://github.com/zhuhsingyuu/Frolic)] Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting  
[[2025 ICML](https://arxiv.org/pdf/2505.13233)] [[code](https://github.com/BIT-DA/ABS)] From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection      
[[2025 CVPR](https://arxiv.org/pdf/2308.01313)] [[code](https://github.com/umd-huang-lab/perceptionCLIP)] PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts   
[[2024 CVPR](https://arxiv.org/pdf/2404.04072)] [[code](https://github.com/vladan-stojnic/ZLaP)] ZLaP: Label Propagation for Zero-shot Classification with Vision-Language Models         
[[2025 CVPRW](https://arxiv.org/pdf/2503.12206)] [[code](https://github.com/ans92/TLAC)] TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification         
[[2023 NeurIPS](https://arxiv.org/pdf/2310.19752)] [[code](https://github.com/idstcv/InMaP)] Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP      
[[2024 ICML](https://arxiv.org/pdf/2406.07236?)] [[code](https://github.com/mlbio-epfl/turtle)] Let Go of Your Labels with Unsupervised Transfer      

# Optimal Transport
<a name="Optimal_Transport"></a>
[[2022 AISTATS](https://arxiv.org/pdf/2110.11773)] [[code](https://github.com/michaelsdr/sinkformers)] Sinkformers: Transformers with Doubly Stochastic Attention      
[[2024 ECCV](https://arxiv.org/pdf/2403.14183)] [[code](https://github.com/cubeyoung/OTSeg)] OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation    
[[2025 CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)] [[code](https://github.com/jianwang91/POT)] POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation    
[[2025 CVPR](https://arxiv.org/pdf/2503.15337)] [[code](https://github.com/EricTan7/RAM)] Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained  Transport     
[[2022 NeurIPS](https://arxiv.org/pdf/2006.09882)] [[code](https://github.com/facebookresearch/swav)] SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments         
[[2023 ICRL](https://arxiv.org/pdf/2210.01253)] [[code](https://github.com/CHENGY12/PLOT)] PLOT: Prompt Learning with Optimal Transport for Vision-Language Models         
[[2024 NeurIPS](https://arxiv.org/pdf/2404.08461)] [[code](https://github.com/sprocketlab/otter)] OTTER: Effortless Label Distribution Adaptation of Zero-shot Models    
[[2025 ICCV](https://arxiv.org/pdf/2506.23822)] [[code](https://github.com/shiming-chen/LaZSL)] Intrepretable Zero-Shot Learning with Locally-Aligned Vision-Language Model     
[[2025 CVPR](https://arxiv.org/pdf/2505.24693)] [[code](https://github.com/jusiro/CLIP-Conformal)] Conformal Prediction for Zero-Shot Models     
[[2025 ICML](https://arxiv.org/abs/2505.04560)] [[code](https://github.com/ghwang-s/abkd)] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence    

<a name="Weakly_Supervised_Semantic_Segmentation"></a>
# Weakly Supervised Semantic Segmentation
[[2022 CVPR](https://arxiv.org/pdf/2203.02664)] [[code](https://github.com/rulixiang/afa)] Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers     
[[2022 CVPR](https://arxiv.org/pdf/2203.02891)] [[code](https://github.com/xulianuwa/MCTformer)] MCTFormer:Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation     
[[2023 CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf)] [[code](https://github.com/xulianuwa/MMCST)] Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization      
[[2023 ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf)] [[code](https://github.com/wpy1999/SAT)] Spatial-Aware Token for Weakly Supervised Object Localization     
[[2023 CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf)] [[code](https://github.com/ShenghaiRong/BECO?tab=readme-ov-file)] Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation       
[[2023 CVPR](https://arxiv.org/pdf/2303.01267)] [[code](https://github.com/rulixiang/ToCo)] ToCo:Token Contrast for Weakly-Supervised Semantic Segmentation     
[[2023 arXiv](https://arxiv.org/pdf/2308.03005)] [[code](https://github.com/xulianuwa/MCTformer)] MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation     
[[2024 CVPR](https://arxiv.org/pdf/2406.11189v1)] [[code](https://github.com/zbf1991/WeCLIP)] Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation     
[[2024 CVPR](https://arxiv.org/pdf/2403.11184)] [[code](https://github.com/Wu0409/DuPL)] DuPL: Dual Student with Trustworthy Progressive Learning for RobustWeakly Supervised Semantic Segmentation     
[[2024 CVPR](https://arxiv.org/pdf/2403.07630)] [[code](https://github.com/Barrett-python/CPAL)] Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation     
[[2024 ECCV](https://arxiv.org/pdf/2409.15801)] DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation     
[[2024 CVPR](https://arxiv.org/pdf/2402.18467)] [[code](https://github.com/zwyang6/SeCo)] Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation     
[[2024 ECCV](https://arxiv.org/pdf/2402.17891)] [[code](https://github.com/youshyee/CoSA)] CoSa:Weakly Supervised Co-training with Swapping Assignments for Semantic Segmentation     
[[2024 IEEE](https://arxiv.org/pdf/2401.11122)] [[code](https://github.com/NUST-Machine-Intelligence-Laboratory/SSC)] SSC:Spatial Structure Constraints for Weakly Supervised Semantic Segmentation     
[[2024 AAAI](https://arxiv.org/pdf/2312.08916)] [[code](https://github.com/Jessie459/feature-self-reinforcement)] Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation     
[[2024 CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf)] [[code](https://github.com/yoon307/CTI)] Class Tokens Infusion for Weakly Supervised Semantic Segmentation   
[[2024 CVPR](https://arxiv.org/pdf/2401.11719)] [[code](https://github.com/Barrett-python/SFC)] SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation     
[[2024 CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_PSDPM_Prototype-based_Secondary_Discriminative_Pixels_Mining_for_Weakly_Supervised_Semantic_CVPR_2024_paper.pdf)] [[code](https://github.com/xinqiaozhao/PSDPM)] PSDPM:Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation     
[[2024 arXiv](https://arxiv.org/pdf/2404.10034)] [[code](https://github.com/shakeebmurtaza/wsol_model_selection)] A Realistic Protocol for Evaluation of Weakly Supervised Object Localization     
[[2025 AAAI](https://arxiv.org/pdf/2412.11076)] [[code](https://github.com/zwyang6/MoRe)] MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation     
[[2025 CVPR](https://arxiv.org/pdf/2501.09333)] [[code](https://github.com/Imageomics/Prompt_CAM)] PROMPT-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis     
[[2025 CVPR](https://arxiv.org/pdf/2503.20826)] [[code](https://github.com/zwyang6/ExCEL)] Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation     
[[2025 CVPR](https://arxiv.org/abs/2403.09974)] [[code](https://github.com/enguangW/GET)] GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery     
[[2025 arXiv](https://arxiv.org/pdf/2501.12632)] [[code](https://github.com/shakeebmurtaza/TeDLOC)] TeD-Loc: Text Distillation for Weakly Supervised Object Localization     
[[2025 arXiv](https://arxiv.org/pdf/2412.20439)] Image Augmentation Agent for Weakly Supervised Semantic Segmentation     
[[2025 CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)] Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation      
[[2025 CVPRW](https://arxiv.org/pdf/2412.13823)] Prompt Categories Cluster for Weakly Supervised Semantic Segmentation    

<a name="Graph_Structure"></a>
# Graph Structure
[[2016 AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10302)] The Constrained Laplacian Rank Algorithm for Graph-Based Clustering    
[[2016 IJCAI](https://www.ijcai.org/Proceedings/16/Papers/269.pdf)] Parameter-Free Auto-Weighted Multiple Graph Learning: A Framework for Multiview Clustering and Semi-Supervised Classification   
[[2023 NeurIPS](https://arxiv.org/pdf/2310.05174)] [[code](https://github.com/GSL-Benchmark/GSLB)] GSLB: The Graph Structure Learning Benchmark     
[[2024 AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28092)] [[code](https://github.com/lhf12278/FCM-ReID)] Catalyst for Clustering-based Unsupervised Object Re-Identification: Feature Calibration    
[[2025 ICLR](https://arxiv.org/pdf/2412.18303)] [[code](https://github.com/Yushu-Li/ECALP)] Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model    
[[network](https://www.gnn.club/?p=2170)] [[code]()] 

<a name="VPR"></a>
# Visual Place Recognition
[[2022 CVPR](https://arxiv.org/pdf/2204.02287)] [[code](https://github.com/gmberton/CosPlace)] CosPlace: Rethinking Visual Geo-localization for Large-Scale Applications   
[[2024 CVPR](https://arxiv.org/pdf/2402.19231)] [[code](https://github.com/Lu-Feng/CricaVPR)] CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition    
[[2024 CVPR](https://arxiv.org/pdf/2405.07364)] [[code](https://github.com/amaralibey/Bag-of-Queries)] BoQ: A Place is Worth a Bag of Learnable Queries    
[[2024 NeurIPS](https://openreview.net/pdf?id=bZpZMdY1sj)] [[code](https://github.com/Lu-Feng/SuperVLAD)] SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition    
[[2024 ECCV](https://arxiv.org/pdf/2409.18049)] [[code](https://github.com/AnyLoc/Revisit-Anything)] Revisit Anything: Visual Place Recognition via Image Segment Retrieval      
[[2025 Arxiv](https://arxiv.org/pdf/2506.04764)] [[code](https://github.com/suhan-woo/HypeVPR)] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition  

<a name="Token_fusion"></a>
# Token Mering, Clustering and Pruning
[[2021 NeurIPS](https://arxiv.org/pdf/2106.11297)] [[code](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)] TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?   
[[2022 CVPR](https://arxiv.org/pdf/2202.11094)] [[code](https://github.com/NVlabs/GroupViT)] GroupViT: Semantic Segmentation Emerges from Text Supervision      
[[2022 CVPR](https://arxiv.org/pdf/2308.03005)] [[code](https://github.com/xulianuwa/MCTformer)] MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation    
[[2023 CVPR](https://arxiv.org/pdf/2301.12597)] [[code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models    
[[2023 ICCV](https://arxiv.org/abs/2210.09996)] Perceptual Grouping in Contrastive Vision-Language Models    
[[2023 ICLR](https://arxiv.org/pdf/2212.06795)] [[code](https://github.com/ChenhongyiYang/GPViT)] GPVIT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation    
[[2023 CVPR](https://arxiv.org/pdf/2302.12242)] [[code](https://github.com/MendelXu/SAN)] SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation    
[[2024 CVPR](https://arxiv.org/pdf/2403.10030)] [[code](https://github.com/mlvlab/MCTF)] Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers  
[[2024 CVPR](https://arxiv.org/pdf/2311.08046)] [[code](https://github.com/PKU-YuanGroup/Chat-UniVi)] Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding    
[[2024 ICLR](https://arxiv.org/pdf/2309.04669)] [[code](https://github.com/jy0205/LaVIT)] LaVIT: Unified language-vision pretraining in LLM with dynamic discrete visual tokenization    
[[2024 arXiv](https://arxiv.org/abs/2407.02392)] [[code](https://github.com/CircleRadon/TokenPacker)] TokenPacker: Efficient Visual Projector for Multimodal LLM    
[[2024 arXiv](https://arxiv.org/pdf/2405.20985)] [[code](https://github.com/yaolinli/DeCo)] DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models    
[[2024 CVPR](https://arxiv.org/pdf/2312.00878)] [[code](https://github.com/WalBouss/GEM)] Grounding Everything: Emerging Localization Properties in Vision-Language Transformers    
[[2025 CVPR](https://arxiv.org/pdf/2504.08966)] [[code](https://github.com/orailix/PACT/tree/main)] PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models   


<a name="Classical_Segmentation"></a>
# Classical segmentation method
[[2015 CVPR](https://arxiv.org/abs/1411.4038)] [[code](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)] FCN: Fully Convolutional Networks for Semantic Segmentation    
[[2016 MICCAI](https://arxiv.org/pdf/1505.04597)] UNet: Convolutional Networks for Biomedical Image Segmentation    
[[2017 arXiv](https://arxiv.org/pdf/1706.05587)] DeepLabV3: Rethinking atrous convolution for semantic image segmentation    
[[2018 CVPR](https://arxiv.org/pdf/1802.02611)] DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation    
[[2019 CVPR](https://arxiv.org/pdf/1901.02446)] Semantic FPN: Panoptic Feature Pyramid Networks    
[[2021 CVPR](https://arxiv.org/pdf/2012.15840)] [[code](https://github.com/fudan-zvg/SETR)] SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers    
[[2021 ICCV](https://arxiv.org/pdf/2105.05633)] [[code](https://github.com/rstrudel/segmenter)] Segmenter: Transformer for Semantic Segmentation    
[[2021 NeurIPS](https://arxiv.org/pdf/2105.15203)] [[code](https://github.com/NVlabs/SegFormer)] SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers    
[[2021 CVPR](https://arxiv.org/pdf/2107.06278)] [[code](https://github.com/facebookresearch/MaskFormer)] MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation    
[[2022 CVPR](https://arxiv.org/pdf/2112.01527)] [[code](https://github.com/facebookresearch/Mask2Former)] Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation    
[[2024 CVPR](https://arxiv.org/pdf/2312.04265)] [[code](https://github.com/w1oves/Rein)] Rein: Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation    

<a name="Classical_detection"></a>
# Classical detection method
[[2015 NeurIPS](https://arxiv.org/pdf/1506.01497)] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks    
[[2020 ECCV](https://arxiv.org/pdf/2005.12872)] [[code](https://github.com/facebookresearch/detr)] DETR: End-to-End Object Detection with Transformers    
[[2021 ICLR](https://arxiv.org/pdf/2010.04159)] [[code](https://github.com/fundamentalvision/Deformable-DETR)] Deformable DETR: Deformable Transformers for End-to-End Object Detection    
[[2023 ICLR](https://arxiv.org/pdf/2203.03605)] [[code](https://github.com/IDEA-Research/DINO)] DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection    

<a name="Backbone"></a>
# Backbone
[[2017 NeurIPS](https://arxiv.org/pdf/1706.03762)] [[code](https://github.com/tensorflow/tensor2tensor)] transfomer: Attention Is All You Need    
[[2021 ICLR](https://arxiv.org/pdf/2010.11929)] [[code](https://github.com/google-research/vision_transformer)] ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale    
[[2021 ICML](https://arxiv.org/pdf/2012.12877)] DeiT: Training data-efficient image transformers & distillation through attention    
[[2021 ICCV](https://arxiv.org/pdf/2103.14030)] [[code](https://github.com/microsoft/Swin-Transformer)] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows    
[[2021 NeurIPS](https://arxiv.org/pdf/2104.13840)] [[code](https://github.com/Meituan-AutoML/Twins)] Twins: Revisiting the Design of Spatial Attention in Vision Transformers    
[[2022 CVPR](https://arxiv.org/pdf/2203.10833)] [[code](https://github.com/htdt/hyp_metric)] Hyperbolic Vision Transformers: Combining Improvements in Metric Learning      
[[2022 ICLR](https://arxiv.org/pdf/2106.08254)] [[code](https://github.com/microsoft/unilm/tree/master/beit)] BEiT: BERT Pre-Training of Image Transformers    
[[2022 CVPR](https://arxiv.org/pdf/2111.06377)] [[code](https://github.com/facebookresearch/mae)] MAE: Masked Autoencoders Are Scalable Vision Learners    
[[2022 CVPR](https://arxiv.org/pdf/2111.11418)] [[code](https://github.com/sail-sg/poolformer)] PoolFormer: MetaFormer is Actually What You Need for Vision    
[[2022 NeurIPS](https://arxiv.org/pdf/2209.08575)] [[code](https://github.com/Visual-Attention-Network/SegNeXt)] SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation    
[[2023 ICCV](https://arxiv.org/pdf/2303.08131)] [[code](https://github.com/IDEA-Research/OpenSeeD)] OpenSeeD: A simple framework for open-vocabulary segmentation and detection    
[[2023 arXiv](https://arxiv.org/pdf/2304.02643)] [[code](https://github.com/facebookresearch/segment-anything)] [[demo](https://segment-anything.com/demo)] SAM: Segment Anything    
[[2024 arXiv](https://arxiv.org/pdf/2408.00714)] [[code](https://github.com/facebookresearch/sam2)] [[demo](https://sam2.metademolab.com/)] SAM 2: Segment Anything in Images and Videos    


<a name="CLIP"></a>
# CLIP
[[2021 ICML](https://arxiv.org/pdf/2103.00020)] [[code](https://github.com/OpenAI/CLIP)] CLIP: Learning transferable visual models from natural language supervision      
[[2022 IJCV](https://arxiv.org/pdf/2109.01134)] [[code](https://github.com/KaiyangZhou/CoOp)] CoOp: Learning to Prompt for Vision-Language Models      
[[2022 ECCV](https://arxiv.org/pdf/2203.12119)] [[code](https://github.com/kmnp/vpt)] VPT: Visual Prompt Tuning      
[[2022 ICLR](https://arxiv.org/pdf/2106.09685)] [[code](https://github.com/microsoft/LoRA)] LoRA: Low-Rank Adaptation of Large Language Models     
[[2022 NeurIPS](https://arxiv.org/pdf/2209.07511)] [[code](https://github.com/azshue/TPT)] TPT: Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models      
[[2022 Arxiv](https://arxiv.org/pdf/2204.03649)] [[code](https://github.com/tonyhuang2022/UPL)] UPL: Unsupervised Prompt Learning for Vision-Language Models     
[[2022 Arxiv](https://arxiv.org/pdf/2212.00784)] [[code](https://github.com/jonkahana/CLIPPR)] CLIPPR: Improving Zero-Shot Models with Label Distribution Priors      
[[2022 CVPR](https://arxiv.org/pdf/2203.05557)] [[code](https://github.com/KaiyangZhou/CoOp)] CoCoOp: Conditional Prompt Learning for Vision-Language Models      
[[2023 CVPR](https://arxiv.org/pdf/2211.10277)] [[code](https://github.com/geekyutao/TaskRe)] TaskRes: Task Residual for Tuning Vision-Language Models      
[[2023 ICML](https://arxiv.org/pdf/2305.00350)] [[code](https://github.com/korawat-tanwisuth/POUF)] POUF: Prompt-Oriented Unsupervised Fine-tuning for Large Pre-trained Models      
[[2023 NeurIPS](https://arxiv.org/pdf/2306.01669)] [[code](https://github.com/BatsResearch/menghini-neurips23-code)] Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning      
[[2023 NeurIPS](https://arxiv.org/pdf/2305.18287)] [[code](https://github.com/jmiemirza/LaFTer)] LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections      
[[2023 PRCV](https://arxiv.org/pdf/2308.11507)] Unsupervised Prototype Adapter for Vision-Language Models      
[[2023 IJCV](https://arxiv.org/pdf/2110.04544)] [[code](https://github.com/gaopengcuhk/CLIP-Adapter)] CLIP-Adapter: Better Vision-Language Models with Feature Adapters      
[[2024 CVPR](https://arxiv.org/pdf/2404.17753)] [[code](https://github.com/YCaigogogo/CODER)] CODER: Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification      
[[2024 CVPR](https://arxiv.org/pdf/2404.02285)] [[code](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline)] LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP      
[[2024 CVPR](https://arxiv.org/pdf/2403.02781)] [[code](https://github.com/zhengli97/PromptKD)] PromptKD: Unsupervised Prompt Distillation for Vision-Language Models      
[[2024 CVPR](https://arxiv.org/pdf/2403.15132)] [[code](https://github.com/alwaysuu/CLIPDenoising)] Transfer CLIP for Generalizable Image Denoising      
[[2024 ECCV](https://arxiv.org/pdf/2404.07204)] BRAVE: Broadening the visual encoding of vision-language model      
[[2024 ICML](https://arxiv.org/pdf/2406.10502)] [[code](https://github.com/vanillaer/CPL-ICML2024)] Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data      
[[2024 CVPR](https://arxiv.org/pdf/2307.12732)] [[code](https://github.com/winycg/CLIP-KD)] CLIP-KD: An Empirical Study of CLIP Model Distillation    
[[2025 WACV](https://arxiv.org/pdf/2408.08855)] [[code](https://github.com/Externalhappy/DPA)] DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models      
[[2025 WACV](https://arxiv.org/pdf/2403.12952)] [[code](https://github.com/elaine-sui/TPS)] Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models      
[[2025 WACV](https://arxiv.org/pdf/2410.08211)] [[code](https://github.com/astra-vision/LatteCLIP)] LATTECLIP: Unsupervised CLIP Fine-Tuning via LMM-Synthetic Texts      
[[2025 ICLR](https://arxiv.org/pdf/2502.04263)] [[code](https://github.com/miccunifi/Cross-the-Gap)] CROSS THE GAP: EXPOSING THE INTRA-MODAL MISALIGNMENT IN CLIP VIA MODALITY INVERSION      
[[2025 ICLR](https://arxiv.org/pdf/2410.05270)] [[code](https://github.com/astra-vision/ProLIP)] CLIP’s Visual Embedding Projector is a Few-shot Cornucopia    
[[2025 CVPR](https://arxiv.org/pdf/2505.23694)]  [[code](https://github.com/Noahsark/DA-VPT)] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers     
[[2025 ICML](https://arxiv.org/pdf/2506.02557)]  [[code](https://github.com/peterant330/KUEA)] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models     
[[2025 CVPR](https://arxiv.org/pdf/2503.16873)] [[code](https://github.com/k0u-id/CCD)] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification   
[[2024 CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf)] [[code](https://github.com/ZjjConan/VLM-MultiModalAdapter)] Multi-Modal Adapter for Vision-Language Models    

<a name="open_segmentation"></a>
# Open vocabulary segmentation
[[2022 ICLR](https://arxiv.org/pdf/2201.03546)] [[code](https://github.com/isl-org/lang-seg)] Lseg: Language-driven semantic segmentation (Supervised)    
[[2022 CVPR](https://arxiv.org/pdf/2112.07910)] [[code](https://github.com/dingjiansw101/ZegFormer)] ZegFormer: Decoupling Zero-Shot Semantic Segmentation    
[[2022 ECCV](https://arxiv.org/pdf/2112.01071)] [[code](https://github.com/chongzhou96/MaskCLIP)] MaskCLIP+: Extract Free Dense Labels from CLIP    
[[2022 ECCV](https://arxiv.org/pdf/2207.08455v2)] ViL-Seg: Open-World Semantic Segmentation via Contrasting and Clustering Vision-Language Embeddings    
[[2022 CVPR](https://arxiv.org/pdf/2202.11094)] [[code](https://github.com/NVlabs/GroupViT)] GroupViT: Semantic Segmentation Emerges from Text Supervision (Open-Vocabulary Zero-Shot)    
[[2022 ECCV](https://arxiv.org/pdf/2112.12143)] OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels    
[[2023 CVPR](https://arxiv.org/pdf/2303.17225)] [[code](https://github.com/bytedance/FreeSeg)] FreeSeg: Unified, Universal, and Open-Vocabulary Image Segmentation    
[[2023 ICML](https://arxiv.org/pdf/2211.14813)] [[code](https://github.com/ArrowLuo/SegCLIP)] SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation (Zero-Shot)   
[[2023 CVPR](https://arxiv.org/pdf/2212.03588)] [[code](https://github.com/ZiqinZhou66/ZegCLIP)] ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation     
[[2023 CVPR](https://arxiv.org/pdf/2212.11270)] [[code](https://github.com/microsoft/X-Decoder/tree/main)] X-Decoder: Generalized Decoding for Pixel, Image, and Language    
[[2023 CVPR](https://arxiv.org/pdf/2303.04803)] [[code](https://github.com/NVlabs/ODISE)] ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models    
[[2023 ICML](https://arxiv.org/pdf/2208.08984)] [[code](https://github.com/mlpc-ucsd/MaskCLIP)] MaskCLIP: Open-Vocabulary Universal Image Segmentation with MaskCLIP    
[[2023 CVPR](https://arxiv.org/pdf/2302.12242)] [[code](https://github.com/MendelXu/SAN)] SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation    
[[2024 ECCV](https://arxiv.org/pdf/2312.12359)] [[code](https://github.com/wysoczanska/clip_dinoiser)] CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation    
[[2024 CVPR](https://arxiv.org/pdf/2311.15537)] [[code](https://github.com/xb534/SED)] SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation    
[[2024 TPAMI](https://arxiv.org/pdf/2306.15880)] [[code](https://github.com/jianzongwu/Awesome-Open-Vocabulary)] Review: Towards Open Vocabulary Learning: A Survey         
[[2025 ICCV](https://arxiv.org/abs/2412.06244)] [[code](https://github.com/HVision-NKU/DenseVLM)] Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction        
[[2024 CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Exploring_Regional_Clues_in_CLIP_for_Zero-Shot_Semantic_Segmentation_CVPR_2024_paper.pdf)] [[code](https://github.com/Jittor/JSeg)] Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation      
[[2025 CVPR](https://arxiv.org/pdf/2505.04410)] [[code](https://github.com/xiaomoguhz/DeCLIP)] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception     

<a name="open_detection"></a>
# Open vocabulary object detection
[[2021 CVPR](https://arxiv.org/pdf/2011.10678)] [[code](https://github.com/alirezazareian/ovr-cnn)] Open-Vocabulary Object Detection Using Captions    
[[2022 ICLR](https://arxiv.org/pdf/2104.13921)] [[code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)] ViLD: Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation    
[[2022 CVPR](https://arxiv.org/pdf/2112.03857)] [[code](https://github.com/microsoft/GLIP)] GLIP: Grounded Language-Image Pre-training    
[[2022 NeurIPS](https://arxiv.org/pdf/2206.05836)] [[code](https://github.com/microsoft/GLIP)] GLIPv2: Unifying Localization and Vision-Language Understanding    

<a name="Other"></a>
# Other Technologies
[[2016 CVPRW](https://arxiv.org/pdf/1609.05158)] pixel shuffle    
[[2020 NeurIPS](https://arxiv.org/pdf/2006.11239)] [[code](https://github.com/hojonathanho/diffusion)] DDPM: Denoising Diffusion Probabilistic Models    
