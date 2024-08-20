# Awesome-Multi-Setting-UIAD [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A taxonomy of unsupervised industrial anomaly detection (UIAD) methods and datasets (updating).

# Contents
- [RGB UIAD](#rgb-uiad)
- [3D UIAD](#3d-uiad)
- [Multimodal UIAD](#multimodal-uiad)

# RGB UIAD
## Methods
|  Name  |  Title  |   Publication  |   Year   |   Code   |   Paradigm   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| US | [**Uninformed Students: Student-Teacher Anomaly Detection With Discriminative Latent Embeddings**](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html) <br> | CVPR | 2020 | [Code](https://github.com/denguir/student-teacher-anomaly-detection) | Teacher-student architecture |
| RD4AD | [**Anomaly Detection via Reverse Distillation From One-Class Embedding**](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Code](https://github.com/hq-deng/RD4AD) | Teacher-student architecture |
| DeSTSeg | [**DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/apple/ml-destseg) | Teacher-student architecture |
| EfficientAD | [**EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies**](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) <br> | WACV | 2024 | [Unofficial Code](https://github.com/rximg/EfficientAD) | Teacher-student architecture |
| CutPaste | [**CutPaste: Self-Supervised Learning for Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html) <br> | CVPR | 2021 | [Unofficial Code](https://github.com/LilitYolyan/CutPaste) |  One-class classification |
| SimpleNet | [**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/DonaldRR/SimpleNet) |  One-class classification |
| FastFlow | [**FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows**](https://arxiv.org/abs/2111.07677) <br> | - | 2021 | [Unofficial Code](https://github.com/gathierry/FastFlow) | Distribution map |
| CFLOW-AD | [**CFLOW-AD: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows**](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html) <br> | WACV | 2022 | [Code](https://github.com/gudovskiy/cflow-ad) |  Distribution map |
| CS-Flow | [**Fully Convolutional Cross-Scale-Flows for Image-Based Defect Detection**](https://openaccess.thecvf.com/content/WACV2022/html/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.html) <br> | WACV | 2022 | [Code](https://github.com/marco-rudolph/cs-flow) | Distribution map | |
| PyramidFlow | [**PyramidFlow: High-Resolution Defect Contrastive Localization Using Pyramid Normalizing Flow**](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/gasharper/PyramidFlow) | Distribution map |
| MSFlow | [**MSFlow: Multiscale Flow-Based Framework for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/document/10384766) <br> | TNNLS | 2024 | [Code](https://github.com/cool-xuan/msflow) | Distribution map |
| PatchCore | [**Towards Total Recall in Industrial Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Code](https://github.com/amazon-science/patchcore-inspection) | Memory bank |
| CFA | [**CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization**](https://ieeexplore.ieee.org/abstract/document/9839549) <br> | IEEE Access | 2022 | [Code](https://github.com/sungwool/CFA_for_anomaly_localization) | Memory bank |
| PNI | [**PNI : Industrial Anomaly Detection using Position and Neighborhood Information**](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Code](https://github.com/wogur110/PNI_anomaly_detection) | Memory bank |
| ReConPatch | [**ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection**](https://openaccess.thecvf.com/content/WACV2024/html/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.html) <br> | WACV | 2024 | [Unofficial Code](https://github.com/travishsu/ReConPatch-TF) | Memory bank |
| DRÆM | [**DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection**](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) <br> | ICCV | 2021 | [Code](https://github.com/VitjanZ/DRAEM) | Reconstruction |
| DSR | [**DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7955_ECCV_2022_paper.php) <br> | ECCV | 2022 | [Code](https://github.com/VitjanZ/DSR_anomaly_detection) | Reconstruction |
| OCR-GAN | [**Omni-Frequency Channel-Selection Representations for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/document/10192551) <br> | TIP | 2023 | [Code](https://github.com/zhangzjn/OCR-GAN) | Reconstruction |
| FOD | [**Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection**](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Code](https://github.com/xcyao00/FOD) | Reconstruction |
| DDAD | [**Anomaly Detection with Conditioned Denoising Diffusion Models**](https://arxiv.org/abs/2305.15956) <br> | - | 2023 | [Code](https://github.com/arimousa/DDAD) | Reconstruction |
| RealNet | [**RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html) <br> | CVPR | 2024 | [Code](https://github.com/cnulab/RealNet) | Reconstruction |
| name | [**title**](web) <br> | venue | year | [Code](code_web) | paradigm |

## Datasets
|  Dataset  |   Resource   |   Year   |   Type   |   Train   |    Test (good)   |   Test (anomaly)   |   Val   |   Total   |   Class   |    Anomaly Type   |    Modal Type   |
|:--------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**MVTec AD**](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html) <br> | [Data](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 2019 | Real | 3629 | 467 | 1258 | - | 5354 | 15 | 73 | RGB |
| [**BTAD**](https://ieeexplore.ieee.org/document/9576231) <br> | [Data](https://avires.dimi.uniud.it/papers/btad/btad.zip) | 2021 | Real | 1799 | 451 | 290 | - | 2540 | 3 | - | RGB |
| [**MPDD**](https://ieeexplore.ieee.org/document/9631567) <br> | [Data](https://vutbr-my.sharepoint.com/personal/xjezek16_vutbr_cz/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxjezek16%5Fvutbr%5Fcz%2FDocuments%2FMPDD&ga=1) | 2021 | Real | 888 | 176 | 282 | - | 1346 | 6 | - | RGB |
| [**MVTec LOCO-AD**](https://link.springer.com/article/10.1007/s11263-022-01578-9) <br> | [Data](https://www.mvtec.com/company/research/datasets/mvtec-loco) | 2022 | Real | 1772 | 575 | 993 | 304 | 3644 | 5 | 89 | RGB |
| [**VisA**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2149_ECCV_2022_paper.php) <br> | [Data](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) | 2022 | Real | 9621 | 0 | 1200 | - | 10821 | 12 | - | RGB |
| [**GoodsAD**](https://ieeexplore.ieee.org/document/10387569) <br> | [Data](https://github.com/jianzhang96/GoodsAD) | 2023 | Real | 3136 | 1328 | 1660 | - | 6124 | 6 | - | RGB |
| [**CID**](https://ieeexplore.ieee.org/document/10504848) <br> | [Data](https://github.com/Light-ZhangTao/Insulator-Defect-Detection) | 2024 | Real | 3900 | 33 | 360 | - | 4293 | 1 | 6 | RGB |
| [**Real-IAD**](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html) <br> | [Data](https://realiad4ad.github.io/Real-IAD/) | 2024 | Real | 72840 | 0 | 78210 | - | 151050 | 30 | 8 | RGB |
| [**RAD**](https://arxiv.org/abs/2406.07176) <br> | [Data](https://github.com/hustCYQ/RAD-dataset) | 2024 | Real | 213 | 73 | 1224 | - | 1510 | 4 | - | RGB |
| [**MIAD**](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/html/Bao_MIAD_A_Maintenance_Inspection_Dataset_for_Unsupervised_Anomaly_Detection_ICCVW_2023_paper.html) <br> | [Data](https://miad-2022.github.io/) | 2023 | Synthetic | 70000 | 17500 | 17500 | - | 105000 | 7 | 13 | RGB |
| [**MAD-Sim**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bc5aef775aacc1650a9790f1428bcea-Abstract-Datasets_and_Benchmarks.html) <br> | [Data](https://drive.google.com/file/d/1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq/view?usp=sharing) | 2023 | Synthetic | 4200 | 638 | 4951 | - | 9789 | 20 | 3 | RGB |
| [**DTD-Synthetic**](https://openaccess.thecvf.com/content/WACV2023/html/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.html) <br> | [Data](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1) | 2024 | Synthetic | 1200 | 357 | 947 | - | 2504 | 12 | - | RGB |
| [**dataset**](web) <br> | [data](data_web) | - | - | - | - | - | - | - | - | - | - |

# 3D UIAD
## Methods
|  Name  |  Title  |   Publication  |   Year   |   Code   |   Paradigm   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| 3D-ST | [**Anomaly Detection in 3D Point Clouds Using Deep Geometric Descriptors**](https://openaccess.thecvf.com/content/WACV2023/html/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.html) <br> | WACV | 2023 | - | Teacher-student architecture |
| Group3AD | [**Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning**](https://arxiv.org/abs/2408.04604) <br> | ACM MM | 2024 | [Code](https://github.com/M-3LAB/Group3AD) | Memory bank |
| R3D-AD | [**R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection**](https://arxiv.org/abs/2407.10862) <br> | ECCV | 2024 | - | Reconstruction |
| name | [**title**](web) <br> | venue | year | [Code](code_web) | paradigm |

## Datasets
|  Dataset  |   Resource   |   Year   |   Type   |   Train   |    Test (good)   |   Test (anomaly)   |   Val   |   Total   |   Class   |    Anomaly Type    |    Modal Type    |
|:--------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**Real3D-AD**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html) <br> | [data](https://github.com/M-3LAB/Real3D-AD) | 2023 | Real | 48 | 604 | 602 | - | 1254 | 12 | 3 |  Point cloud |
| [**Anomaly-ShapeNet**](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Towards_Scalable_3D_Anomaly_Detection_and_Localization_A_Benchmark_via_CVPR_2024_paper.html) <br> | [data](https://github.com/Chopper-233/Anomaly-ShapeNet) | 2023 | Synthetic | 208 | 780 | 943 | - | 1931 | 50 | 7 | Point cloud |
| [**dataset**](web) <br> | [data](data_web) | - | - | - | - | - | - | - | - | - | - |

# Multimodal UIAD
## Methods
|  Name  |  Title  |   Publication  |   Year   |   Code   |   Paradigm   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| BTF | [**Back to the Feature: Classical 3D Features Are (Almost) All You Need for 3D Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Horwitz_Back_to_the_Feature_Classical_3D_Features_Are_Almost_All_CVPRW_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/eliahuhorwitz/3D-ADS) | - |
| AST | [**Asymmetric Student-Teacher Networks for Industrial Anomaly Detection**](https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html) <br> | WACV | 2023 | [Code](https://github.com/marco-rudolph/AST) | Teacher-student architecture |
| MMRD | [**Rethinking Reverse Distillation for Multi-Modal Anomaly Detection**](https://ojs.aaai.org/index.php/AAAI/article/view/28687) <br> | AAAI | 2024 | - | Teacher-student architecture |
| M3DM | [**Multimodal Industrial Anomaly Detection via Hybrid Fusion**](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/nomewang/M3DM) | Memory bank |
| CPMF | [**Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection**](https://arxiv.org/abs/2303.13194) <br> | - | 2023 | [Code](https://github.com/caoyunkang/CPMF) | Memory bank |
| Shape-Guided | [**Shape-Guided Dual-Memory Learning for 3D Anomaly Detection**](https://proceedings.mlr.press/v202/chu23b.html) <br> | ICML | 2023 | [Code](https://github.com/jayliu0313/Shape-Guided) | Memory bank |
| LSFA | [**Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection**](https://arxiv.org/abs/2401.03145) <br> | ECCV | 2024 | - | Memory bank |
| ITNM | [**Incremental Template Neighborhood Matching for 3D anomaly detection**](https://dl.acm.org/doi/10.1016/j.neucom.2024.127483) <br> | Neurocomputing | 2024 | - | Memory bank |
| CMDIAD | [**Incomplete Multimodal Industrial Anomaly Detection via Cross-Modal Distillation**](https://arxiv.org/abs/2405.13571) <br> | - | 2024 | [Code](https://github.com/evenrose/CMDIAD) | Memory bank |
| M3DM-NR | [**M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising**](https://arxiv.org/abs/2406.02263) <br> | - | 2024 | - | Memory bank |
| EasyNet | [**EasyNet: An Easy Network for 3D Industrial Anomaly Detection**](https://dl.acm.org/doi/10.1145/3581783.3611876) <br> | ACM MM | 2023 | [Code](https://github.com/TaoTao9/EasyNet) | Reconstruction |
| DBRN | [**Dual-Branch Reconstruction Network for Industrial Anomaly Detection with RGB-D Data**](https://arxiv.org/abs/2311.06797) <br> | - | 2023 | - | Reconstruction |
| 3DSR | [**Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation**](https://openaccess.thecvf.com/content/WACV2024/html/Zavrtanik_Cheating_Depth_Enhancing_3D_Surface_Anomaly_Detection_via_Depth_Simulation_WACV_2024_paper.html) <br> | WACV | 2024 | [Code](https://github.com/VitjanZ/3DSR) | Reconstruction |
| CFM | [**Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping**](https://openaccess.thecvf.com/content/CVPR2024/html/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.html) <br> | CVPR | 2024 | [Code](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping) | Reconstruction |
| 3DRÆM | [**Keep DRÆMing: Discriminative 3D anomaly detection through anomaly simulation**](https://www.sciencedirect.com/science/article/pii/S0167865524000862) <br> | PRL | 2024 | - | Reconstruction |
| name | [**title**](web) <br> | venue | year | [Code](code_web) | paradigm |

## Datasets
|  Dataset  |   Resource   |   Year   |   Type   |   Train   |    Test (good)   |   Test (anomaly)   |   Val   |   Total   |   Class   |    Anomaly Type   |    Modal Type   |
|:--------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**MVTec 3D-AD**](https://www.scitepress.org/PublicationsDetail.aspx?ID=x2WTZgpY0KM=&t=1) <br> | [data](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) | 2021 | Real | 2656 | 294 | 948 | 294 | 4147 | 10 | 41 | RGB & Point cloud |
| [**PD-REAL**](https://arxiv.org/abs/2311.04095) <br> | [data](https://drive.google.com/file/d/1EBapy9BGnwmZJ9Rxh0uTF-Xii9FU0YCG/view?usp=sharing) | 2023 | Real | 2399 | 300 | 530 | 300 | 3529 | 15 | 6 | RGB & Point cloud |
| [**Eyecandies**](https://openaccess.thecvf.com/content/ACCV2022/html/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.html) <br> | [data](https://eyecan-ai.github.io/eyecandies/download) | 2022 | Synthetic | 10000 | 2250 | 2250 | 1000 | 15500 | 10 | - | RGB & Depth |
| [**dataset**](web) <br> | [data](data_web) | - | - | - | - | - | - | - | - | - | - |
