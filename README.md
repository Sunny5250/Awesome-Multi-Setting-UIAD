# Awesome-Multi-Setting-UIAD [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A taxonomy of unsupervised industrial anomaly detection (UIAD) methods and datasets (updating).

# RGB UIAD
## Datasets
|  Title  |   Venue  |   Date   |   Code   |   Paradigm   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**Uninformed Students: Student-Teacher Anomaly Detection With Discriminative Latent Embeddings**](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html) <br> | CVPR | 2020 | [Code](https://github.com/denguir/student-teacher-anomaly-detection) | Teacher-student architecture |
| [**Anomaly Detection via Reverse Distillation From One-Class Embedding**](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Code](https://github.com/hq-deng/RD4AD) | Teacher-student architecture |
| [**DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/apple/ml-destseg) | Teacher-student architecture |
| [**EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies**](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) <br> | WACV | 2024 | [Unofficial Code](https://github.com/rximg/EfficientAD) | Teacher-student architecture |
| [**CutPaste: Self-Supervised Learning for Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html) <br> | CVPR | 2021 | [Unofficial Code](https://github.com/LilitYolyan/CutPaste) |  One-class classification |
| [**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/DonaldRR/SimpleNet) |  One-class classification |
| [**FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows**](https://arxiv.org/abs/2111.07677) <br> | - | 2021 | [Unofficial Code](https://github.com/gathierry/FastFlow) | Distribution map |
| [**CFLOW-AD: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows**](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html) <br> | WACV | 2022 | [Code](https://github.com/gudovskiy/cflow-ad) |  Distribution map |
| [**Fully Convolutional Cross-Scale-Flows for Image-Based Defect Detection**](https://openaccess.thecvf.com/content/WACV2022/html/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.html) <br> | WACV | 2022 | [Code](https://github.com/marco-rudolph/cs-flow) | Distribution map | |
| [**PyramidFlow: High-Resolution Defect Contrastive Localization Using Pyramid Normalizing Flow**](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Code](https://github.com/gasharper/PyramidFlow) | Distribution map |
| [**MSFlow: Multiscale Flow-Based Framework for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/document/10384766) <br> | TNNLS | 2024 | [Code](https://github.com/cool-xuan/msflow) | Distribution map |
| [**Towards Total Recall in Industrial Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Code](https://github.com/amazon-science/patchcore-inspection) | Memory bank |
| [**CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization**](https://ieeexplore.ieee.org/abstract/document/9839549) <br> | IEEE Access | 2022 | [Code](https://github.com/sungwool/CFA_for_anomaly_localization) | Memory bank |
| [**PNI : Industrial Anomaly Detection using Position and Neighborhood Information**](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Code](https://github.com/wogur110/PNI_anomaly_detection) | Memory bank |
| [**ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection**](https://openaccess.thecvf.com/content/WACV2024/html/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.html) <br> | WACV | 2024 | [Unofficial Code](https://github.com/travishsu/ReConPatch-TF) | Memory bank |
| [**DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection**](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) <br> | ICCV | 2021 | [Code](https://github.com/VitjanZ/DRAEM) | Reconstruction |
| [**DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7955_ECCV_2022_paper.php) <br> | ECCV | 2022 | [Code](https://github.com/VitjanZ/DSR_anomaly_detection) | Reconstruction |
| [**Omni-Frequency Channel-Selection Representations for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/document/10192551) <br> | TIP | 2023 | [Code](https://github.com/zhangzjn/OCR-GAN) | Reconstruction |
| [**Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection**](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Code](https://github.com/xcyao00/FOD) | Reconstruction |
| [**Anomaly Detection with Conditioned Denoising Diffusion Models**](https://arxiv.org/abs/2305.15956) <br> | - | 2023 | [Code](https://github.com/arimousa/DDAD) | Reconstruction |
| [**RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html) <br> | CVPR | 2024 | [Code](https://github.com/cnulab/RealNet) | Reconstruction |
| [**title**](web) <br> | WACV | 2022 | [Code](code_web) | paradigm |

## Methods

# 3D UIAD


# Multimodal UIAD

