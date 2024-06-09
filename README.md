# Awesome research works for On-device AI

### Mobile Edge AI Systems

Along with the rapid development of AI and deep learning, DNN models have been widely used in various applications. However, the high computational comlexity of DNN models makes it difficult to deploy them on mobile and edge devices with limited computing resources. This repo collects the research works presenting a system that can efficiently execute DNN models on mobile and edge devices.

- Most-relevant conference: ACM ***MobiSys***, ACM ***MobiCom***, ACM ***Sensys***
- Relevant conference: ACM ***EuroSys***, ACM ***IPSN***, USENIX ***NSDI***, USENIX ***ATC***, ***MLSys***

#### Efficient Inference using Heterogeneous Processors (e.g., CPU, GPU, NPU, etc.)
- [Sensys 2023] Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625789)
- [MobiSys 2023] NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3472381.3479910)
- [ATC 2023] Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices [[paper]](https://www.usenix.org/system/files/atc23-sung.pdf)
- [SenSys 2022] BlastNet: Exploiting Duo-Blocks for Cross-Processor Real-Time DNN Inference [[paper]](https://dl.acm.org/doi/pdf/10.1145/3560905.3568520)
- [MobiSys 2022] Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948)
- [MobiSys 2022] CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices [[paper]](https://dl.acm.org/doi/pdf/10.1145/3498361.3538932)
- [MobiCom 2020] Heimdall: Mobile GPU Coordination Platform for Augmented Reality Applicationss [[paper]](https://dl.acm.org/doi/10.1145/3372224.3419192)  


#### On-device LLM/NLP
- [MobiCom 2024] Mobile Foundation Model as Firmware [[paper]](https://arxiv.org/pdf/2308.14363) [[code]](https://github.com/UbiquitousLearning/MobileFM)

#### On-device Training, Model Adaptation
- [MobiCom 2023] Cost-effective On-device Continual Learning over Memory Hierarchy with Miro [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3613297)
- [MobiCom 2023] AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3592529)
- [MobiSys 2023] ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596852)
- [SenSys 2023] On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625814)
- [MobiCom 2022] Mandheling: mixed-precision on-device DNN training with DSP offloading [[paper]](https://dl.acm.org/doi/abs/10.1145/3495243.3560545)
- [MobiSys 2022] Memory-efficient DNN training on mobile devices [[paper]](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)

#### Profilers
- [SenSys 2023] nnPerf: Demystifying DNN Runtime Inference Latency on Mobile Platforms [[paper]](https://dl.acm.org/doi/10.1145/3625687.3625797)
- [MobiSys 2021] nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices [[paper]](https://dl.acm.org/doi/10.1145/3458864.3467882)

#### Application-centric Approaches
- [MobiSys 2023] OmniLive: Super-Resolution Enhanced 360° Video Live Streaming for Mobile Devices
 [[paper]](https://dl.acm.org/doi/pdf/10.1145/3581791.3596851)
- [IPSN 2023] PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators [[paper]](https://dl.acm.org/doi/pdf/10.1145/3583120.3587045)
- [MobiCom 2022] MobiDepth: real-time depth estimation using on-device dual cameras [[paper]](https://dl.acm.org/doi/10.1145/3495243.3560517)
- [MobiSys 2022] DeepMix: Mobility-aware, Lightweight, and Hybrid 3D Object Detection for Headsets [[paper]](https://doi.org/10.1145/3498361.3538945)
- [MobiCom 2021] Flexible high-resolution object detection on edge devices with tunable latency [[paper]](https://dl.acm.org/doi/abs/10.1145/3447993.3483274)
- [MobiCom 2020] EagleEye: Wearable Camera-based Person Identification in Crowded Urban Spaces [[paper]](https://dl.acm.org/doi/10.1145/3372224.3380881)
- [MobiCom 2020] NEMO: Enabling Neural-enhanced Video Streaming on Commodity Mobile Devices [[paper]](https://dl.acm.org/doi/10.1145/3372224.3419185)


#### Server-Edge Collaborative Inference
- [MobiCom 2023] AccuMO: Accuracy-Centric Multitask Offloading in Edge-Assisted Mobile Augmented Reality [[paper]](https://dl.acm.org/doi/pdf/10.1145/3570361.3592531)
- [IPSN 2023] CoEdge: A Cooperative Edge System for Distributed Real-Time Deep Learning Tasks [[paper]](https://dl.acm.org/doi/pdf/10.1145/3583120.3586955)


---


### Efficient AI methods
DNN model pruning, quantization, compression, efficient ViT, etc. are the most popular methods to reduce the computational complexity of DNN models. This repo collects the research works presenting efficient AI methods.
- Relevant conference: ***CVPR***, ***ICLR***, ***NeurIPS***, ***ICML***, ***ICCV***, ***ECCV***, ***AAAI***
  
#### Pruning and Compression
- [CVPR 2023] DepGraph: Towards Any Structural Pruning [[paper]](https://arxiv.org/abs/2301.12900) [[code]](https://github.com/VainF/Torch-Pruning)
- [ICML 2023] Efficient Latency-Aware CNN Depth Compression via Two-Stage Dynamic Programming [[paper]](https://arxiv.org/abs/2301.12187) [[code]](https://github.com/snu-mllab/Efficient-CNN-Depth-Compression)
- [NeurIPS 2022] Structural Pruning via Latency-Saliency Knapsack [[paper]](https://arxiv.org/abs/2210.06659) [[code]](https://github.com/NVlabs/HALP)

#### Efficient Vision Transformer (ViT)
- [ICLR 2023 top 5%] Token Merging: Your ViT but Faster [[paper]](https://arxiv.org/abs/2210.09461) [[code]](https://github.com/facebookresearch/ToMe)
- [ICCV 2023] Rethinking Vision Transformers for MobileNet Size and Speed [[paper]](https://arxiv.org/abs/2212.08059) [[code]](https://github.com/snap-research/EfficientFormer)
- [ICCV 2023] EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction [[paper]](https://arxiv.org/abs/2205.14756) [[code]](https://github.com/mit-han-lab/efficientvit)
- [CVPR 2023] SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer [[paper]](https://arxiv.org/abs/2303.17605) [[code]](https://github.com/mit-han-lab/sparsevit)
- [CVPR 2022 *Oral*] PoolFormer: MetaFormer Is Actually What You Need for Vision [[paper]](https://arxiv.org/abs/2111.11418) [[code]](https://github.com/sail-sg/poolformer)

#### Elastic Neural Networks
- [CVPR 2023 *Highlight*] Stitchable Neural Networks [[paper]](https://arxiv.org/abs/2302.06586) [[code]](https://github.com/ziplab/SN-Net)

#### Efficient Neural Radiance Fields (NeRF)
- [CVPR 2023] Real-Time Neural Light Field on Mobile Devices [[paper]](https://arxiv.org/abs/2212.08057) [[code]](https://github.com/snap-research/MobileR2L)
- [ECCV 2022] R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis [[paper]](https://arxiv.org/abs/2203.17261) [[code]](https://github.com/snap-research/R2L)