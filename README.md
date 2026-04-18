Overview
This project implements a Masked Autoencoder (MAE) from scratch using base PyTorch — no pretrained weights, no high-level wrappers. The model learns visual representations by reconstructing images where 75% of patches have been randomly masked, following the asymmetric ViT-based encoder-decoder design introduced by He et al. (Meta AI, 2021).
The encoder only processes the visible 25% of patches. The decoder reconstructs the full image from those sparse tokens plus learnable mask tokens. The entire system is trained with self-supervision — no labels required.

Architecture
Input Image (224×224)
       ↓
Patchify → 196 patches of 16×16
       ↓
Random Masking → keep 49 (25%), mask 147 (75%)
       ↓
ViT-Base Encoder (768 dim, 12 layers, 12 heads)
       ↓
Linear Projection → Decoder Dimension (384)
       ↓
Insert Learnable Mask Tokens + Positional Embeddings
       ↓
ViT-Small Decoder (384 dim, 12 layers, 6 heads)
       ↓
MSE Loss on masked patches only
Model Configuration
ComponentModelHidden DimLayersHeadsParamsEncoderViT-Base (B/16)7681212~86MDecoderViT-Small (S/16)384126~22MTotal~107.8M

Dataset
Tiny ImageNet — 100,000 training images across 200 classes, resized to 224×224.

Platform: Kaggle
Dataset: akash2sharma/tiny-imagenet
Accelerator: Kaggle T4 × 2 (Dual GPU)


Training Details
HyperparameterValueMask Ratio75%Patch Size16 × 16Batch Size32Epochs9OptimizerAdamWLearning Rate1e-4Weight Decay0.05LR SchedulerCosine AnnealingPrecisionMixed (torch.cuda.amp)LossMSE on masked patches only

Results
MetricScorePSNR16.40 dBSSIM0.4649

Note: These results reflect training under Kaggle's free GPU memory constraints (9 epochs, batch size 32). The original MAE paper trains for 1600 epochs on full ImageNet. With more compute — larger batch size, more epochs, and tuned hyperparameters — reconstruction quality would improve significantly. The architecture and training pipeline are fully correct; compute is the limiting factor.


Qualitative Reconstructions
The model outputs a 3-panel view for evaluation:
Masked Input (75% removed)Model ReconstructionGround Truth(see notebook)(see notebook)(see notebook)

Project Structure
├── 22f_3863___22f_3324_AI_ASS02_MAE.ipynb   # Main notebook
├── README.md
Notebook Sections

Environment Setup — installs, imports, hyperparameters
Dataset Loading — Tiny ImageNet via KaggleHub
Model Architecture

Patchify / Unpatchify / Random Masking
Patch Embedding
Transformer Block (shared)
ViT Encoder (Base)
ViT Decoder (Small)
Full MAE Model


Training Setup — loss, optimizer, scheduler, mixed precision, DataParallel
Visualization Module — masked input / reconstruction / ground truth
Quantitative Evaluation — PSNR & SSIM
Gradio App — interactive demo with masking ratio slider


Gradio App
A deployed Gradio app lets you:

Upload any image
Adjust the masking ratio (0.5 – 0.9)
See the real-time reconstruction


How to Run
On Kaggle (Recommended)

Create a new Kaggle notebook
Add the Tiny ImageNet dataset
Enable GPU T4 × 2 accelerator
Upload and run 22f_3863___22f_3324_AI_ASS02_MAE.ipynb

Dependencies
bashpip install torch torchvision einops gradio torchmetrics

Key Implementation Details

No mask tokens in encoder — the encoder only sees visible patches, keeping it efficient
Positional embeddings added before masking — the model always knows where visible patches came from
Loss computed only on masked patches — visible patches are excluded from the training signal
ids_restore — tracks original patch ordering so the decoder can reconstruct spatially correct outputs
Mixed precision + DataParallel — necessary to train a 107.8M parameter model on free-tier GPUs


References

He, K. et al. (2021). Masked Autoencoders Are Scalable Vision Learners. Meta AI.
Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.


Authors
NameRoll NoBushra Abad22F-3863Tayyaba Imtiyaz22F-3324
Course: Generative AI (AI4009) | Semester: Spring 2026 | Institution: FAST NUCES
