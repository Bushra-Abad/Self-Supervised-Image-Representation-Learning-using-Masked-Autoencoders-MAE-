📌 Overview
Neural Storyteller is an end-to-end multimodal deep learning system that generates natural language captions for images using a Sequence-to-Sequence (Seq2Seq) architecture. Given any image, the model produces a human-like sentence describing what's happening in it.
Built as part of Assignment 1 — Generative AI (AI4009) at FAST NUCES, Spring 2026.
Authors: Bushra Abad (22F-3863)  |  Tayyaba Imtiaz (22F-3324)

🎯 How It Works
📷 Image
   ↓
🔍 ResNet50 (Pre-trained CNN)
   → Extracts 2048-dim feature vector
   ↓
🔗 EncoderCNN (Linear Projection)
   → Projects to hidden_size (1024-dim)
   ↓
🗣️ DecoderRNN (4-layer LSTM)
   → Generates caption word by word
   ↓
📝 "A group of people playing frisbee in a park."

🗂️ Project Structure
├── Gen_AI_image_captioning.ipynb   # Main Jupyter notebook
├── encoder_model.pth               # Saved encoder weights
├── decoder_model.pth               # Saved decoder weights
├── flickr30k_features.pkl          # Pre-extracted image features
├── word2idx.pkl                    # Vocabulary (word → index)
├── idx2word.pkl                    # Vocabulary (index → word)
└── README.md

🛠️ Tech Stack
ComponentToolFrameworkPyTorchFeature ExtractionResNet50 (torchvision)Sequence Model4-layer LSTMDatasetFlickr30k (31,000+ images, 150,000+ captions)InferenceGreedy Search & Beam SearchEvaluationBLEU-4, Precision, Recall, F1DeploymentGradioPlatformKaggle (Dual GPU T4)

⚙️ Setup & Installation
1. Clone the Repository
bashgit clone https://github.com/your-username/neural-storyteller.git
cd neural-storyteller
2. Install Dependencies
bashpip install torch torchvision gradio nltk scikit-learn tqdm pandas matplotlib Pillow
3. Download the Dataset
Add the Flickr30k dataset to your Kaggle notebook or download it locally.

🚀 Running the Project
Step 1 — Extract Image Features
Run the feature extraction cell once. This uses ResNet50 to convert all images into 2048-dim vectors and saves them to flickr30k_features.pkl.
python# Extracts features for all 31,000+ images
# Only needs to be run once — cached in .pkl file
Step 2 — Preprocess Captions
Loads captions.txt, cleans text, builds vocabulary (~18,083 words), and adds <start>, <end>, and <pad> tokens.
Step 3 — Train the Model
pythonembed_size  = 512
hidden_size = 1024
num_layers  = 4
num_epochs  = 20
optimizer   = Adam
loss        = CrossEntropyLoss(ignore_index=pad_idx)
Step 4 — Generate Captions
python# Greedy Search — fast, picks highest probability word at each step
greedy_search(image_features, encoder, decoder, idx2word, ...)

# Beam Search — explores multiple paths, picks most coherent caption
beam_search(image_features, encoder, decoder, idx2word, beam_width=3, ...)
Step 5 — Launch Gradio App
pythondemo.launch(share=True)
# Upload any image → get a caption instantly

🏛️ Model Architecture
EncoderCNN
Input:  2048-dim ResNet50 feature vector
Linear: 2048 → hidden_size (1024)
BN:     BatchNorm1d
Output: 1024-dim encoded image representation
DecoderRNN
Input:       Word embeddings (embed_size=512)
Hidden init: Encoder output
LSTM:        4 layers, hidden_size=1024
Output:      Linear(1024 → vocab_size=18083)

📊 Results
MetricScoreBLEU-4—Precision—Recall—F1-Score—

📝 Fill in your scores after training.

Loss Curve
Training and validation loss plotted over 20 epochs — showing steady convergence with no significant overfitting.
Sample Predictions
ImageGround TruthGenerated Captionimg1.jpga dog runs on the beacha brown dog is running on the sandimg2.jpgtwo children playing soccertwo boys are playing on a field

🌍 Real-World Applications

📷 Auto-captioning for visually impaired users
🔍 Image search and indexing
🏥 Medical image description
📱 Social media accessibility tools
🤖 Visual Question Answering (VQA) foundation


📋 Assignment Details
DetailInfoCourseGenerative AI — AI4009UniversityFAST NUCESSemesterSpring 2026AssignmentNo. 1PlatformKaggle (GPU T4 x2)

📎 Links

📓 Kaggle Notebook
📝 Medium Blog Post
💼 LinkedIn Post


🤝 Authors
NameRoll NoBushra Abad22F-3863Tayyaba Imtiaz22F-3324
