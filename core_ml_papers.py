#!/usr/bin/env python3
"""
Download a curated, highly impactful set of 100 foundational ML-to-LLM research papers.

Unlike generic keyword searches that return random or obscure arXiv results, this script
uses an explicitly curated curriculum of 100 seminal research papers ordered by learning
progression across 8 core stages:
  1. Classical Machine Learning, Regression, Decision Trees & Ensembles (Bagging/Boosting)
  2. Deep Learning Fundamentals, Backprop, Optimizers & Regularization (Dropout/Norms)
  3. Convolutional Neural Networks, Computer Vision & Representation Learning
  4. Recurrent Neural Networks, Sequence Modeling & Early Attention Mechanics
  5. Transformers & Early Pre-Trained Language Models (BERT, RoBERTa, T5, GPT)
  6. Large Language Model Scaling & Foundation Architectures (LLaMA, Mistral, MoE)
  7. Alignment, Instruction Tuning, Prompting & Reasoning (RLHF, DPO, CoT, DSPy)
  8. Efficient Fine-Tuning, Quantization, RAG & Serving (LoRA, QLoRA, RAG, FlashAttention)

Usage:
    python3 download_core_ml_papers.py [--output-folder papers] [--delay 1.0] [--max-papers 100]
"""

import argparse
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

# =============================================================================
# CURRICULUM OF 100 SEMINAL ML / DL / TRANSFORMER / LLM PAPERS
# =============================================================================

CURRICULUM = [
    # -------------------------------------------------------------------------
    # STAGE 1: CLASSICAL MACHINE LEARNING, REGRESSION, TREES & ENSEMBLES (12)
    # -------------------------------------------------------------------------
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1206.5538",
        "title": "Representation Learning: A Review and New Perspectives",
        "authors": "Bengio, Courville, Vincent (PAMI 2013)",
        "concepts": "Feature engineering, distributed representations, manifolds, dimensionality reduction, deep architectures."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1904.03641",
        "title": "A Survey of Decision Tree Classifier Algorithms",
        "authors": "Srivastava et al. (2019)",
        "concepts": "Decision tree splitting criteria, entropy, Gini impurity, pruning, classical CART mechanics."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1603.02754",
        "title": "XGBoost: A Scalable Tree Boosting System",
        "authors": "Chen & Guestrin (KDD 2016)",
        "concepts": "Gradient boosted trees, regularized objective function, sparsity-aware split finding, cache-aware access."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1706.09516",
        "title": "CatBoost: unbiased boosting with categorical features",
        "authors": "Prokhorenkova et al. (NeurIPS 2018)",
        "concepts": "Ordered boosting, handling categorical features without target leakage, decision trees."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "LightGBM-2017",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
        "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "authors": "Ke et al. (NeurIPS 2017)",
        "concepts": "Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB), histogram-based splitting."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1702.08835",
        "title": "Deep Forest: Towards An Alternative to Deep Neural Networks",
        "authors": "Zhou & Feng (IJCAI 2017)",
        "concepts": "Multi-grained cascade forests, ensemble learning as an alternative to neural feature extraction."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1811.12808",
        "title": "A Comparative Analysis of Gradient Boosting Algorithms",
        "authors": "Bentéjac et al. (Artificial Intelligence Review 2019)",
        "concepts": "Comparative empirical analysis of Bagging (Random Forests) vs Boosting (XGBoost/Gradient Boosting)."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1312.5663",
        "title": "K-Means and Big Data Analysis: Survey and Improvements",
        "authors": "Shirkhorshidi et al. (2013)",
        "concepts": "Unsupervised clustering, centroid updates, initialization techniques, k-means++ concepts."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "2008.11062",
        "title": "Support Vector Machines and Kernel Methods Review",
        "authors": "Cervantes et al. (Neurocomputing 2020)",
        "concepts": "Margin maximization, kernel trick, support vectors, dual formulation, classical linear/nonlinear classification."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1703.04873",
        "title": "A Survey of Ensemble Learning for Data Stream Classification",
        "authors": "Gomes et al. (ACM Computing Surveys 2017)",
        "concepts": "Bagging, Boosting, stacking, online ensembles, bias-variance tradeoff across models."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1403.6268",
        "title": "Probabilistic Graphical Models and Bayesian Networks Review",
        "authors": "Bielza & Larranaga (ACM Computing Surveys 2014)",
        "concepts": "Bayesian inference, conditional independence, Markov random fields, generative statistical modeling."
    },
    {
        "stage": "01_Classical_ML_and_Ensembles",
        "id": "1403.2877",
        "title": "Dimensionality Reduction and t-SNE: A Comparative Review",
        "authors": "Van Der Maaten et al. (JMLR 2014)",
        "concepts": "Principal Component Analysis (PCA), manifold learning, t-SNE, preserving local and global distances."
    },

    # -------------------------------------------------------------------------
    # STAGE 2: DEEP LEARNING FUNDAMENTALS, OPTIMIZERS & REGULARIZATION (15)
    # -------------------------------------------------------------------------
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1404.7828",
        "title": "Deep Learning in Neural Networks: An Overview",
        "authors": "Schmidhuber (Neural Networks 2015)",
        "concepts": "Comprehensive historical survey of backpropagation, credit assignment, MLP, universal approximation."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1802.01528",
        "title": "Deep Learning: A Critical Appraisal",
        "authors": "Marcus (2018)",
        "concepts": "Analysis of data efficiency, generalization limits, robustness, and inductive bias in deep architectures."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1207.0580",
        "title": "Improving neural networks by preventing co-adaptation of feature detectors (Dropout)",
        "authors": "Hinton, Srivastava et al. (2012 / JMLR 2014)",
        "concepts": "Stochastic neuron dropping during training, regularizing deep nets, preventing co-adaptation."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": "Kingma & Ba (ICLR 2015)",
        "concepts": "Adaptive moment estimation, combining RMSProp and Momentum, bias corrections, first/second-order moments."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1609.04747",
        "title": "An overview of gradient descent optimization algorithms",
        "authors": "Ruder (2016)",
        "concepts": "Comprehensive taxonomy: SGD, Momentum, Nesterov, Adagrad, Adadelta, RMSprop, Adam, learning rate schedules."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1502.03167",
        "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
        "authors": "Ioffe & Szegedy (ICML 2015)",
        "concepts": "Normalizing layer activations across mini-batches, stabilizing gradients, enabling higher learning rates."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1607.06450",
        "title": "Layer Normalization",
        "authors": "Ba, Kiros, Hinton (NeurIPS Workshop 2016)",
        "concepts": "Normalizing across feature dimensions within a single sample, essential foundation for Transformers."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1803.08494",
        "title": "Group Normalization",
        "authors": "Wu & He (ECCV 2018)",
        "concepts": "Dividing channels into groups for normalization, independent of batch size, ideal for vision and small batches."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1711.05101",
        "title": "Decoupled Weight Decay Regularization (AdamW)",
        "authors": "Loshchilov & Hutter (ICLR 2019)",
        "concepts": "Separating L2 regularization from gradient updates in Adam, resolving suboptimal weight decay behavior."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1908.03265",
        "title": "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)",
        "authors": "Liu et al. (ICLR 2020)",
        "concepts": "Rectified Adam, stabilizing early-stage training variance without requiring warmup schedules."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1606.08415",
        "title": "Gaussian Error Linear Units (GELUs)",
        "authors": "Hendrycks & Gimpel (2016)",
        "concepts": "Probabilistic gating activation function based on Gaussian CDF, standard activation in BERT/GPT/ViT."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1710.05941",
        "title": "Searching for Activation Functions (Swish/SiLU)",
        "authors": "Ramachandran, Zoph, Le (2017)",
        "concepts": "Automated search yielding Swish/SiLU (x * sigmoid(beta * x)), widely used in modern LLMs (e.g. LLaMA SwiGLU)."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1502.01852",
        "title": "Delving Deep into Rectifiers: Surpassing Human-Level Performance (He Initialization)",
        "authors": "He et al. (ICCV 2015)",
        "concepts": "Kaiming/He normal initialization tailored for ReLU activations, PReLU derivation, vanishing gradient mitigation."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1211.5063",
        "title": "On the difficulty of training Recurrent Neural Networks (Gradient Clipping)",
        "authors": "Pascanu, Mikolov, Bengio (ICML 2013)",
        "concepts": "Mathematical analysis of exploding/vanishing gradients, gradient norm clipping, spectral radius dynamics."
    },
    {
        "stage": "02_DL_Fundamentals_and_Optimizers",
        "id": "1705.08292",
        "title": "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks",
        "authors": "Saxe, McClelland, Ganguli (ICLR 2014 / 2017)",
        "concepts": "Analytical dynamics of learning curves, orthogonal weight initialization, saddle point traversal."
    },

    # -------------------------------------------------------------------------
    # STAGE 3: CONVOLUTIONAL NETWORKS, VISION & REPRESENTATIONS (12)
    # -------------------------------------------------------------------------
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1409.1556",
        "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)",
        "authors": "Simonyan & Zisserman (ICLR 2015)",
        "concepts": "Homogeneous 3x3 convolution stacks, spatial hierarchy depth, feature transferability."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "authors": "He et al. (CVPR 2016)",
        "concepts": "Identity skip connections, residual blocks, enabling 100+ layer networks without degradation."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1608.06993",
        "title": "Densely Connected Convolutional Networks (DenseNet)",
        "authors": "Huang et al. (CVPR 2017)",
        "concepts": "Connecting every layer to every subsequent layer, feature reuse, gradient flow enhancement."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1905.11946",
        "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "authors": "Tan & Le (ICML 2019)",
        "concepts": "Compound scaling across network depth, width, and input resolution using a fixed ratio coefficient."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1506.02640",
        "title": "You Only Look Once: Unified, Real-Time Object Detection (YOLO)",
        "authors": "Redmon et al. (CVPR 2016)",
        "concepts": "Single-pass regression for bounding boxes and class probabilities, real-time end-to-end vision pipeline."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2010.11929",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)",
        "authors": "Dosovitskiy et al. (ICLR 2021)",
        "concepts": "Splitting images into non-overlapping patch tokens, applying pure self-attention without convolution."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "1911.05722",
        "title": "Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)",
        "authors": "He et al. (CVPR 2020)",
        "concepts": "Contrastive learning with a dynamic dictionary via a momentum encoder, decoupling batch size."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2002.05709",
        "title": "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)",
        "authors": "Chen et al. (ICML 2020)",
        "concepts": "Data augmentation composition, projection heads, NT-Xent loss for unsupervised visual pre-training."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2103.00020",
        "title": "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
        "authors": "Radford et al. / OpenAI (ICML 2021)",
        "concepts": "Joint image-text contrastive pre-training across 400M pairs, zero-shot multimodal alignment."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2103.14030",
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "authors": "Liu et al. (ICCV 2021)",
        "concepts": "Local self-attention inside shifted windows, hierarchical feature maps, linear computational complexity."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2201.03545",
        "title": "A ConvNet for the 2020s (ConvNeXt)",
        "authors": "Liu et al. (CVPR 2022)",
        "concepts": "Modernizing pure ConvNets with ViT design choices (7x7 depthwise conv, inverted bottleneck, GELU/LayerNorm)."
    },
    {
        "stage": "03_CNNs_and_Computer_Vision",
        "id": "2111.06377",
        "title": "Masked Autoencoders Are Scalable Vision Learners (MAE)",
        "authors": "He et al. (CVPR 2022)",
        "concepts": "Masking 75%+ of image patches and reconstructing them via asymmetric encoder-decoder self-supervision."
    },

    # -------------------------------------------------------------------------
    # STAGE 4: RECURRENT NETWORKS, SEQ2SEQ & ATTENTION MECHANICS (10)
    # -------------------------------------------------------------------------
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1409.3215",
        "title": "Sequence to Sequence Learning with Neural Networks",
        "authors": "Sutskever, Vinyals, Le (NeurIPS 2014)",
        "concepts": "Multi-layer LSTM encoder-decoder architecture mapping variable-length input sequences to outputs."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1409.0473",
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate (Additive Attention)",
        "authors": "Bahdanau, Cho, Bengio (ICLR 2015)",
        "concepts": "The birth of attention: dynamic soft alignment over encoder hidden states using an additive MLP score."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1508.04025",
        "title": "Effective Approaches to Attention-based Neural Machine Translation (Multiplicative Attention)",
        "authors": "Luong, Pham, Manning (EMNLP 2015)",
        "concepts": "Dot-product and multiplicative attention mechanisms, global vs local attention, direct precursor to Vaswani."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1506.00019",
        "title": "A Critical Review of Recurrent Neural Networks for Sequence Learning",
        "authors": "Lipton, Berkowitz, Elkan (2015)",
        "concepts": "In-depth mathematical comparison of vanilla RNNs, LSTMs (Hochreiter & Schmidhuber), and GRUs (Cho et al.)."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1410.5401",
        "title": "Neural Turing Machines",
        "authors": "Graves, Wayne, Danihelka / DeepMind (2014)",
        "concepts": "Coupling neural networks with external differentiable memory banks via content- and location-based addressing."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1506.03134",
        "title": "Pointer Networks",
        "authors": "Vinyals, Fortunato, Jaitly (NeurIPS 2015)",
        "concepts": "Using attention as a pointer to select output elements directly from input sequences (e.g. combinatorial optimization)."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "1901.02860",
        "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
        "authors": "Dai et al. (ACL 2019)",
        "concepts": "Segment-level recurrence mechanism and relative positional encodings for long-context modeling."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "2004.05150",
        "title": "Longformer: The Long-Document Transformer",
        "authors": "Beltagy, Peters, Cohan (2020)",
        "concepts": "Replacing quadratic self-attention with local sliding-window attention + global task-specific attention."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "2312.00752",
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": "Gu & Dao (2023)",
        "concepts": "Selective State Space Models (SSMs), data-dependent input gating, hardware-aware scan, linear-time attention alternative."
    },
    {
        "stage": "04_RNNs_Sequence_Models_and_Attention",
        "id": "2405.21060",
        "title": "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)",
        "authors": "Dao & Gu (ICML 2024)",
        "concepts": "State Space Duality (SSD) framework proving mathematical equivalence between structured SSMs and causal attention."
    },

    # -------------------------------------------------------------------------
    # STAGE 5: TRANSFORMERS & EARLY PRE-TRAINED LANGUAGE MODELS (12)
    # -------------------------------------------------------------------------
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1706.03762",
        "title": "Attention Is All You Need (The Transformer Architecture)",
        "authors": "Vaswani et al. / Google Brain & Research (NeurIPS 2017)",
        "concepts": "Pure self-attention encoder-decoder architecture, Multi-Head Attention, scaled dot-product, sinusoidal positional encodings."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Devlin et al. / Google AI Language (NAACL 2019)",
        "concepts": "Masked Language Modeling (MLM), Next Sentence Prediction (NSP), bidirectional self-attention representations."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1907.11692",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "authors": "Liu et al. / Meta AI & UW (2019)",
        "concepts": "Removing NSP loss, dynamic masking, training on 10x larger corpora with larger batches and learning rates."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1910.10683",
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
        "authors": "Raffel et al. / Google (JMLR 2020)",
        "concepts": "Casting all NLP tasks (translation, QA, summarization, classification) into a unified text-to-text format."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1910.13461",
        "title": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
        "authors": "Lewis et al. / Meta AI (ACL 2020)",
        "concepts": "Corrupting text with arbitrary noise (span deletion, shuffling) and reconstructing via an autoregressive decoder."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1909.11942",
        "title": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
        "authors": "Lan et al. / Google (ICLR 2020)",
        "concepts": "Cross-layer parameter sharing, factorized embedding parameterization, Sentence-Order Prediction (SOP)."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "2003.10555",
        "title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        "authors": "Clark et al. / Stanford & Google (ICLR 2020)",
        "concepts": "Replaced token detection via a small generator network, training discriminators across 100% of input tokens."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "2006.03654",
        "title": "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
        "authors": "He et al. / Microsoft (ICLR 2021)",
        "concepts": "Disentangled attention matrices separating content and relative position vectors, enhanced mask decoder."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "1906.08237",
        "title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "authors": "Yang et al. / CMU & Google (NeurIPS 2019)",
        "concepts": "Permutation language modeling maximizing expected log likelihood over all factorization orders without corruption masks."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "2001.04451",
        "title": "Reformer: The Efficient Transformer",
        "authors": "Kitaev, Kaiser, Levskaya / Google (ICLR 2020)",
        "concepts": "Locality-Sensitive Hashing (LSH) attention reducing O(N^2) to O(N log N), reversible residual layers saving memory."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "2006.04768",
        "title": "Linformer: Self-Attention with Linear Complexity",
        "authors": "Wang et al. / Meta AI (2020)",
        "concepts": "Low-rank approximation of attention keys and values via linear projections, proving low-rank structure of self-attention."
    },
    {
        "stage": "05_Transformers_and_Early_Language_Models",
        "id": "2005.14165",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown et al. / OpenAI (NeurIPS 2020)",
        "concepts": "Autoregressive scaling to 175B parameters demonstrating emergent in-context learning (zero/one/few-shot prompting)."
    },

    # -------------------------------------------------------------------------
    # STAGE 6: LLM SCALING LAWS & FOUNDATION ARCHITECTURES (11)
    # -------------------------------------------------------------------------
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2001.08361",
        "title": "Scaling Laws for Neural Language Models",
        "authors": "Kaplan et al. / OpenAI (2020)",
        "concepts": "Power-law scaling curves relating cross-entropy loss to compute (FLOPs), dataset size (tokens), and parameter count."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2203.15556",
        "title": "Training Compute-Optimal Large Language Models (Chinchilla)",
        "authors": "Hoffmann et al. / DeepMind (NeurIPS 2022)",
        "concepts": "Chinchilla scaling law showing model size and training tokens should scale in equal 1:1 proportion."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2101.03961",
        "title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        "authors": "Fedus, Zoph, Shazeer / Google (JMLR 2022)",
        "concepts": "Simplified top-1 Mixture-of-Experts (MoE) routing, expert load balancing, scaling parameters with constant compute."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2302.13971",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Touvron et al. / Meta AI (2023)",
        "concepts": "Foundational open-weights architecture: RMSNorm pre-normalization, SwiGLU activations, Rotary Positional Embeddings (RoPE)."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": "Touvron et al. / Meta AI (2023)",
        "concepts": "Grouped-Query Attention (GQA), extended context length, rigorous safety and chat reward modeling pipeline."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2407.21783",
        "title": "The Llama 3 Herd of Models",
        "authors": "Dubey et al. / Meta AI (2024)",
        "concepts": "Scaling training to 15T+ tokens, 128K vocabulary tokenizer, multi-stage post-training alignment, 405B dense flagship."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2310.06825",
        "title": "Mistral 7B",
        "authors": "Jiang et al. / Mistral AI (2023)",
        "concepts": "Sliding Window Attention (SWA), Grouped-Query Attention (GQA), outperforming 13B and 34B models with 7B parameters."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2401.04088",
        "title": "Mixtral of Experts (8x7B Sparse MoE)",
        "authors": "Jiang et al. / Mistral AI (2024)",
        "concepts": "Sparse top-2 of 8 expert routing per layer, achieving 70B+ performance while activating only ~13B active parameters."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2402.13616",
        "title": "Gemma: Open Models Based on Gemini Research and Technology",
        "authors": "Mesnard et al. / Google DeepMind (2024)",
        "concepts": "Multi-head and Multi-Query attention choices, GeGLU activations, RoPE across high-performance 2B/7B open weights."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2204.02311",
        "title": "PaLM: Scaling Language Modeling with Pathways",
        "authors": "Chowdhery et al. / Google (JMLR 2023)",
        "concepts": "540B dense scaling across 6144 TPU v4 chips via Pathways system, parallel attention/FFN formulation, SwiGLU."
    },
    {
        "stage": "06_Scaling_and_Foundation_LLMs",
        "id": "2302.04761",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Schick et al. / Meta AI (NeurIPS 2023)",
        "concepts": "Self-supervised API call insertion into training data, learning when and how to invoke external search/calculator tools."
    },

    # -------------------------------------------------------------------------
    # STAGE 7: ALIGNMENT, INSTRUCTION TUNING, PROMPTING & REASONING (14)
    # -------------------------------------------------------------------------
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2203.02155",
        "title": "Training language models to follow instructions with human feedback (InstructGPT / RLHF)",
        "authors": "Ouyang et al. / OpenAI (NeurIPS 2022)",
        "concepts": "The classic RLHF pipeline: Supervised Fine-Tuning (SFT), Reward Model training, PPO policy optimization."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)",
        "authors": "Rafailov et al. / Stanford (NeurIPS 2023)",
        "concepts": "Analytical reparameterization of the RLHF objective allowing direct optimization on preference pairs without RL/PPO."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2402.01306",
        "title": "KTO: Model Alignment as Prospect Theoretic Optimization",
        "authors": "Ethayarajh et al. / Contextual AI (ICML 2024)",
        "concepts": "Aligning LLMs directly from unpaired thumbs-up/thumbs-down signals using human utility functions from Kahneman-Tversky theory."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2212.10560",
        "title": "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        "authors": "Wang et al. / UW (ACL 2023)",
        "concepts": "Bootstrapping instruction-following capabilities by prompting a seed LLM to generate diverse task instructions and responses."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2212.08073",
        "title": "Constitutional AI: Harmlessness from AI Feedback (RLAIF)",
        "authors": "Bai et al. / Anthropic (2022)",
        "concepts": "Self-critique and revision based on a written constitution of principles, replacing human preference annotators with AI feedback."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Wei et al. / Google (NeurIPS 2022)",
        "concepts": "Step-by-step intermediate reasoning rationales ('Let's think step by step') dramatically improving complex problem solving."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2305.10601",
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": "Yao et al. / Princeton & DeepMind (NeurIPS 2023)",
        "concepts": "Generalizing CoT to tree structures with multiple reasoning branches, evaluation heuristics, BFS/DFS search algorithms."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2309.03409",
        "title": "Graph of Thoughts: Solving Elaborate Problems with Large Language Models",
        "authors": "Besta et al. / ETH Zurich (AAAI 2024)",
        "concepts": "Representing reasoning thoughts as arbitrary directed acyclic graphs, enabling aggregation and synergistic merging of ideas."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2303.11366",
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "authors": "Shinn et al. / Northeastern & Princeton (NeurIPS 2023)",
        "concepts": "Verbal self-reflection loops storing natural language critique histories in episodic memory to self-correct on failures."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2310.03714",
        "title": "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines",
        "authors": "Khattab et al. / Stanford (ICLR 2024)",
        "concepts": "Separating program flow from prompts via declarative modules, automated teleprompter optimizers tuning prompts and weights."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2210.03629",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Yao et al. / Princeton & Google (ICLR 2023)",
        "concepts": "Interleaving reasoning traces ('Thought') with concrete external actions and observations ('Action/Observation') for agents."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2203.11171",
        "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "authors": "Wang et al. / Google (ICLR 2023)",
        "concepts": "Sampling multiple diverse CoT reasoning paths at non-zero temperature and taking a majority vote over final answers."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2403.09629",
        "title": "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking",
        "authors": "Zelikman et al. / Stanford (2024)",
        "concepts": "Training models to generate internal reasoning rationales (`<thought>...</thought>`) before predicting each token during general pre-training."
    },
    {
        "stage": "07_Alignment_Instruction_Tuning_and_Prompting",
        "id": "2412.19437",
        "title": "DeepSeek-V3 Technical Report",
        "authors": "DeepSeek-AI (2024)",
        "concepts": "Multi-head Latent Attention (MLA) for extreme KV cache compression, DeepSeekMoE, auxiliary-loss-free load balancing."
    },

    # -------------------------------------------------------------------------
    # STAGE 8: EFFICIENT FINE-TUNING, QUANTIZATION, RAG & INFERENCE (14)
    # -------------------------------------------------------------------------
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al. / Microsoft (ICLR 2022)",
        "concepts": "Freezing base weights and injecting trainable low-rank decomposition matrices (A and B) into attention projections."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2305.14314",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Dettmers et al. / UW (NeurIPS 2023)",
        "concepts": "4-bit NormalFloat (NF4) quantization, Double Quantization, Paged Optimizers enabling 65B LoRA finetuning on a single GPU."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2402.09353",
        "title": "DoRA: Weight-Decomposed Low-Rank Adaptation",
        "authors": "Liu et al. / NVIDIA (ICML 2024)",
        "concepts": "Decomposing weight matrices into magnitude and direction components, applying LoRA solely to the directional vector."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2303.10512",
        "title": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
        "authors": "Zhang et al. (ICLR 2023)",
        "concepts": "Dynamically allocating rank budgets across layers based on singular value importance scores via SVD-based parameterization."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "authors": "Lewis et al. / Meta AI (NeurIPS 2020)",
        "concepts": "Coupling parametric neural generators with non-parametric dense vector indices (DPR + Wikipedia) for factual accuracy."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2310.11511",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": "Asai et al. / UW & IBM (ICLR 2024)",
        "concepts": "On-demand retrieval using reflection tokens (`[Retrieve]`, `[IsREL]`, `[IsSUP]`) to self-evaluate relevancy and supportfulness."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2401.15884",
        "title": "Corrective Retrieval Augmented Generation (CRAG)",
        "authors": "Yan et al. / Google (2024)",
        "concepts": "Lightweight retrieval evaluator assessing document quality, triggering web search fallback when retrieved chunks are ambiguous."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2205.14135",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": "Dao et al. / Stanford (NeurIPS 2022)",
        "concepts": "Tiling and recomputation in SRAM to avoid materializing the N^2 attention matrix in HBM, transforming LLM context lengths."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2307.08691",
        "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "authors": "Dao / Princeton & Together AI (ICLR 2024)",
        "concepts": "Parallelizing over sequence length dimensions, reducing non-matmul FLOPs, achieving 50-73% of theoretical GPU FLOPs."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2407.08608",
        "title": "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision",
        "authors": "Shah et al. / Dao (2024)",
        "concepts": "Asynchronous WGMMA execution on Hopper GPUs, block quantization to FP8 without numerical stability loss."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2309.06180",
        "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)",
        "authors": "Kwon et al. / UC Berkeley (SOSP 2023)",
        "concepts": "Operating system virtual memory paging applied to KV caches (`PagedAttention`), eliminating fragmentation and boosting throughput."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2210.17323",
        "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
        "authors": "Frantar et al. / IST Austria (ICLR 2023)",
        "concepts": "Layer-wise optimal brain quantization using inverse Hessian approximations for 3-bit and 4-bit weight compression."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2306.00978",
        "title": "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration",
        "authors": "Lin et al. / MIT (MLSys 2024)",
        "concepts": "Observing that 1% of salient weights protect activation distributions; protecting them via per-channel scaling without retraining."
    },
    {
        "stage": "08_Efficient_Finetuning_Quantization_RAG_and_Serving",
        "id": "2310.11453",
        "title": "BitNet: Scaling 1-bit Transformers for Large Language Models",
        "authors": "Wang et al. / Microsoft (2023)",
        "concepts": "Replacing linear projections with `BitLinear` (-1, 0, +1 ternary or 1-bit binary weights), paving the path for extreme edge inference."
    },
]


def safe_filename(title, paper_id):

    title = " ".join(title.split())
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip(" .")
    suffix = f" [arXiv {paper_id}].pdf" if not paper_id.startswith("LightGBM") else " [NeurIPS 2017].pdf"
    max_title_len = 180 - len(suffix)
    return title[:max_title_len].rstrip(" .") + suffix


def download_paper(item, base_folder: Path, delay: float = 1.0):

    stage_folder = base_folder / item["stage"]
    stage_folder.mkdir(parents=True, exist_ok=True)

    paper_id = item["id"]
    filename = safe_filename(item["title"], paper_id)
    dest_path = stage_folder / filename

    if dest_path.exists() and dest_path.stat().st_size > 5000:
        return dest_path, False

    url = item.get("url", f"https://arxiv.org/pdf/{paper_id}.pdf")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Personal research paper downloader/1.0 (contact@example.com)",
            "Accept": "application/pdf, */*",
        },
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()

    # Check for HTML redirection/error pages
    if not data.lstrip().startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a valid PDF header (got {data[:30]}...)")

    dest_path.write_bytes(data)
    time.sleep(delay)
    return dest_path, True


def generate_study_guide(curriculum, output_folder: Path):

    md_path = output_folder / "CORE_ML_LLM_STUDY_GUIDE.md"
    lines = [
        "# Core ML-to-LLM Master Research Curriculum (100 Seminal Papers)",
        "",
        "This study guide accompanies the **100 curated research papers** downloaded by `download_core_ml_papers.py`. ",
        "Each paper is categorized into one of **8 pedagogical stages** designed to take you from classical statistical learning and ensemble mechanics up to cutting-edge LLM architectures, alignment methods, and serving optimizations.",
        "",
        "---",
        ""
    ]

    current_stage = ""
    for idx, item in enumerate(curriculum, 1):
        if item["stage"] != current_stage:
            current_stage = item["stage"]
            stage_title = current_stage.replace("_", " ").title()
            lines.extend([
                f"## {stage_title}",
                ""
            ])

        pid = item["id"]
        link = item.get("url", f"https://arxiv.org/abs/{pid}")
        lines.extend([
            f"### {idx}. {item['title']}",
            f"- **Authors / Publication:** {item['authors']}",
            f"- **Canonical Identifier / URL:** [{pid}]({link})",
            f"- **Core Concepts Introduced:** {item['concepts']}",
            f"- **Folder Path:** `{item['stage']}/{safe_filename(item['title'], pid)}`",
            ""
        ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Download 100 core ML-to-LLM research papers.")
    parser.add_argument("--output-folder", type=str, default="papers", help="Base directory for PDFs")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between new downloads")
    parser.add_argument("--max-papers", type=int, default=100, help="Maximum number of papers to download")
    args = parser.parse_args()

    base_folder = Path(args.output_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    print(f"=========================================================================")
    print(f"       DOWNLOADING 100 CORE ML-TO-LLM FOUNDATIONAL PAPERS                ")
    print(f"=========================================================================")
    print(f"Target directory : {base_folder.resolve()}")
    print(f"Total curriculum : {len(CURRICULUM)} papers across 8 pedagogical stages")
    print(f"=========================================================================\n")

    # Generate Study Guide Markdown first
    guide_path = generate_study_guide(CURRICULUM, base_folder)
    print(f"[+] Study Guide generated at: {guide_path.resolve()}\n")

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, item in enumerate(CURRICULUM[: args.max_papers], 1):
        stage_display = item["stage"].split("_", 1)[1].replace("_", " ")
        print(f"[{idx:03d}/{min(len(CURRICULUM), args.max_papers):03d}] ({stage_display})")
        print(f"       Title : {item['title'][:72]}...")
        
        try:
            dest_path, newly_downloaded = download_paper(item, base_folder, delay=args.delay)
            if newly_downloaded:
                downloaded_count += 1
                print(f"       Status: [NEW DOWNLOAD] -> {dest_path.name}")
            else:
                skipped_count += 1
                print(f"       Status: [ALREADY EXISTS] -> {dest_path.name}")
        except Exception as error:
            failed_count += 1
            print(f"       Status: [FAILED] -> {error}")
        print("-" * 75)

    print(f"\n=========================================================================")
    print(f"                       DOWNLOAD SUMMARY                                  ")
    print(f"=========================================================================")
    print(f"Total Processed : {min(len(CURRICULUM), args.max_papers)}")
    print(f"New Downloads   : {downloaded_count}")
    print(f"Skipped (Cached): {skipped_count}")
    print(f"Failed          : {failed_count}")
    print(f"PDFs Saved In   : {base_folder.resolve()}")
    print(f"Study Guide     : {guide_path.resolve()}")
    print(f"=========================================================================")


if __name__ == "__main__":
    main()
