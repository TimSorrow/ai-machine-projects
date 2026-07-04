# Sequence and Multimodal Representation Learning Portfolio

This repository contains a portfolio of three specialized Machine Learning projects demonstrating advanced representation learning, sequence modeling, and multimodal architectures for transaction and event data.

The projects include:
1. **Self-Supervised Sequence Learning (CoLES)**: Contrastive Learning for Event Sequences (CoLES) applied to discrete transaction streams.
2. **Behavioral Transformer (SASRec/BERT4Rec style)**: Self-attention sequence modeling for next-behavior prediction and downstream regression modeling (such as Customer Value forecasting).
3. **Multimodal Risk Fusion**: A unified deep learning architecture in PyTorch combining sequence embeddings, tabular metadata, and graph-structured user-merchant interactions for anomaly/fraud detection.

---

## Portfolio Structure

```text
ai-machine-projects/
├── README.md                              <- This portfolio summary & run instructions
├── coles_transaction_pretraining/          <- Project 1: CoLES Transaction Sequence Pre-training
│   ├── data_generator.py                  <- Synthesizes realistic transactional sequences
│   ├── dataset.py                         <- Sliding window sequence splitter for contrastive views
│   ├── model.py                           <- PyTorch GRU/Transformer sequence encoder
│   ├── train_coles.py                     <- Self-supervised contrastive training (InfoNCE/Cosine)
│   └── downstream_eval.py                 <- Evaluates representations on downstream tasks
│
├── bert4rec_behavioral_transformer/       <- Project 2: Clickstream Transformer & Value Forecasting
│   ├── event_generator.py                 <- Clickstream session & event sequence generator
│   ├── transformer_model.py               <- PyTorch SASRec-style causal self-attention model
│   ├── train_transformer.py               <- Auto-regressive next-event sequence training
│   └── ltv_forecasting.py                 <- Predicts customer value from pre-trained sequence embeddings
│
└── multimodal_risk_fusion/                <- Project 3: Tabular, Sequential, and Graph Fusion
    ├── multimodal_model.py                <- PyTorch model fusing MLP, Sequence, and Graph embeddings
    └── train_fusion.py                    <- End-to-end training pipeline for multi-modal risk detection
```

---

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch mlflow scikit-learn numpy pandas
```

### Running the Projects

#### 1. CoLES Transaction Pre-Training
Pre-train the transactional encoder using self-supervised contrastive learning:
```bash
python coles_transaction_pretraining/train_coles.py
```
This pre-trains the sequence encoder using simulated transaction logs and records parameters/metrics to MLflow. Then, run the downstream evaluation:
```bash
python coles_transaction_pretraining/downstream_eval.py
```

#### 2. Clickstream Transformer & Value Forecasting
Train the behavioral transformer to predict next-user actions:
```bash
python bert4rec_behavioral_transformer/train_transformer.py
```
Once pre-trained, forecast customer value using the trained embeddings:
```bash
python bert4rec_behavioral_transformer/ltv_forecasting.py
```

#### 3. Multimodal Risk Fusion
Train the unified multimodal network (combining tabular device data, sequential transactions, and user-merchant graph representation) for anomaly detection:
```bash
python multimodal_risk_fusion/train_fusion.py
```

---

## Detailed Tech Stack & Methodology

### Project 1: Contrastive Learning for Event Sequences (CoLES)
- **Methodology**: Contrastive Learning for Event Sequences (CoLES) treats sub-sequences of events from the same user as positive pairs, and sub-sequences from different users as negative pairs. This teaches the sequence encoder (implemented as a Bidirectional GRU or Transformer) to capture invariant behavioral signatures.
- **Application & Value**: Replaces manual feature engineering (e.g., aggregate statistics) with rich, multi-dimensional embedding vectors. Downstream classifiers can ingest these embeddings to predict targets (such as credit defaults or fraudulent activities) with higher AUC and fewer hand-crafted features.

### Project 2: Behavioral Transformer (SASRec/BERT4Rec)
- **Methodology**: Implements a self-attention transformer that models clickstream activity (user journeys and events). It learns to predict the next actions, creating high-level representations of user behavior states.
- **Application & Value**: These sequence representations capture user intent, allowing precise forecasting of downstream metrics like customer lifetime value, churn, or conversion propensity.

### Project 3: Multimodal Risk Fusion
- **Methodology**: Combines distinct sources of information into a single PyTorch model:
  1. **Sequential**: Real-time action sequences processed by a recurrent encoder.
  2. **Tabular**: Static/dynamic metadata context processed by a multi-layer perceptron.
  3. **Graph**: Bipartite interaction relations (e.g. user-merchant connections) represented as graph node embeddings.
- **Application & Value**: Essential for complex risk and fraud systems, where predictions are rarely defined by transaction attributes alone, but rather by the intersection of suspicious device attributes, sequence anomalies, and relation graphs.
