# Sequence and Multimodal Representation Learning Portfolio

This portfolio contains three specialized Machine Learning projects demonstrating state-of-the-art representation learning and sequence modeling architectures for fintech, tailored for the **Applied Research Engineer (Risks)** position at **Fintech Platform**.

The projects demonstrate:
1. **Self-Supervised Sequence Learning (CoLES)**: Contrastive Learning for Event Sequences applied to discrete transaction streams.
2. **Behavioral Transformer (SASRec/BERT4Rec style)**: Self-attention sequence modeling for next-behavior prediction and downstream LTV forecasting.
3. **Multimodal Risk Fusion**: A unified deep learning architecture combining sequence embeddings, tabular metadata, and graph-structured user-merchant interactions for real-time transaction fraud detection.

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
│   └── downstream_eval.py                 <- Evaluates representations on Credit Risk & Fraud
│
├── bert4rec_behavioral_transformer/       <- Project 2: Clickstream Transformer & LTV Forecasting
│   ├── event_generator.py                 <- Clickstream session & event sequence generator
│   ├── transformer_model.py               <- PyTorch SASRec-style causal self-attention model
│   ├── train_transformer.py               <- Auto-regressive next-event sequence training
│   └── ltv_forecasting.py                 <- Predicts LTV from pre-trained sequence embeddings
│
└── multimodal_risk_fusion/                <- Project 3: Tabular, Sequential, and Graph Fusion
    ├── multimodal_model.py                <- PyTorch model fusing MLP, Sequence, and Graph embeddings
    └── train_fusion.py                    <- End-to-end training pipeline for Transaction Anti-fraud
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
First, pre-train the transactional encoder using self-supervised contrastive learning:
```bash
python coles_transaction_pretraining/train_coles.py
```
This will pre-train the sequence encoder using a simulated credit card transaction log and log model configurations to MLflow. Then, run the downstream evaluation to test default and fraud prediction:
```bash
python coles_transaction_pretraining/downstream_eval.py
```

#### 2. Clickstream Transformer & LTV Forecasting
Train the behavioral transformer to predict next-user actions:
```bash
python bert4rec_behavioral_transformer/train_transformer.py
```
Once pre-trained, forecast Customer Lifetime Value using the trained embeddings:
```bash
python bert4rec_behavioral_transformer/ltv_forecasting.py
```

#### 3. Multimodal Risk Fusion
Train the unified multimodal network (tabular device data, sequential transactions, user-merchant graph representation) for transaction fraud detection:
```bash
python multimodal_risk_fusion/train_fusion.py
```

---

## Detailed Tech Stack & Methodology

### Project 1: Contrastive Learning for Event Sequences (CoLES)
- **Methodology**: Contrastive Learning for Event Sequences (CoLES) treats sub-sequences of transactions from the same user as positive pairs, and sub-sequences from different users as negative pairs. This teaches the sequence encoder (implemented as a Bidirectional GRU or Transformer) to capture the invariant behavioral signatures of a user.
- **Fintech Value**: Replaces manual feature engineering (e.g., aggregate sums, counts) with rich, multi-dimensional embedding vectors. Downstream classifiers (like XGBoost or Logistic Regression) can ingest these embeddings to predict default risk and anti-fraud with higher AUC and fewer hand-crafted features.

### Project 2: Behavioral Transformer (SASRec/BERT4Rec)
- **Methodology**: Implements a self-attention transformer that models clickstream activity (actions like checking limits, viewing cards, chatting with support). It learns to predict the next actions, creating high-level representations of user journey states.
- **Fintech Value**: These sequence representations capture user intent, allowing precise forecasting of downstream metrics like LTV, churn, or propensity to buy premium products.

### Project 3: Multimodal Risk Fusion
- **Methodology**: Combines distinct sources of information into a single PyTorch model:
  1. **Sequential**: Real-time transaction sequences processed by a recurrent/attention encoder.
  2. **Tabular**: Static/dynamic context (device OS, VPN usage, location metadata) processed by a multi-layer perceptron.
  3. **Graph**: Bipartite relations (user interacting with specific merchants) represented as graph node embeddings.
- **Fintech Value**: Essential for transaction anti-fraud systems, where fraud is rarely defined by transaction amount alone, but rather by the intersection of suspicious device attributes, sequence abnormalities, and untrusted merchants in the transaction graph.
