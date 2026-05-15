# HOMO-LUMO Property Cliff Detection using Graph Neural Networks

This project explores molecular property cliff detection on the QM9 dataset using Graph Neural Networks (GNNs).  
The goal is to identify structurally similar molecule pairs that exhibit large differences in HOMO-LUMO gap values.

## Models Implemented
- Shared-Encoder GNN (Final Model)
- Dual-Tower GNN
- Morgan Fingerprint + Random Forest
- Morgan Fingerprint + MLP
- TF-IDF SMILES + Logistic Regression

## Evaluation Metrics Used
- PR-AUC
- ROC-AUC
- F1 Score
- Precision
- Recall
