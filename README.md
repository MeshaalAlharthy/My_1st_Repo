# 🔒 Transactions Sensitivity Classification for Emirate of Makkah

An advanced Arabic document classification project to predict the sensitivity level of Emirate of Makkah transaction documents. This system aims to support secure data governance and assist in quick, informed decision-making.

---

## Project Overview

This project goes beyond a standard fine-tuning of AraBERT. It introduces multiple new ideas to make the model suitable for real-world governmental use cases.

---

## Customizations and Innovations

- Special token [SEC]: Added to highlight important keywords related to security and improve context understanding.
- Admin-aware metadata integration: The الإدارة (Administration) column was encoded using one-hot encoding and passed through a custom neural network layer, so the model learns organizational context, not just text.
- Dual embedding strategy: We combined the CLS token embedding and mean-pooled embeddings to better represent full-document semantics.
- Balanced focal loss: A custom loss function with class weights to handle data imbalance and improve predictions for rare classes like "Top Secret".
- LoRA fine-tuning: Used Low-Rank Adaptation (LoRA) for efficient fine-tuning, reducing training costs while keeping high accuracy.

---

## Classification Labels

The model classifies documents into four levels:

- Top Secret (سري للغاية)
- Secret (سري)
- Restricted (مقيد)
- Public (عام)

---

## Dataset

Source: Emirate of Makkah transaction data.

Text columns included:

- Keywords and unique identifiers
- Process description
- Expected impact (financial, reputation, health, safety, operational, security, stakeholders)
- Type of personal data
- Related forms

Additional metadata column:

-  (Administration), used as extra features to improve prediction accuracy.

---

## Technical Highlights

- Base model: aubmindlab/bert-base-arabertv02
- One-hot encoded admin metadata integrated with text embeddings
- Custom balanced focal loss to solve imbalance issues
- LoRA for efficient and lightweight parameter tuning
- Advanced preprocessing combining multiple Arabic text columns

---

## Results

- Weighted F1-score above 94%
- Improved performance on rare classes
- High robustness against noisy real-world data
- Ready to support secure classification workflows in practical scenarios

---

## Usage

### Installation

```bash
pip install torch torchvision torchaudio transformers datasets accelerate peft evaluate scikit-learn sentencepiece
