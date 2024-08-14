# Sentiment Analysis Using TinyBERT

This repository contains two Jupyter notebooks designed to perform sentiment analysis using TinyBERT, a lightweight version of BERT (Bidirectional Encoder Representations from Transformers). The notebooks focus on leveraging the pretrained TinyBERT model for inference and training on a sentiment analysis dataset.

## Notebooks

### 1. `sentiment_inference.ipynb`
This notebook is focused on the inference process using a pretrained TinyBERT model for sentiment analysis. It includes the following steps:
- **Model Loading:** Load the pretrained TinyBERT model and tokenizer.
- **Data Preprocessing:** Prepare the input text data for sentiment analysis by tokenizing it into a format suitable for TinyBERT.
- **Sentiment Inference:** Perform inference on the input text to predict sentiment (e.g., positive, negative, neutral).
- **Results Interpretation:** Display and interpret the sentiment predictions.

### 2. `TinyBERT_Sentiment.ipynb`
This notebook is dedicated to training and fine-tuning TinyBERT on a custom sentiment analysis dataset. The key steps include:
- **Dataset Loading:** Load and explore the sentiment analysis dataset.
- **Data Preprocessing:** Tokenize and preprocess the dataset to make it compatible with TinyBERT.
- **Model Training:** Fine-tune the TinyBERT model on the training dataset.
- **Evaluation:** Evaluate the model's performance on the test set, including metrics like accuracy, precision, recall, and F1-score.
- **Inference:** Perform sentiment analysis on new text data using the fine-tuned TinyBERT model.

## Requirements

To run these notebooks, you need to install the following dependencies:

- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `jupyter`

You can install the required packages using the following command:

pip install transformers torch pandas scikit-learn numpy matplotlib seaborn jupyter

## Usage
