# CIFAR-10 Image Classification with PyTorch & Streamlit

## Overview
This project demonstrates an end-to-end deep learning pipeline using the CIFAR-10 dataset.  
It includes training a CNN with PyTorch and deploying it with Streamlit.

## Structure
- **notebooks/cifar10_training.ipynb** — Model training and evaluation.
- **streamlit_app.py** — Web app for interactive image classification.
- **models/cifar_net.pth** — Saved trained model weights.
- **requirements.txt** — Python dependencies.
- **src/** — Helper scripts (data loading, model definition, etc.).

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Classes
['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
