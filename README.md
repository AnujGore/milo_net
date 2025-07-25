# MILO

**Matrix-Input Linear Operator**

---

## 📌 Overview

**MILO** is an experimental neural network module that processes **2D input data directly**, without flattening it into a vector. Instead, the input matrix is **nestled between two learnable weight matrices**, preserving its inherent 2D structure and enabling richer, more expressive transformations.

---

## ✨ Why MILO?

- **No flattening:** Keep your data in its natural 2D form — like images, grids, or relational matrices.
- **Bilinear transformation:** Computes `Y = W1 * X * W2ᵗ` for flexible 2-sided learning.
---

## 🗂️ Project Structure

```plaintext
MILO/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── model.py      # MILO layer definition
│   ├── train.py      # Training loop
│   ├── main.py       
│   └── utils.py      # Helper functions
│
├── notebooks/
│   └── main.ipynb
```


## Updates

### July 25, 2025

The basic version of MILO is ready. It's trained on MNIST dataset and outputs 10 probabilities. The argmax is selected as the predicted label. Next step would be to develop this further into a ping-pong model where multiple MILONets of opposing structure should have the same input and output respectively, like a horizontal hour-glass. The lowest dimensional network can hopefully be used to generate some nice images.
