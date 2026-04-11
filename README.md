# Visual Similarity Retrieval using Siamese Networks

A simple machine learning based web application that uses a Siamese Neural Network to calculate the visual similarity between clothing items. The project features a FastAPI backend, a dynamic drag-and-drop frontend, and an interactive 2D t-SNE visualization of the model's embeddings.

## ✨ Features

* **Similarity Tester:** Upload two images of clothing to instantly calculate their Euclidean distance and verify if they are a visual match.
* **Interactive Dashboard:** Explore the AI's "brain" via a Plotly-powered t-SNE scatter plot. Hover over data points to reveal the exact test images and see how the model organically clustered categories together.
* **Aggressive Preprocessing Pipeline:** Real-time image formatting (resizing, grayscale conversion, color inversion, normalization) to match the model's training parameters.

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn, Python
* **Machine Learning:** TensorFlow, Keras (Siamese Architecture), Scikit-Learn
* **Frontend:** HTML5, CSS3 (CSS Variables Theme), Vanilla JavaScript
* **Data Visualization:** Plotly.js, t-SNE

## 📂 Project Structure

```text
├── data/
│   ├── embeddings_2d.npy       # t-SNE reduced 2D coordinates
│   ├── images_base64.json      # Base64 encoded test images for the dashboard
│   └── y_test.npy              # True labels for the test dataset
├── index.html                  # Main UI template
├── style.css                   # Theme and layout styling
├── script.js                   # Client-side logic and fetching
├── main.py                     # FastAPI server and ML prediction logic
├── fashion_siamese.keras       # Trained model weights (Fashion MNIST)
└── README.md                   # Project documentation

```
## 🚀 Installation & Setup

Follow these steps to set up the project in a completely isolated environment on your local machine.

### 1. Create a Project Folder & Clone
First, create a new folder for the project and navigate into it:
```bash
mkdir fashion-similarity-app
cd fashion-similarity-app
git clone https://github.com/swethanissankara-boop/fashion-similarity-ai.git .
```

### 2. Create a Virtual Environment
It is highly recommended to isolate your project dependencies. Run the command for your operating system:

**For Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**For Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
With your virtual environment activated `(.venv)`, install all the required Machine Learning and Backend libraries in one go:
```bash
pip install fastapi uvicorn python-multipart pillow tensorflow numpy
```

### 4. Start the Application
Spin up the FastAPI server to bring the AI online:
```bash
python -m uvicorn main:app --reload
```
*Wait for the terminal to display `Application startup complete`, then double-click `index.html` to open the app!*

## 🧠 How It Works (The ML Architecture)

Instead of traditional classification, this model utilizes Metric Learning:

1. Twin branch Convolutional Neural Networks (CNNs) extract feature vectors from the input images, compressing them into 128-dimensional embeddings.

2. A custom Lambda layer calculates the absolute difference between these embeddings.

3. The model was trained using a custom Contrastive Loss function on the Fashion MNIST dataset, penalizing the network for pulling dissimilar items close together and rewarding it for clustering similar items.

4. During inference, model.load_weights() is utilized to bypass Keras versioning conflicts, injecting the learned mathematics into a freshly compiled local architecture.

## ⚠️ Limitations & Future Enhancements

Because the base model was trained exclusively on Fashion MNIST (28x28 grayscale images on black backgrounds), the current backend pipeline aggressively forces real-world photos into this specific format. This can result in accuracy drops for highly complex, full-color, or poorly lit real-world photographs.

## Roadmap for V2:

* Swap the custom CNN architecture for a heavy-duty, pre-trained base model (e.g., MobileNetV2, ResNet50).

* Fine-tune on the full-color DeepFashion dataset.

* Implement a Vector Database (e.g., Pinecone, Milvus) to shift the application from 1-to-1 verification to 1-to-Many search retrieval.

* Containerize the application using Docker for cloud deployment.

## 🤝 Contributing

Contributions, issues, and feature requests are always welcome! 
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgements

This project was made possible by the following open-source tools and datasets:
* [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research for the training data.
* [Plotly.js](https://plotly.com/javascript/) for powering the interactive 2D t-SNE scatter plots.
* [FastAPI](https://fastapi.tiangolo.com/) for the lightning-fast Python backend.

