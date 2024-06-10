# Fashion Recommendation System

## Overview

This project aims to develop a Fashion Recommendation System utilizing various machine learning models including Convolutional Neural Networks (CNN), Bayesian Regression, Singular Value Decomposition (SVD), and Ensemble Trees. The dataset used for training and evaluation is the MNIST dataset, which consists of grayscale images of fashion items.

## Project Motive

The primary objective of this project is to build an efficient and accurate recommendation system that can predict and recommend fashion items to users based on their preferences. By leveraging different machine learning techniques, we aim to compare the performance of these models and determine the most effective approach. From our experiments, the CNN model has shown the best performance among the techniques employed.

## Models Used

1. **Convolutional Neural Networks (CNN)**
2. **Bayesian Regression**
3. **Singular Value Decomposition (SVD)**
4. **Ensemble Trees**

### Performance Summary

- **Convolutional Neural Networks (CNN)**: Achieved the highest accuracy and provided the best recommendations, outperforming other models.
- **Bayesian Regression**: Demonstrated solid performance but fell short compared to CNN.
- **Singular Value Decomposition (SVD)**: Effective in dimensionality reduction and provided decent recommendations.
- **Ensemble Trees**: Provided robust predictions but were not as effective as CNN.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of fashion items, with each image being 28x28 pixels. The dataset is divided into:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image is labeled with a corresponding fashion item category, making it a supervised learning problem.

## Project Structure

- `data/`: Contains the MNIST dataset.
- `models/`: Includes the implementation of different models.
  - `cnn.ipynb`: Implementation of the CNN model.
  - `bayesian_regression.ipynb`: Implementation of Bayesian Regression.
  - `svd.py`: Implementation of SVD.
  - `ensemble_tree.ipynb`: Implementation of Ensemble Trees.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model evaluation.
- `scripts/`: Scripts for data preprocessing and model training.
- `results/`: Contains the results and performance metrics of different models.
- `README.md`: Project documentation.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manvitha2004/fashion-recommendation-system.git
   cd fashion-recommendation-system
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the MNIST dataset:**
   The dataset can be downloaded from [MNIST official site](http://yann.lecun.com/exdb/mnist/).

4. **Preprocess the data:**
   ```bash
   python scripts/preprocess_data.py
   ```

5. **Train the models:**
   ```bash
   python scripts/train_cnn.py
   python scripts/train_bayesian_regression.py
   python scripts/train_svd.py
   python scripts/train_ensemble_tree.py
   ```

## Usage

After training the models, you can evaluate their performance and make predictions using the following commands:

1. **Evaluate CNN model:**
   ```bash
   python scripts/evaluate_cnn.py
   ```

2. **Evaluate Bayesian Regression model:**
   ```bash
   python scripts/evaluate_bayesian_regression.py
   ```

3. **Evaluate SVD model:**
   ```bash
   python scripts/evaluate_svd.py
   ```

4. **Evaluate Ensemble Tree model:**
   ```bash
   python scripts/evaluate_ensemble_tree.py
   ```

## Results

The evaluation results and performance metrics for each model are stored in the `results/` directory. Based on the experiments conducted, the CNN model demonstrated superior performance in terms of accuracy and recommendation quality.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

Thank you for using our Fashion Recommendation System! We hope it helps in making insightful fashion recommendations.
