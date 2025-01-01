

# Face Recognition using Pre-trained VGG Model

## Objective
The objective of this project is to recognize aligned faces from a dataset containing over 10,000 images of 100 individuals. A pre-trained VGG model for face recognition was used to generate embeddings for each image, followed by classification using SVM. The final model achieved an accuracy of over 96%.

---

## Dataset Description
- The dataset contains over 10,000 images of 100 individuals.
- Each image is pre-aligned and resized to a uniform size.
- Images are labeled with the name of the person they depict.

---

## Methodology

### 1. Preprocessing
- **Face Embeddings:** A pre-trained VGG model with pre-trained weights was used to extract 2,622-dimensional embeddings for each image.
- **Distance Visualization:** The pairwise distances between embeddings were calculated and visualized to assess separability.

### 2. Dataset Splitting
The dataset was split into training and testing sets using the following indices:
- **Training Set:** Images where `(index % 9 != 0)`.
- **Testing Set:** Images where `(index % 9 == 0)`.

Resulting shapes:
- Training Features (X_train): (9573, 2622)
- Testing Features (X_test): (1197, 2622)
- Training Labels (y_train): (9573, )
- Testing Labels (y_test): (1197, )

### 3. Label Encoding
- Labels were encoded using `LabelEncoder`.
- Encoded labels were standardized using `StandardScaler`.

### 4. Dimensionality Reduction
- **Principal Component Analysis (PCA):**
  - The covariance matrix of the standardized features was calculated.
  - Eigenvalues and eigenvectors were derived to determine the explained variance ratio.
  - To retain 95% of the cumulative explained variance, the first 347 principal components were selected.
  - Features were transformed into this reduced-dimensional space.

### 5. Classification
- **Model Selection:** A Support Vector Machine (SVM) classifier with RBF kernel was used.
- **Hyperparameter Tuning:** Grid search was performed with the following parameters:
  - Kernel: `rbf`
  - Gamma: `[1e-2, 1e-3, 1e-4]`
  - C: `[1, 10, 100, 1000]`
  - Class Weights: `[balanced, None]`
- **Best Model:** `SVC(C=1, gamma=0.001, kernel='rbf', class_weight='balanced')`.

### 6. Performance Evaluation
- **Training Accuracy:** 99.5%
- **Testing Accuracy:** 96.5%
- **Classification Report:** Precision, recall, and F1-score were calculated for each individual in the dataset, with most scores exceeding 0.90.

---

## Results Visualization

### Distance Visualization
- Pairwise distances between images were visualized for better understanding of feature separability.

### PCA Visualization
- Bar plot of individual explained variance ratios and cumulative variance explained.
- Highlighted the number of components required to retain 95% variance (347 components).

### Sample Predictions
- Individual predictions were displayed for:
  1. The 10th image from the test set.
  2. Randomly selected 20 images from the test set.
- Predictions were color-coded:
  - Green: Correct Predictions
  - Red: Incorrect Predictions

---

## Code Highlights
### Embedding Extraction
```python
embedding = vgg_face_descriptor.predict(np.expand_dims(sample_img, axis=0))[0]
```

### PCA Transformation
```python
pca = PCA(n_components=347, random_state=random_state, svd_solver='full', whiten=True)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)
```

### SVM Classifier
```python
svc_pca = SVC(C=1, gamma=0.001, kernel='rbf', class_weight='balanced', random_state=random_state)
svc_pca.fit(X_train_pca, y_train)
```

### Visualization of Predictions
```python
plt.imshow(sample_img)
plt.title(f"A: {actual_name} \n P: {pred_name}", color='green' if actual_name == pred_name else 'red')
plt.show()
```

---

## Conclusion
- The VGG pre-trained model effectively extracted embeddings, enabling accurate face recognition with SVM.
- PCA reduced feature dimensions, retaining 95% variance while improving computational efficiency.
- The SVM classifier achieved over 96% accuracy, demonstrating its robustness.

---

## Future Work
- Explore other dimensionality reduction techniques.
- Experiment with deep learning classifiers such as CNNs or Transformers.
- Fine-tune the VGG model for better feature extraction.

---

## Requirements
- Python 3.8+
- TensorFlow
- scikit-learn
- OpenCV
- Matplotlib

---

## Usage
1. Clone the repository.
2. Install the required libraries.
3. Run the notebook to reproduce the results.

---

## Acknowledgements
- **VGG Face Model:** Pre-trained weights used for embedding generation.
- **Scikit-learn:** Dimensionality reduction and classification utilities.
- **Matplotlib:** Visualization tools.
