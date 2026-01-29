# Lecture 2: Image Classification & Linear Classifiers

## Course Overview

This is Lecture 2 of the CS231N Computer Vision course, covering the following topics:

- Definition and challenges of image classification tasks
- Data-Driven Approach
- K-Nearest Neighbor (KNN) algorithm
- Hyperparameter tuning strategies
- Linear Classifier
- Softmax classifier and loss functions

**Learning Objectives**:

1. Understand the core challenges of image classification
2. Master the three-step workflow of the data-driven approach
3. Understand how K-Nearest Neighbor algorithm works and its limitations
4. Learn best practices for hyperparameter tuning
5. Master algebraic, visual, and geometric interpretations of linear classifiers
6. Understand the Softmax classifier and cross-entropy loss function

---

## Table of Contents

1. [Image Classification Task Definition](#1-image-classification-task-definition)
2. [Core Challenges of Image Classification](#2-core-challenges-of-image-classification)
3. [Data-Driven Approach](#3-data-driven-approach)
4. [K-Nearest Neighbor Algorithm (KNN)](#4-k-nearest-neighbor-algorithm-knn)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Linear Classifier](#6-linear-classifier)
7. [Softmax Classifier and Loss Function](#7-softmax-classifier-and-loss-function)
8. [Key Takeaways Summary](#key-takeaways-summary)
9. [References](#references)

---

![00:00:48](../../assets/lecture-2/screenshot-01KG3M86JCFEXWV48NGY9WQFFT.png)
[00:00:48](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 1. Image Classification Task Definition

### 1.1 What is Image Classification?

**Image Classification** is one of the core tasks in computer vision, defined as:

- **Input**: An image
- **Output**: A label selected from a predefined label set
- **Goal**: Assign the correct category label to the input image



![00:01:45](../../assets/lecture-2/screenshot-01KG3MA84E84D7GCZ6A4SWNWX6.png)
[00:01:45](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Example**: Given an image and label set {cat, dog, truck, plane, ...}, the system needs to output "cat"

### 1.2 Semantic Gap

For humans, image classification is an extremely simple task because our cognitive system can understand images holistically. But for computers, this is a huge challenge:

- **Human perspective**: Sees a cat
- **Computer perspective**: Sees a numerical matrix (tensor)

**Digital representation of images**:

```
Image size: 800 × 600 pixels
RGB three channels: Red, Green, Blue
Data structure: 800 × 600 × 3 three-dimensional tensor
Pixel value range: 0-255 (8-bit data)
```

The RGB value of each pixel is an integer between 0-255, which is the standard 24-bit (3×8-bit) RGB format.

---

![00:03:50](../../assets/lecture-2/screenshot-01KG3MF8A7X7YH7DK885FRTVHN.png)
[00:03:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 2. Core Challenges of Image Classification

### 2.1 Viewpoint Variation

**Problem**: Even if the object remains stationary, simply moving the camera (translation, rotation) will cause all pixel values to change.

- For humans: Same object
- For computers: Completely different data points (all 800×600×3 = 1,440,000 numerical values change)

![00:05:50](../../assets/lecture-2/screenshot-01KG3MJWJFGMQMTR10DP6FK2YK.png)
[00:05:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 2.2 Illumination

**Problem**: RGB pixel values are functions of lighting conditions.

$$\text{Pixel Value} = f(\text{Object Reflectance}, \text{Light Source}, \text{Camera Response})$$

The same object under different lighting will produce different pixel values:

- Cat in bright sunlight
- Cat in a dark room
- For humans: Still the same cat
- For computers: Completely different numerical values

![00:06:09](../../assets/lecture-2/screenshot-01KG3MKFSD3R7DDQSN7MHAETD9.png)
[00:06:09](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 2.3 Other Major Challenges

| Challenge Type | Description | Impact |
| ------------ | --------------------- | ---------------------------------- |
| **Background Clutter** | Background Clutter | Target object may be confused with background |
| **Scale Variation** | Scale Variation | Objects appear at different sizes in images |
| **Occlusion** | Occlusion | Parts of objects are occluded |
| **Deformation** | Deformation | Objects change shape (e.g., different poses of cats) |
| **Intra-class Variation** | Intra-class Variation | Large appearance differences among same-class objects (different breeds of cats) |

![00:07:34](../../assets/lecture-2/screenshot-01KG3MP2BRPZ3DZTB2C32P2Q17.png)
[00:07:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



![00:09:03](../../assets/lecture-2/screenshot-01KG3MTYZAYT461M9HPTH518PT.png)
[00:09:03](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:09:35](../../assets/lecture-2/screenshot-01KG3MVXWXRQMFST1BWPRA1WET.png)
[00:09:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:10:06](../../assets/lecture-2/screenshot-01KG3MWW6REFBMRDS9P8VW0QWC.png)
[00:10:06](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 2.4 Occlusion Problem Example

Even though only the cat's tail and a small portion of fur can be seen, humans can infer that this is a cat based on **context** (living room, sofa) rather than a tiger or raccoon.

![00:08:16](../../assets/lecture-2/screenshot-01KG3MRQBZH4H9SFPPPX4GXM3B.png)
[00:08:16](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 3. Data-Driven Approach

### 3.1 Limitations of Traditional Methods

**Rule-based approach** (such as edge detection):

```python
# Pseudocode example
def recognize_cat(image):
    edges = detect_edges(image)
    corners = find_corners(edges)
    features = extract_features(corners)
    if matches_cat_pattern(features):
        return "cat"
```

**Problems**:

- Difficult to scale (each category requires hand-crafted rules)
- Poor generalization ability
- Sensitive to variations



![00:10:56](../../assets/lecture-2/screenshot-01KG3MZ1QXWXS805MCZ84AVRKJ.png)
[00:10:56](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:11:53](../../assets/lecture-2/screenshot-01KG3N0S90P5JMPM2E885JR1P2.png)
[00:11:53](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:12:00](../../assets/lecture-2/screenshot-01KG3N10R53Y32KMX1G5ZZCVF3.png)
[00:12:00](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:13:24](../../assets/lecture-2/screenshot-01KG3N3JWN128FDXE6G6V2D0CR.png)
[00:13:24](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

---

### 3.2 Three-Step Workflow of Data-Driven Approach

Modern machine learning adopts a **Data-Driven Approach**, which includes three core steps:

**Step 1: Collect Dataset**

- Collect a large number of images and their corresponding labels
- Example datasets: ImageNet, CIFAR-10, CIFAR-100

**Step 2: Train Classifier**

```python
def train(images, labels):
    """
    Use machine learning algorithms to train the model
    Input: Training images and labels
    Output: Trained model
    """
    model = build_model()
    model.fit(images, labels)
    return model
```

**Step 3: Evaluate Classifier**

```python
def predict(model, test_images):
    """
    Test the model on new images
    Input: Model and test images
    Output: Predicted labels
    """
    predictions = model.predict(test_images)
    return predictions
```

---

![00:16:15](../../assets/lecture-2/screenshot-01KG3N8SG00RFTPTEB7YP9HHHP.png)
[00:16:15](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 4. K-Nearest Neighbor Algorithm (KNN)

![00:16:58](../../assets/lecture-2/screenshot-01KG3NA38VKPBRTTVFSD9VEQA8.png)
[00:16:58](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 4.1 Algorithm Principle

K-Nearest Neighbor is one of the simplest data-driven methods, and its core idea is: **similar inputs should produce similar outputs**.

![00:17:50](../../assets/lecture-2/screenshot-01KG3NBNSD2RYTDVFYQGD9S6XS.png)
[00:17:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 4.2 Algorithm Implementation

**Training Phase**:

```python
def train(images, labels):
    """
    Training function: Simply memorize all data
    Time complexity: O(1)
    """
    # Store all training data in memory
    self.train_images = images
    self.train_labels = labels
```

**Prediction Phase**:

```python
def predict(test_image):
    """
    Prediction function: Find the most similar training image
    Time complexity: O(N) - N is the number of training samples
    """
    # Compute distances between test image and all training images
    distances = compute_distances(test_image, self.train_images)
    
    # Find K nearest neighbors
    k_nearest = find_k_smallest(distances, k)
    
    # Vote to decide label
    predicted_label = majority_vote(k_nearest)
    return predicted_label
```

![00:18:19](../../assets/lecture-2/screenshot-01KG3NCJZV4AQSZZJJ33PAPJ34.png)
[00:18:19](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



### 4.3 Distance Metrics

#### L1 Distance (Manhattan Distance)

$$d_{L1}(I_1, I_2) = \sum_{p} |I_1^p - I_2^p|$$

where $p$ iterates over all pixel positions.

**Implementation Example**:

```python
import numpy as np

def L1_distance(image1, image2):
    """
    Compute the L1 distance between two images
    """
    return np.sum(np.abs(image1 - image2))
```



![00:19:10](../../assets/lecture-2/screenshot-01KG3NE4DJWYMC4MKF3N4Z9PRY.png)
[00:19:10](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:20:16](../../assets/lecture-2/screenshot-01KG3NG5AAZFZG5YB281XBAPR8.png)
[00:20:16](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:20:36](../../assets/lecture-2/screenshot-01KG3NGRC5BF8C7TTMZ72ZA19D.png)
[00:20:36](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:21:31](../../assets/lecture-2/screenshot-01KG3NJDPTK707SHSSFVQ73NRM.png)
[00:21:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

---

### 4.5 Advantages and Disadvantages of KNN

**Advantages**:
✓ Simple to implement
✓ Easy to understand
✓ Training time is O(1)

**Disadvantages**:
✗ Prediction time is O(N), inefficient
✗ Requires storing all training data
✗ Pixel-based distance has weak semantic meaning
✗ Poor performance on high-dimensional data (curse of dimensionality)

**Computational Complexity Analysis**:

| Phase | Time Complexity | Description |
| ---- | ---------- | ---------------------- |
| Training | O(1) | Only store data |
| Prediction | O(N) | Need to compare with all training samples |

This is opposite to the ideal situation: we want slow training and fast prediction, because training can be done offline.

![00:24:56](../../assets/lecture-2/screenshot-01KG3NRQ2MJJ51G5405NYYFQ4C.png)
[00:24:56](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:25:56](../../assets/lecture-2/screenshot-01KG3NTHKMRCZJB8B1Y23N2CDB.png)
[00:25:56](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:26:10](../../assets/lecture-2/screenshot-01KG3NTZ5M8ATEDH7EB1G7MAXJ.png)
[00:26:10](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:28:03](../../assets/lecture-2/screenshot-01KG3P1GY5KF6J3TR59C0JMBJ7.png)
[00:28:03](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Characteristics**:

- Sensitive to coordinate axis rotation
- Suitable when features have clear physical meaning

#### L2 Distance (Euclidean Distance)

$$d_{L2}(I_1, I_2) = \sqrt{\sum_{p} (I_1^p - I_2^p)^2}$$

```python
def L2_distance(image1, image2):
    """
    Compute the L2 distance between two images
    """
    return np.sqrt(np.sum((image1 - image2) ** 2))
```

**Characteristics**:

- Insensitive to coordinate axis rotation
- More commonly used in general cases

#### L1 vs L2 Visualization

In 2D space:

- **L1 Distance**: Iso-distance lines form a diamond (square rotated 45°)
- **L2 Distance**: Iso-distance lines form a circle

```
L1: Points at distance r from origin form a square
    |x| + |y| = r

L2: Points at distance r from origin form a circle
    √(x² + y²) = r
```

![00:29:26](../../assets/lecture-2/screenshot-01KG3P41MN2FSHP3T0PJXAMTWJ.png)
[00:29:26](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:34:29](../../assets/lecture-2/screenshot-01KG3PD9X18EQQG5NV7R1Z086W.png)
[00:34:29](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:35:17](../../assets/lecture-2/screenshot-01KG3PERH035B279EYHJKN6XNT.png)
[00:35:17](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

---

### 4.4 Choosing K

**K=1 (Nearest Neighbor)**:

- Advantages: Simple and direct
- Disadvantages: Sensitive to noise, decision boundary is not smooth

**K>1 (K-Nearest Neighbors)**:

- Advantages: More robust to noise
- Disadvantages: May have voting ties

**Decision Boundary Visualization**:

- K=1: Jagged decision boundary, prone to overfitting
- K=3,5,7: Smoother decision boundaries
- K too large: May lead to underfitting

## 5. Hyperparameter Tuning

### 5.1 What are Hyperparameters?

**Hyperparameters** are parameters that need to be set before training begins, such as:

- K value in K-Nearest Neighbor
- Distance metric function (L1, L2, etc.)
- Learning rate (covered in later lectures)
- Regularization strength (covered in later lectures)

![00:36:10](../../assets/lecture-2/screenshot-01KG3PGCVD2GE6AAEX2X8REGGR.png)
[00:36:10](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.2 Methods for Setting Hyperparameters

#### Method 1: Choose on Training Set (❌ Not Recommended)

```python
# Wrong example
best_k = None
best_accuracy = 0
for k in [1, 3, 5, 7, 9]:
    accuracy = evaluate_on_train(k)  # Evaluate on training set
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy
```

**Problem**:

- K=1 will always achieve 100% accuracy on training set (memorizing training data)
- Cannot evaluate generalization ability

#### Method 2: Choose on Test Set (❌ Not Recommended)

```python
# Wrong example
best_k = None
best_accuracy = 0
for k in [1, 3, 5, 7, 9]:
    accuracy = evaluate_on_test(k)  # Evaluate on test set
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy
```

**Problem**:

- Equivalent to "cheating"
- Cannot evaluate model performance on truly unseen data
- Overfitting to test set

#### Method 3: Use Validation Set (✓ Recommended)

```python
# Correct example
# Step 1: Split data
train_set, val_set, test_set = split_data(data)

# Step 2: Choose hyperparameters on validation set
best_k = None
best_val_accuracy = 0
for k in [1, 3, 5, 7, 9]:
    model = train(train_set, k)
    val_accuracy = evaluate(model, val_set)
    if val_accuracy > best_val_accuracy:
        best_k = k
        best_val_accuracy = val_accuracy

# Step 3: Evaluate on test set with best hyperparameters
final_model = train(train_set, best_k)
test_accuracy = evaluate(final_model, test_set)
```

**Data Split Example**:

- Training set: 50,000 samples (for training model)
- Validation set: 10,000 samples (for choosing hyperparameters)
- Test set: 10,000 samples (for final evaluation)

#### Method 4: Cross-Validation (✓✓ Most Recommended, but Computationally Expensive)

**K-Fold Cross-Validation**:

```python
def cross_validation(data, k_folds=5):
    """
    K-fold cross-validation
    """
    fold_size = len(data) // k_folds
    accuracies = []
    
    for fold in range(k_folds):
        # Split data
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        val_set = data[val_start:val_end]
        train_set = data[:val_start] + data[val_end:]
        
        # Train and evaluate
        model = train(train_set)
        accuracy = evaluate(model, val_set)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)
```

**5-Fold Cross-Validation Diagram**:

```
Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]
```

**Advantages**:

- More reliable hyperparameter selection
- Full utilization of data
- Reduces impact of randomness

**Disadvantages**:

- High computational cost (need to train K times)
- Less commonly used in deep learning (large datasets, time-consuming training)

![00:38:41](../../assets/lecture-2/screenshot-01KG3PMZXAASBP7NBBNKHKWZVW.png)
[00:38:41](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:38:46](../../assets/lecture-2/screenshot-01KG3PPDBP5A7GM0CH05E4DGCG.png)
[00:38:46](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.3 Best Practices

**Small Datasets**:

- Use cross-validation

**Large Datasets**:

- Use single validation set
- Rely on experience and intuition

**Never**:

- Tune hyperparameters on test set
- Evaluate generalization performance on training set



![00:39:52](../../assets/lecture-2/screenshot-01KG3PR9A5MV1JPKW0FPAC4GH4.png)
[00:39:52](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.4 CIFAR-10 Experimental Results

**Dataset**:

- 10 categories
- 50,000 training images
- 10,000 test images
- Image size: 32×32×3

**Experimental Setup**:

- Use 5-fold cross-validation
- Test different K values (K=1,3,5,7,9,11...)

**Results**:

- Best K value: K=7
- Best accuracy: ~29%
- Random guessing baseline: 10% (10 categories)

![00:41:40](../../assets/lecture-2/screenshot-01KG3PTTBZ7KK5HP514GWCN3X0.png)
[00:41:40](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:42:15](../../assets/lecture-2/screenshot-01KG3PVN351A0WWH68DGXT7Y8W.png)
[00:42:15](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:43:04](../../assets/lecture-2/screenshot-01KG3PWSC4EEVT6084P64ZQ36N.png)
[00:43:04](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:43:35](../../assets/lecture-2/screenshot-01KG3PXGNDYXMJB2KHMJEFQVE5.png)
[00:43:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:44:12](../../assets/lecture-2/screenshot-01KG3PYCTJV4QEN06Z639451RM.png)
[00:44:12](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:46:02](../../assets/lecture-2/screenshot-01KG3Q0YW27F58GV3BV7S5KPWP.png)
[00:46:02](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Conclusion**: KNN performs moderately on CIFAR-10 because pixel distance does not capture semantic similarity well.

---



## 6. Linear Classifier

![00:46:33](../../assets/lecture-2/screenshot-01KG3Q1P7EMD9CMB7GARMPVNW6.png)
[00:46:33](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



### 6.1 Parametric Methods

Unlike KNN, linear classifiers are a **parametric method**:

**KNN (Non-parametric)**:

- Training: Store all data
- Prediction: Iterate through all data

**Linear Classifier (Parametric)**:

- Training: Learn parameters W and b
- Prediction: Fast computation of $f(x, W, b)$

### 6.2 Mathematical Representation

A linear classifier is defined as:

$$f(x, W, b) = Wx + b$$

**Symbol Explanation**:

- $x$: Input image (flattened into vector)
- $W$: Weight matrix
- $b$: Bias vector
- $f(x, W, b)$: Output scores (scores for each category)

**Dimension Analysis** (using CIFAR-10 as example):

```
Input image: 32 × 32 × 3 = 3,072 dimensions
Output categories: 10 classes

x: [3,072 × 1] column vector
W: [10 × 3,072] weight matrix
b: [10 × 1] bias vector

Output = Wx + b: [10 × 1] score vector
```

**Computation Example**:

```python
import numpy as np

# Input image (flattened)
x = np.random.randn(3072, 1)  # 32x32x3 = 3072

# Weights and biases
W = np.random.randn(10, 3072)  # 10 categories
b = np.random.randn(10, 1)

# Compute scores
scores = W.dot(x) + b  # [10, 1]
```

![00:48:14](../../assets/lecture-2/screenshot-01KG3Q42AJZXPF7QJ5F0V70C37.png)
[00:48:14](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:49:26](../../assets/lecture-2/screenshot-01KG3Q5R8V0GVZ6YYENSXNKCE2.png)
[00:49:26](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:50:09](../../assets/lecture-2/screenshot-01KG3Q6RSWJVW9DY3A1W2Y3YK5.png)
[00:50:09](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:50:25](../../assets/lecture-2/screenshot-01KG3Q74M18BA6518QVQV0Q3HB.png)
[00:50:25](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 6.3 Role of Bias Term

**Without bias**: $f(x, W) = Wx$

- All classification hyperplanes must pass through the origin

**With bias**: $f(x, W, b) = Wx + b$

- Allows hyperplane translation
- Provides greater flexibility
- Helpful for class imbalance

**Example**: If samples of a certain category tend toward positive values, the bias can adjust that category's baseline score.

### 6.4 Three Interpretations of Linear Classifiers

#### 6.4.1 Algebraic Interpretation

A linear classifier is a simple function:

$$\begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_{10} \end{bmatrix} = \begin{bmatrix} w_1^T \\ w_2^T \\ \vdots \\ w_{10}^T \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{3072} \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_{10} \end{bmatrix}$$

The score for each category $c$ is:
$$s_c = w_c^T x + b_c = \sum_{i=1}^{3072} w_{c,i} x_i + b_c$$

![00:50:59](../../assets/lecture-2/screenshot-01KG3Q7YDWABWYMSANEV1J6PTD.png)
[00:50:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

#### 6.4.2 Visual Interpretation

**Each row of weight matrix W is a template for a category**:

```
Row 1 of W = cat template (3,072 numbers)
Row 2 of W = dog template (3,072 numbers)
...
Row 10 of W = truck template (3,072 numbers)
```

**Prediction Process**:

1. Compute inner product between input image and each template
2. Inner product value represents similarity
3. Select the category with highest score

**Visualization of CIFAR-10 Learned Templates**:

- Car template: May show car outline
- Airplane template: May show airplane shape
- Each category learns only one template (this is a limitation of linear classifiers)

![00:52:14](../../assets/lecture-2/screenshot-01KG3Q9Q1A52GBJ5C2WR8JVESP.png)
[00:52:14](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

#### 6.4.3 Geometric Interpretation

In high-dimensional space, linear classifiers define **classification hyperplanes**:

**2D Case** (2 features):

- Classification boundary is a line
- Equation: $w_1x_1 + w_2x_2 + b = 0$

**3D Case** (3 features):

- Classification boundary is a plane

**High-dimensional Case** (3,072 features):

- Classification boundary is a hyperplane

**Multi-class Case**:

- K categories require K hyperplanes
- Space is divided into K regions

**Geometric Meaning of Bias**:

- Controls distance of hyperplane from origin
- Allows hyperplane not to pass through origin

![00:52:52](../../assets/lecture-2/screenshot-01KG3QAJXPA04SMYP6Z0REMWGX.png)
[00:52:52](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 6.5 Limitations of Linear Classifiers

Linear classifiers cannot solve the following problems:

**Problem 1: XOR Problem**

```
Class 1: Quadrants 1 and 3
Class 2: Quadrants 2 and 4
```

Cannot be separated by a single line.

**Problem 2: Concentric Circle Problem**

```
Class 1: Points at distance [1, 2] from origin
Class 2: Other points
```

Requires nonlinear boundaries.

**Problem 3: Multi-modal Problem**

```
Class 1: Three separate regions
Class 2: Space between the three regions
```

A single linear classifier can only learn one template, unable to capture intra-class diversity.

**Solutions** (covered in later lectures):

- Use nonlinear activation functions
- Stack multiple linear layers to form neural networks
- Use feature transformations

---

![00:54:49](../../assets/lecture-2/screenshot-01KG3QJ1ZM8TD7NVMH08HH35BD.png)
[00:54:49](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:55:51](../../assets/lecture-2/screenshot-01KG3QKHAMP15FZWB6PNYTR65A.png)
[00:55:51](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:56:34](../../assets/lecture-2/screenshot-01KG3QMHK5T6VHB42EQVPYWBV3.png)
[00:56:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:56:59](../../assets/lecture-2/screenshot-01KG3QN3PBV88TW2R8V22ZM09E.png)
[00:56:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 7. Softmax Classifier and Loss Function

![00:57:31](../../assets/lecture-2/screenshot-01KG3QNVNK0B9RRTZ1JXMSHHQN.png)
[00:57:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 7.1 Need for Loss Function

Given the linear function $f(x, W) = Wx + b$ can compute scores, now we need:

**Question 1**: How to evaluate the quality of current parameters W?
**Answer**: Define Loss Function

**Question 2**: How to find optimal W?
**Answer**: Optimization algorithms (next lecture content)

![00:59:49](../../assets/lecture-2/screenshot-01KG3QS3CF1ZMH1CPHJMGBR53R.png)
[00:59:49](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 7.2 Loss Function Definition

**Mathematical Representation**:

$$L = \frac{1}{N} \sum_{i=1}^{N} L_i(f(x_i, W), y_i)$$

where:

- $N$: Number of training samples
- $L_i$: Loss of the i-th sample
- $f(x_i, W)$: Model's predicted scores for the i-th sample
- $y_i$: True label of the i-th sample

**Intuitive Understanding**:

- Loss function quantifies the model's "unhappiness level"
- Smaller loss, better model
- Goal: Minimize loss

![01:02:33](../../assets/lecture-2/screenshot-01KG3QWYQT9CNHKJ6ZM558HCMF.png)
[01:02:33](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 7.3 Softmax Classifier

#### 7.3.1 From Scores to Probabilities

The linear function outputs **unnormalized scores (logits)**:

$$s = f(x, W) = [s_1, s_2, \ldots, s_K]$$

**Problem**: Scores have no probabilistic meaning (can be negative, can be very large)

**Solution**: Use Softmax function to convert to probability distribution:

$$P(Y=k|X=x_i) = \frac{e^{s_k}}{\sum_{j=1}^{K} e^{s_j}}$$

**Properties of Softmax**:

1. **Non-negativity**: $e^{s_k} > 0$
2. **Normalization**: $\sum_{k=1}^{K} P(Y=k|X=x_i) = 1$
3. **Monotonicity**: Higher scores lead to higher probabilities
4. **Differentiability**: Facilitates gradient descent optimization

#### 7.3.2 Numerical Example

Suppose scores for a sample are:

```
s = [3.2, 5.1, -1.7]  # cat, car, frog
True label: cat (index 0)
```

**Step 1: Exponentiation**

```
exp(s) = [e^3.2, e^5.1, e^-1.7]
       = [24.5, 164.0, 0.18]
```

**Step 2: Normalization**

```
sum = 24.5 + 164.0 + 0.18 = 188.68

P(cat) = 24.5 / 188.68 = 0.13
P(car) = 164.0 / 188.68 = 0.87
P(frog) = 0.18 / 188.68 = 0.001
```

**Interpretation**: The model believes this image is a cat with 13% probability (incorrect prediction!)

![01:03:28](../../assets/lecture-2/screenshot-01KG3RQDRPWZPGPRZXNNJB06XH.png)
[01:03:28](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 7.4 Cross-Entropy Loss

#### 7.4.1 Definition

For a single sample, the Softmax loss is defined as:

$$L_i = -\log P(Y=y_i|X=x_i) = -\log \left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)$$

**Simplified**:
$$L_i = -s_{y_i} + \log \sum_j e^{s_j}$$

#### 7.4.2 Intuitive Understanding

**Goal**: Maximize the probability of the correct category
$$\max P(Y=y_i|X=x_i)$$

**Equivalent to**: Minimize negative log probability
$$\min -\log P(Y=y_i|X=x_i)$$

**Why use logarithm?**

1. Convert multiplication to addition (numerical stability)
2. Map probability (0-1) to loss (0-∞)
3. Penalizes incorrect predictions more severely

#### 7.4.3 Loss Range

**Theoretical Range**:

- Minimum value: 0 (when $P(Y=y_i|X=x_i) = 1$)
- Maximum value: $\infty$ (when $P(Y=y_i|X=x_i) \to 0$)

**Initialization Check**:
With random initialization, all category probabilities should be approximately equal:

$$P(Y=k|X=x_i) \approx \frac{1}{C}$$

Therefore, initial loss should be approximately:

$$L_i \approx -\log\left(\frac{1}{C}\right) = \log(C)$$

**CIFAR-10 Example** (C=10):
$$L_i \approx \log(10) = 2.3$$

**Debugging Tip**: If initial loss is not $\log(C)$, there's a bug in the implementation!

### 7.5 Information-Theoretic Interpretation of Cross-Entropy

#### 7.5.1 Relationship with KL Divergence

**True distribution** $p$ (one-hot encoding):
$$p = [0, 0, \ldots, 1, \ldots, 0]$$
(Only the correct category is 1, others are 0)

**Predicted distribution** $q$ (Softmax output):
$$q = [P(Y=1), P(Y=2), \ldots, P(Y=K)]$$

**KL Divergence**:
$$D_{KL}(p \| q) = \sum_k p_k \log \frac{p_k}{q_k}$$

When $p$ is one-hot:
$$D_{KL}(p \| q) = -\log q_{y_i}$$

This is exactly the Softmax loss!

#### 7.5.2 Relationship with Entropy

**Cross-Entropy**:
$$H(p, q) = -\sum_k p_k \log q_k$$

**Relationship**:
$$H(p, q) = H(p) + D_{KL}(p \| q)$$

For one-hot encoded $p$:

- $H(p) = 0$ (completely certain)
- Therefore $H(p, q) = D_{KL}(p \| q)$

**Origin of Name**: This is why Softmax loss is also called **Cross-Entropy Loss**.

### 7.6 Implementation Details

#### Numerical Stability

Computing $e^{s_k}$ directly may cause numerical overflow.

**Improvement**:
$$\frac{e^{s_k}}{\sum_j e^{s_j}} = \frac{Ce^{s_k}}{C\sum_j e^{s_j}} = \frac{e^{s_k + \log C}}{\sum_j e^{s_j + \log C}}$$

Choose $C = e^{-\max_j s_j}$:

```python
def softmax_stable(scores):
    """
    Numerically stable Softmax implementation
    """
    # Subtract maximum to prevent overflow
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores)
    return probs

def cross_entropy_loss(scores, y):
    """
    Cross-entropy loss
    """
    probs = softmax_stable(scores)
    loss = -np.log(probs[y])
    return loss
```

---

![01:06:26](../../assets/lecture-2/screenshot-01KG3RVKNCSK563RCAE658MHMK.png)
[01:06:26](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## Key Takeaways Summary

### Key Concepts

1. **Challenges of Image Classification**
   - Semantic gap: Human perception vs computer representation
   - Viewpoint, illumination, occlusion, deformation, and other variations

2. **Data-Driven Approach**
   - Collect data → Train model → Evaluate predictions
   - Replaces hand-crafted rules

3. **K-Nearest Neighbor Algorithm**
   - Simple but inefficient (training O(1), prediction O(N))
   - Distance metrics: L1 (Manhattan) vs L2 (Euclidean)
   - Limited effectiveness in practical applications (pixel-based distance)

4. **Hyperparameter Tuning**
   - Use validation set, not test set
   - Cross-validation provides more reliable results
   - In deep learning, commonly use single validation set

5. **Linear Classifier**
   - Parametric method: $f(x, W) = Wx + b$
   - Three interpretations: Algebraic (weighted sum), Visual (template matching), Geometric (hyperplane)
   - Limitations: Cannot solve linearly inseparable problems

6. **Softmax Classifier**
   - Convert scores to probabilities: Softmax function
   - Loss function: Cross-entropy (negative log likelihood)
   - Initialization check: Loss should be approximately $\log(C)$

### Important Formulas

| Concept | Formula |
| ---------- | ---------------------------------------------------- |
| L1 Distance | $d_{L1}(I_1, I_2) = \sum_p \|I_1^p - I_2^p\|$ |
| L2 Distance | $d_{L2}(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$ |
| Linear Classifier | $f(x, W, b) = Wx + b$ |
| Softmax | $P(Y=k\|X) = \frac{e^{s_k}}{\sum_j e^{s_j}}$ |
| Cross-Entropy Loss | $L = -\log P(Y=y_i\|X=x_i)$ |
| Total Loss | $L = \frac{1}{N} \sum_{i=1}^N L_i$ |

### Next Lecture Preview

- **Optimization Algorithms**: How to find optimal W?
- **Gradient Descent**: Iteratively updating parameters
- **Backpropagation**: Efficiently computing gradients
- **Neural Networks**: Stacking multiple linear layers

---

## References

### Datasets

- **CIFAR-10**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **ImageNet**: [https://www.image-net.org/](https://www.image-net.org/)
