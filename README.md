# Custom ResNet on CIFAR-10 (90% Accuracy in 24 Epochs)

This repository contains a PyTorch implementation of a custom ResNet architecture designed to achieve **90% accuracy** on the CIFAR-10 dataset in just **24 epochs**. The model is trained using the **One Cycle Learning Rate Policy** and utilizes advanced data augmentation techniques.

## Key Objectives & Constraints

* **Architecture:** A specific custom ResNet (DavidNet style) with 3 main layers and residual connections in Layer 1 and Layer 3.
* **Training Strategy:** One Cycle Policy (Total Epochs = 24, Max at Epoch = 5).
* **Optimization:** ADAM Optimizer & CrossEntropyLoss.
* **Augmentation:** RandomCrop, Horizontal Flip, and CutOut (8x8).
* **Batch Size:** 512.
* **Target Accuracy:** > 90%.

---

## Model Architecture

The network follows a specific custom structure defined as follows:

### 1. PrepLayer
* Conv 3x3 (s1, p1) >> BN >> ReLU [64k]

### 2. Layer 1
* **Conv Block:** Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> ReLU [128k]
* **ResBlock:** (Conv 3x3 >> BN >> ReLU >> Conv 3x3 >> BN >> ReLU)
* **Add:** `X + ResBlock(X)`

### 3. Layer 2
* **Conv Block:** Conv 3x3 >> MaxPool2D >> BN >> ReLU [256k]

### 4. Layer 3
* **Conv Block:** Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> ReLU [512k]
* **ResBlock:** (Conv 3x3 >> BN >> ReLU >> Conv 3x3 >> BN >> ReLU)
* **Add:** `X + ResBlock(X)`

### 5. Output Layer
* MaxPooling with Kernel Size 4
* Fully Connected (FC) Layer
* SoftMax

---

## 🛠️ Data Augmentation

To prevent overfitting and improve generalization, the following transforms are applied:

1.  **Random Crop:** 32x32 (after padding of 4)
2.  **FlipLR:** Random Horizontal Flip
3.  **CutOut:** 8x8 regions

```python
# Augmentation Pipeline
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    Cutout(n_holes=1, length=8), # Custom Cutout
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```

## Dataset

In this current project, we are using CIFAR10 data set. The following transformations are made to the training data:

* Random crop 32x32 after adding a padding layer of 4
* Horizontal Flip
* Coarse Dropout (8x8)
* Normalization.

After the augmentations, these are some of the sample images.

![image](https://github.com/JVaishnavi/ERA_Assgn10/assets/11015405/24304881-7c01-4834-8b47-39f846e024a1)

## Model Summary

<img width="600" height="933" alt="image" src="https://github.com/user-attachments/assets/319867a9-5385-42bb-8f6b-7831f9f73023" />


## One Cycle Policy

Find LR has suggested 5.7E-02 as the learning rate
![image](https://github.com/JVaishnavi/ERA_Assgn10/assets/11015405/31fcd0e8-6a89-44df-9e11-a24864096b42)

## Epochs

<img width="600" height="857" alt="image" src="https://github.com/user-attachments/assets/6a37604f-87b0-494e-b8eb-b31fd279d484" />

We see that the model is able to give >90% accuracy in just 12 epochs

