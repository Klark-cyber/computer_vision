# Computer Vision — Food Menu Detector & CV Fundamentals

> A collection of computer vision exercises in Python, culminating in a trained image classifier that identifies food dishes from photos.

## Overview

This repository documents a hands-on progression through core computer vision tools — data visualization, image scraping, OpenCV fundamentals — leading up to **Menu Detector**, a transfer-learning image classifier built with PyTorch that recognizes five food categories from a photo.

## Notebooks

### `menu_detector_model.ipynb` — the core project
A food image classifier built via transfer learning on MobileNetV2.

- **Classes:** hamburger, hot_dog, dessert (chocolate cake + cheesecake merged), kebab, pilaf
- **Base dataset:** Food101 (for hamburger, hot dog, and dessert classes)
- **Custom data:** kebab and pilaf images collected separately via web scraping (Food101 doesn't include these), since they aren't represented in Food101
- **Model:** pretrained MobileNetV2 (ImageNet weights) with a replaced final classifier layer for 5-class output
- Custom PyTorch `Dataset` class with corrupted-image handling (skips unreadable files instead of crashing the training loop)
- Standard preprocessing pipeline: resize to 224×224, tensor conversion, ImageNet normalization
- Training loop with train/validation split (80/20), Adam optimizer, cross-entropy loss, and best-model checkpointing based on validation accuracy
- Inference notebook: upload an image, get top-4 predicted classes with confidence scores

### `data_scraping.ipynb`
Automated image collection for classes underrepresented in Food101, using `bing-image-downloader` to gather kebab and pilaf images, then organizing them into the Google Drive dataset folder structure expected by the training notebook.

### `opencv.ipynb`
OpenCV fundamentals: reading and displaying images, inspecting image shape/channels (BGR), grayscale conversion, resizing, cropping, rotation, and flipping. Includes a real-time webcam face detection demo using a Haar Cascade classifier, drawing bounding boxes on detected faces frame-by-frame.

### `Matplotlib.ipynb`
Data visualization fundamentals: line plots, styled lines, multi-line comparisons, scatter plots, bar charts, histograms, subplots, saving figures to disk, and 3D plotting (wireframe and scatter) with `mpl_toolkits.mplot3d`.

## Tech Stack

**Core:** Python, PyTorch, torchvision

**Computer Vision:** OpenCV (`cv2`), Haar Cascade classifiers

**Data & Visualization:** NumPy, Matplotlib, PIL (Pillow)

**Data Collection:** bing-image-downloader

**Environment:** Google Colab, Google Drive (dataset storage)

## Key Techniques

- Transfer learning (fine-tuning a pretrained CNN instead of training from scratch)
- Custom `Dataset` / `DataLoader` implementation with error handling for corrupted images
- Addressing class imbalance / missing classes in a public dataset via targeted web scraping
- Real-time classical computer vision (Haar Cascade face detection) alongside a modern deep learning classifier
- Model checkpointing based on validation performance, not just training loss

## Notes

This is a learning-oriented repository — the notebooks build up from visualization and OpenCV basics toward the trained classifier, rather than being a single deployed application. The Menu Detector model itself is a complete, working transfer-learning pipeline: data collection → preprocessing → training → validation → inference.
