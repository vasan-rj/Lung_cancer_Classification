# &#x20;Lung Cancer Classification using Hybrid CNN + ViT and AI Agent-Based Reporting

> Final Year B.Tech Project | National Institute of Technology Puducherry
> **Achieved 97.33% test accuracy on LIDC-IDRI CT scan dataset**

---

## 📌 Project Overview

This project presents an automated framework for lung cancer classification using a **hybrid CNN + Vision Transformer (ViT)** model, integrated with \*\*AI Agents \*\* for generating detailed radiology reports. The system processes CT scan images to classify four types of lung conditions and outputs a comprehensive, PDF-formatted diagnostic report powered by the **Gemini API**.

---

## 🗂️ Dataset

* **Source**: [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* **Total Images**: 1000 CT scans
* **Classes**:

  * Adenocarcinoma
  * Squamous Cell Carcinoma
  * Large Cell Carcinoma
  * Normal Lung Tissue
* **Split**:

  * Train: 613
  * Validation: 72
  * Test: 315
* **Annotations**: Expert radiologist labels

---

## ⚙️ Methodology

### 🖼️ Image Processing & Feature Extraction

* Grayscale normalization, resizing, and label encoding
* Deep features: **DenseNet121**
* Handcrafted features: **GLCM**, **Wavelet Transform**, **HOG**

### 🧠 Model Architecture

* Hybrid of **Convolutional Neural Networks (CNN)** and **Vision Transformer (ViT)**
* Trained for 50 epochs with standard classification metrics: accuracy, precision, recall, F1-score

### 📌 Lesion Localization

* **Contour-based bounding box** with scale factor 0.8
* Minimum area thresholding for small lesion detection

### 📄 Radiology Report Generation

* Utilizes **Gemini API** (LLM) to generate:

  * Diagnosis
  * Treatment recommendation
  * Lifestyle guidance
* Output in **PDF format** using NLP and image-based predictions

---

## 📊 Results

| Metric                   | Value                                                   |
| ------------------------ | ------------------------------------------------------- |
| **Test Accuracy**        | 97.33%                                                  |
| **F1-Score**             | 0.88 – 0.99 (class-wise)                                |
| **Precision/Recall**     | High across all classes                                 |
| **Radiologist Feedback** | Positive validation for diagnosis and report generation |

---

## 🧰 Tech Stack

* **Languages**: Python
* **Libraries**: TensorFlow, Keras, OpenCV, scikit-learn
* **Models**: DenseNet121, Vision Transformer (ViT)
* **Feature Extractors**: GLCM, HOG, Wavelet
* **LLM API**: Gemini API
* **AI Agent Framework:  Agno**
* **Output**: PDF Reports
* **Tools**: Jupyter, LabelEncoder, Matplotlib, seaborn

---

## 📄 License

This project is intended for educational and academic research purposes. Please cite appropriately when using.

---

## 🙏 Acknowledgments

* **National Institute of Technology Puducherry**
* **Dr. Venkatesan M**, Associate Professor, Dept. of CSE
* **LIDC-IDRI** for the CT scan dataset

---
