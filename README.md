# GitEill Project

# Organoid Image Preprocessing Tool

An automated, interactive pipeline for preprocessing microscopy images of organoids, specifically designed for fluorescence and brightfield images. 

This tool is built with **Python, OpenCV, and Gradio**, providing an end-to-end workflow: automatic image type detection, adaptive preprocessing (denoising, background correction, illumination correction, deconvolution, and CLAHE), and quantitative quality evaluation.

> **UI Preview**: 

---

## Why make this tool?
Microscopy images often suffer from uneven illumination, background noise, and low contrast. Manual tuning of parameters for each image is time-consuming and subjective. 

This tool solves this by:
1. **Auto-Flagging**: Automatically determines if an image is fluorescence or brightfield, and decides which preprocessing steps are needed.
2. **Parameterized Pipeline**: Allows manual fine-tuning of parameters via an intuitive web UI.
3. **Quantitative Evaluation**: Computes metrics (Clearness, Noise, Contrast, Entropy) before and after processing, and visualizes the intensity histograms.
4. **Batch Processing & Reporting**: Processes multiple images simultaneously and exports a CSV quality report.

---

## How to Use

### 1. Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/YourUsername/Organoid-Preprocessing.git
cd Organoid-Preprocessing
pip install -r requirements.txt
