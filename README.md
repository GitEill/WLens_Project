# GitEill Project

# WLens Image Preprocessing Tool (Tempo name)

An automated, interactive pipeline for preprocessing microscopy images of organoids, specifically designed for fluorescence and brightfield images. 

This tool is built with **Python, OpenCV, and Gradio**, providing an end-to-end workflow: automatic image type detection, adaptive preprocessing (denoising, background correction, illumination correction, deconvolution, and CLAHE), and quantitative quality evaluation.

Live Demo: https://project-36g8.onrender.com
(Note: If the app shows a 503 error or takes a moment to load, please wait 30–50 seconds for the free-tier server to wake up.)
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
git clone https://github.com/GitEill/WLens_project.git
cd WLens_project
pip install -r requirements.txt
```
### 2. Run the interactive UI
```bash
python "gd run.py"
```

### 3. Steps in the UI
Upload Images: Open the web UI and upload one or more .tif or .tiff microscopy images.

**Select Processing Mode:**

**Auto Mode:** Automatically detects image type and applies recommended preprocessing steps.

**Save the processed images:** Check this flag to allow all processed image on the list to be saved in "preprocessed images" under the installation pathway after the run. 
Note: This function is not available in live demo

**Manual Mode:** Uncheck the Auto Mode to expand manual steps and adjust parameters (see The Parameters section below).

Note: In the current version, the tool runs in a fixed order: Denoise → Background Correction → Illumination Correction → Deconvolution → CLAHE.

**Run the Pipeline:** Click the Run button to process the images. You can view the original and processed images side by side along with their intensity histograms.

**Review Metrics:** Check the computed quality metrics (Clearness, Noise, Contrast, Entropy) to evaluate preprocessing effectiveness.

**Export Results:** Download the processed images and a CSV report summarizing the quantitative metrics for your dataset.

---

## The Parameters (Manual Mode)

When switching to **Manual Mode**, you can fine-tune the following parameters:

| Processing Step | UI Parameter | Description |
| :--- | :--- | :--- |
| **Denoise** | Kernel (large / small) | Gaussian blur kernel size. Larger kernels remove more noise but may also blur fine cellular details. |
| **Background Correction** | Rolling ball radius | The radius of the ball in the rolling-ball algorithm. A larger radius is better for slowly varying uneven backgrounds. |
| **Background Correction** | Polyfit downsample | The downsampling factor used before fitting a 2D polynomial surface. Lower values speed up fitting but might miss fine background variations. |
| **Illumination Correction** | CLAHE clip limit (flu / bri) | Threshold for contrast limiting. Higher values increase local contrast but may amplify background noise in homogeneous regions. |
| **Deconvolution** | RL iterations | Number of Richardson-Lucy iterations. More iterations increase sharpness but also amplify noise if not regularized. |
| **Deconvolution** | PSF sigma | Standard deviation of the Gaussian Point Spread Function (PSF). It represents the expected blur level of your microscope optics. |
| **CLAHE (Post-process)** | Tile grid | Size of the grid for local histogram equalization. Smaller grids enhance very local details but may cause blocky artifacts. |

---

## Technical Highlights
### 1. Image Type Determination
The tool extracts corner intensity, global standard deviation, and low-intensity pixel ratios to automatically classify the image as flu (fluorescence) or bri (brightfield). This prevents applying inappropriate processing (e.g., rolling ball background correction on brightfield images).

### 2. Advanced Preprocessing
**Illumination Correction:** CLAHE applied to the L-channel in the LAB color space to improve local contrast.

**Background Correction:** Rolling ball algorithm (for fluorescence) or 2nd-degree polynomial fitting (for brightfield).

**Deconvolution:** Richardson-Lucy (RL) algorithm with Tikhonov regularization to enhance sharpness without over-amplifying noise.

### 3. Thread-Safe Parameter System
Implemented a dynamic parameter dictionary (config.make_params) that allows the Gradio UI to safely pass local parameters into the preprocessing pipeline without global variable conflicts, ensuring safe concurrent processing.

---

## Evaluation Metrics & Histogram
### Evaluation Metrics
The tool applies quantitative measurements to assess how well an image process performs. The pipeline calculates:

**Clearness Ratio:** Variance of Laplacian (higher means sharper).

**Noise Ratio:** Standard deviation of high-frequency components (lower means smoother).

**Contrast Ratio:** Global standard deviation (higher means better contrast).

**Entropy Ratio:** Shannon entropy (higher means more information content).


### How to Read the Histogram


The figure compares the intensity distributions before (blue) and after (red) preprocessing.
Pixel intensity shown in x-axis and ranged from 0-255. y-axis demonstrates the pixel distribution across an image. 

For an instance, a flu image on histogram: 
Original Image (Blue): Typically shows a sharp peak near 0 (dark background) and a long, sparse tail (cell signals and noise).
Processed Image (Red):
If CLAHE or illumination correction is applied, the background peak broadens to the right (enhancing local contrast).
If Denoising is applied, the high-intensity tail is suppressed (smoothing out shot noise).

Interpretation: Aggressive contrast enhancement (e.g., high CLAHE clip limits) will visually improve the image but might decrease the absolute Laplacian sharpness metric, because smoothing out edges reduces high-frequency gradients. The histogram visually explains the trade-off between local contrast and absolute sharpness.

---

## Future work
-Increase versatility of current imaging preprocessing steps

-Add on Deep-learning-based semantic segmentation for microscopy images

-Integration with image analysis tools

-AI deployment 
