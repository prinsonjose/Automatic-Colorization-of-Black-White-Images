# 🎨 Automatic Colorization of Black & White Images

Turn black & white photos into color with a single click. This project uses a pretrained deep learning model (Zhang et al., ECCV 2016) run through OpenCV's DNN module, and ships with both a **Streamlit web app** and a **step-by-step Jupyter notebook**.

---

## Features

- Upload a B&W photo and get a colorized version in seconds
- Side-by-side view of the original and the colorized result
- One-click download of the result as a high-quality JPEG
- Supports JPG, JPEG, PNG, WEBP and BMP
- Model files download automatically on first run, with size checks to catch corrupt or partial downloads
- Friendly error messages for corrupt uploads or failed model loading
- Thread-safe inference, so multiple users can share one running app
- Large images are downscaled automatically to avoid running out of memory
- Educational notebook that walks through every stage of the pipeline

---

## How It Works

The model colorizes images in the **LAB color space**, where **L** is lightness (the black & white part) and **a** / **b** are the two color channels.

1. **Convert to LAB**: the image is converted to grayscale and then to LAB. Only the L channel is kept, so an already-colored image is treated as B&W.
2. **Predict color**: the L channel is resized to 224×224, centered around zero, and fed into the CNN, which predicts the **a** and **b** channels. The network classifies each pixel into one of 313 color bins (loaded from `pts_in_hull.npy`).
3. **Upscale the color**: the predicted a/b channels are resized back to the original image size using cubic interpolation.
4. **Recombine**: the original full-resolution L channel is merged with the predicted a/b channels and converted back to RGB. Because the original lightness is preserved, fine detail is not lost.

> **Reference:** R. Zhang, P. Isola, A. Efros. *Colorful Image Colorization.* ECCV 2016. [Project page](https://richzhang.github.io/colorization/) · [GitHub](https://github.com/richzhang/colorization)

---

## Project Structure

```
Automatic Colorization of Black & White Images/
├── app.py                 # Streamlit web interface
├── colorizer.py           # Model download/loading and colorization logic
├── Automatic Colorization of Black & White Images.ipynb   # Step-by-step notebook (Colab)
├── requirements.txt       # Python dependencies
├── gitignore              # Rename to .gitignore (ignores models/, caches)
└── models/                # Created on first run
    ├── colorization_deploy_v2.prototxt
    ├── colorization_release_v2.caffemodel   (~129 MB)
    └── pts_in_hull.npy
```

---

## Getting Started

### Prerequisites

- Python 3.9 or newer (developed on Python 3.12)
- An internet connection on first run (to download the ~129 MB model)

### Installation

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

1. Upload a black & white image.
2. Click **Start Process**.
3. View the colorized result on the right and click **Download Image** to save it.

The first launch downloads the model files into `models/`. Later launches reuse them.

---

## Using the Notebook

`Automatic Colorization of Black & White Images.ipynb` is designed for **Google Colab** and demonstrates each step of the pipeline with visualizations: LAB channel breakdown, model input, prediction, and before/after comparison.

1. Upload the notebook to [Google Colab](https://colab.research.google.com/).
2. Run the cells in order.
3. Upload an image when prompted; the colorized result is shown and can be downloaded.

---

## Using the Colorizer in Your Own Code

`colorizer.py` can be used independently of the web app:

```python
from colorizer import colorize_image

with open("old_photo.jpg", "rb") as f:
    result_bytes = colorize_image(f.read())

with open("colorized.jpg", "wb") as f:
    f.write(result_bytes)
```

`colorize_image(image_bytes, max_side=1500)` takes the raw bytes of an image file and returns the colorized image as JPEG bytes. It raises `ValueError` if the input is not a readable image.

---

## Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| `MAX_SIDE` | `colorizer.py` | `1500` | Longest image side (in pixels) before downscaling. Increase it if you have more RAM. |
| `NET_INPUT_SIZE` | `colorizer.py` | `224` | Fixed input size of the network. Do not change. |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.35` | Web interface |
| `opencv-python-headless >= 4.8` | Image processing and DNN inference |
| `numpy >= 1.24` | Array operations |

---

## Limitations

- Colors are a **plausible guess**, not historical truth. The model predicts the most likely colors based on what it learned from its training data, so results can be muted or inaccurate for unusual scenes.
- Images with a longest side above `MAX_SIDE` are downscaled before processing.
- The model runs on CPU through OpenCV, so very large images take longer.

---

## Troubleshooting

**Model fails to download:** check your internet connection. You can also download the three files manually and place them in the `models/` folder. The expected sizes are 9,945 bytes (`.prototxt`), 128,946,764 bytes (`.caffemodel`) and 5,088 bytes (`.npy`).

**`models/` not found when launching from another folder:** this is handled automatically, since paths are resolved relative to `colorizer.py`.

**Corrupt image error:** make sure the file is a valid JPG, PNG, WEBP or BMP.

---

## Acknowledgements

- Richard Zhang, Phillip Isola and Alexei A. Efros for the original *Colorful Image Colorization* model
- [OpenCV](https://opencv.org/) for the DNN module
- [Streamlit](https://streamlit.io/) for the web framework

---

