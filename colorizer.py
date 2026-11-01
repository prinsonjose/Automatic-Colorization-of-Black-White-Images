"""
colorizer.py
------------
Colorizes black & white images using the Zhang et al. (ECCV 2016)
"Colorful Image Colorization" Caffe model, run through OpenCV's DNN module.

How it works (short version):
  1. Convert the image to the LAB color space. "L" is lightness (the B&W part),
     "a" and "b" are the two color channels.
  2. Feed only the L channel to the neural network, which predicts a and b.
  3. Combine the ORIGINAL full-resolution L with the predicted a/b and
     convert back to RGB.
"""

import os
import threading
import urllib.request

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# FIX: build the path from this file's location instead of the current working
# directory. Before, launching the app from another folder made it look for
# "models/" in the wrong place and try to re-download everything.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# path -> (download url, expected size in bytes)
# The expected size lets us detect truncated/corrupt downloads.
MODEL_FILES = {
    PROTOTXT_PATH: (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
        9945,
    ),
    MODEL_PATH: (
        "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
        128946764,
    ),
    POINTS_PATH: (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
        5088,
    ),
}

# Images larger than this (longest side, in pixels) are downscaled before
# processing to avoid running out of memory. Change it if you have more RAM.
MAX_SIDE = 1500

# The network always works on a fixed 224x224 input.
NET_INPUT_SIZE = 224

# --------------------------------------------------------------------------
# Model download / loading
# --------------------------------------------------------------------------
_net = None
# Protects loading the network (so two users don't load it twice)...
_load_lock = threading.Lock()
# ...and protects inference. FIX: Streamlit serves many users from one process
# and they all share the same network object. Without a lock, two simultaneous
# requests could overwrite each other's input blob.
_infer_lock = threading.Lock()


def _download_models():
    """Download any missing (or corrupt) model files."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, (url, expected_size) in MODEL_FILES.items():
        # FIX: check the size too, not just existence. A half-finished download
        # used to be accepted as valid and would break every later run.
        if os.path.exists(path) and os.path.getsize(path) == expected_size:
            continue

        print(f"Downloading {os.path.basename(path)} ...")
        tmp_path = path + ".part"
        try:
            # Download to a temporary file first...
            urllib.request.urlretrieve(url, tmp_path)
            if os.path.getsize(tmp_path) != expected_size:
                raise RuntimeError(
                    f"Downloaded {os.path.basename(path)} has an unexpected size "
                    "(download incomplete or file changed)."
                )
            # ...then move it into place only once it is complete.
            os.replace(tmp_path, path)
        finally:
            # Clean up the temp file if anything went wrong.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def get_network():
    """
    Return the colorization network, loading it on first use.

    FIX: the model used to load when the module was imported, which froze the
    app on startup and crashed it if the download failed. Now it loads lazily,
    so the app can show a spinner or a proper error message.
    """
    global _net
    with _load_lock:
        if _net is None:
            _download_models()
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

            # The 313 cluster centers of the ab color space ("color bins").
            # They are loaded into the network's class8_ab layer.
            pts = np.load(POINTS_PATH).transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]

            # Scaling factor required by the model's final layer.
            net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
                np.full((1, 313), 2.606, dtype="float32")
            ]
            _net = net
    return _net


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def colorize_image(image_bytes: bytes, max_side: int = MAX_SIDE) -> bytes:
    """
    Take the raw bytes of an image file (jpg/png/webp/...) and return the
    colorized image as JPEG bytes.

    Raises ValueError if the bytes are not a readable image.
    """
    net = get_network()

    # Decode the file. cv2.imdecode returns None for corrupt/unsupported data.
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the image. The file may be corrupted.")

    # Downscale very large images (never upscale small ones).
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        h, w = image.shape[:2]

    # Step 1: get the lightness (L) channel.
    # Going through grayscale means an already-colored image is treated as B&W
    # (its original colors are discarded). L is on a 0-100 scale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    L = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]

    # Step 2: prepare the network input: 224x224, centered around zero.
    # INTER_AREA gives cleaner results when shrinking.
    L_small = cv2.resize(L, (NET_INPUT_SIZE, NET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(L_small - 50)

    # Step 3: run the network (one request at a time; see _infer_lock above).
    with _infer_lock:
        net.setInput(blob)
        ab = net.forward("class8_ab")[0].transpose((1, 2, 0))

    # Step 4: stretch the predicted color back to the image size.
    # INTER_CUBIC gives smoother color edges than the default bilinear.
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    # Step 5: merge FULL-resolution L with the predicted ab and go back to RGB.
    lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    rgb = np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)
    out_bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Encode as high-quality JPEG.
    ok, encoded = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to encode the output image.")
    return encoded.tobytes()
