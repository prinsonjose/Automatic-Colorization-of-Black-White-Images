"""
app.py
------
Streamlit front end for the B&W image colorizer.
Run with:  streamlit run app.py
"""

import streamlit as st

import colorizer
from colorizer import colorize_image

# FIX: set_page_config was called twice. It must be called once, and it must be
# the first Streamlit command in the script.
st.set_page_config(
    page_title="Automatic Colorization of Black & White Images",  # browser tab title
    page_icon="🎨",  # browser tab icon
    layout="wide",
)

st.title("Automatic Colorization of Black & White Images")


@st.cache_resource(show_spinner="Loading colorization model (first run may download ~129 MB)...")
def load_model():
    """Load the network once and reuse it for every user and rerun."""
    return colorizer.get_network()


# Show a readable error instead of a traceback if the model can't be loaded
# (for example no internet on the very first run).
try:
    load_model()
except Exception as e:
    st.error(f"Could not load the colorization model: {e}")
    st.stop()

# FIX: accept more formats. The colorizer handles anything OpenCV can decode.
bw_image = st.file_uploader(
    "Upload a Black and White Image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

# --------------------------------------------------------------------------
# Session state: remember the result between Streamlit reruns
# --------------------------------------------------------------------------
if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# FIX: when the user uploads a different image (or removes it), clear the old
# result. Before, the previous colorized picture stayed next to the new upload.
current_id = bw_image.file_id if bw_image is not None else None
if current_id != st.session_state.last_file_id:
    st.session_state.last_file_id = current_id
    st.session_state.result_bytes = None

# Two equal columns: original on the left, result on the right.
# (The unused middle "divider" column and its CSS were removed.)
left_col, right_col = st.columns(2, gap="large")

with left_col:
    if bw_image is not None:
        # FIX: a corrupt upload used to crash the page right here, because
        # st.image can't display it. Show a friendly message instead and skip
        # the rest of this column.
        try:
            st.image(bw_image, caption="Original")
            preview_ok = True
        except Exception:
            preview_ok = False
            st.error("This file could not be read as an image. Please upload a valid image.")

        if preview_ok and st.button("Start Process", type="primary"):
            with st.spinner("Colorizing your image..."):
                # FIX: catch errors and show a friendly message, not a traceback.
                try:
                    st.session_state.result_bytes = colorize_image(bw_image.getvalue())
                except Exception as e:
                    st.session_state.result_bytes = None
                    st.error(f"Couldn't colorize this image: {e}")

with right_col:
    if st.session_state.result_bytes is not None:
        st.image(st.session_state.result_bytes, caption="Colorized")
        st.caption(
            f"Images larger than {colorizer.MAX_SIDE}px on the longest side "
            "are downscaled before processing."
        )
        st.download_button(
            "Download Image",
            data=st.session_state.result_bytes,
            file_name="colorized_image.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("Your colorized image will appear here.")
