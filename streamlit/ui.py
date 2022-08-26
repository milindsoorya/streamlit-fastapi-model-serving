import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/server_divnoise"



def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("DeepLabV3 image segmentation")

st.write(
    """Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Denoise Image"):

    col1, col2 = st.columns(2)

    if input_image:
        denoised = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        denoised_image = Image.open(io.BytesIO(denoised.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Denoised")
        col2.image(denoised_image, use_column_width=True)

    else:
        # handle case with no image
        st.write("Insert an image!")
