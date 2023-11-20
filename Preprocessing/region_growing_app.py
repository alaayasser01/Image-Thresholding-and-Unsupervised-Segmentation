from region_growing import region_growing
import cv2
import numpy as np
import streamlit as st

st.title("Region Growing Segmentation")
st.sidebar.title("Parameters")

# Upload image file
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Seed point selection
    seed_point_x = st.sidebar.slider("Seed point X coordinate", 0, img.shape[1], img.shape[1] // 2)
    seed_point_y = st.sidebar.slider("Seed point Y coordinate", 0, img.shape[0], img.shape[0] // 2)
    seed_point = (seed_point_x, seed_point_y)

    # Threshold value selection
    threshold = st.sidebar.slider("Threshold", 1, 15, 10)

    # Region growing segmentation
    segmented = region_growing(img, seed_point, threshold)

    # Display original and segmented images
    st.image([img, segmented], caption=["Original", "Segmented"], width=200)