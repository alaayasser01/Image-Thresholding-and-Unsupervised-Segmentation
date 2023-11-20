import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
import segmentation as s
from agglomerative import *
import thresholding as t

st.set_page_config(layout="wide")


def main():
    selected = option_menu(
        menu_title=None,
        options=['Thresholding', 'Segmentation'],
        orientation="horizontal"
    )

    if selected == "Segmentation":
        with st.sidebar:

            uploaded_image = st.file_uploader(
                "Upload Image", type=["jpg", "jpeg", "png"])

            algo = st.selectbox("Choose Segmentation Algorithm:", [
                                'K-Means', 'Region Growing', 'Agglomerative', 'Mean Shift'])

            segment = st.button("Segment")

        image_col, edited_col = st.columns(2)
        if uploaded_image:
            with image_col:
                st.image(uploaded_image, use_column_width=True)

        if algo == "K-Means":
            with st.sidebar:
                k = st.slider("Number of Clusters", 2, 10, 2, 1)
        elif algo == "Region Growing" and uploaded_image:
            with st.sidebar:
                threshold = st.sidebar.slider("Threshold", 1, 15, 10)
                file_bytes = np.asarray(
                    bytearray(uploaded_image.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                seed_point_x = st.sidebar.slider(
                    "Seed point X coordinate", 0, img.shape[1], img.shape[1] // 2)
                seed_point_y = st.sidebar.slider(
                    "Seed point Y coordinate", 0, img.shape[0], img.shape[0] // 2)
        elif algo == "Agglomerative":
            with st.sidebar:
                k = st.slider("Number of Clusters", 2, 10, 2, 1)

        if segment and algo == "K-Means":
            segmented_img = s.kmeans_segmentation(
                k, f"images/{uploaded_image.name}")

            with edited_col:
                st.image(segmented_img, use_column_width=True)
        elif segment and algo == "Region Growing":
            # Seed point selection
            seed_point = (seed_point_x, seed_point_y)

            # Region growing segmentation
            segmented = s.region_growing(img, seed_point, threshold)
            with edited_col:
                st.image(segmented, use_column_width=True)

        elif segment and algo == "Agglomerative":
            image = cv2.imread(f"images/{uploaded_image.name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image.reshape((-1, 3))
            agglo = AgglomerativeClustering(n=k, initial_n=25)
            agglo.fit(pixels)
            new_img = [[agglo.center_pred(pixel)
                        for pixel in row] for row in image]
            new_img = np.array(new_img, np.uint8)

            with edited_col:
                st.image(new_img, use_column_width=True)

        elif segment and algo == "Mean Shift":
            image = plt.imread(f"images/{uploaded_image.name}")

            bw = 0.1*np.max(image)
            segmented_image, num_clusters = s.meanShift(image, bw, 3)
            with edited_col:
                st.image(segmented_image, use_column_width=True)
    elif selected == "Thresholding":
        with st.sidebar:

            uploaded_image = st.file_uploader(
                "Upload Image", type=["jpg", "jpeg", "png"])

            method = st.selectbox("Choose Thresholding Method:", [
                'Optimal', 'Otsu', 'Spectral', 'Local'])

            apply = st.button("Threshold")

        image_col, edited_col = st.columns(2)
        if uploaded_image:
            with image_col:
                st.image(uploaded_image, use_column_width=True)
        if method == "Local":
            b_size = st.sidebar.slider("Block Size", 5, 30, 15, 1)

        if apply and method == "Optimal":
            img = np.array(Image.open(
                f"images/{uploaded_image.name}").convert('L'))
            optimal_threshold = t.optimal_threshold(img)
            edited_img = t.apply_threshold(img, optimal_threshold)
            with edited_col:
                st.image(edited_img, use_column_width=True)

        elif apply and method == "Otsu":
            img = np.array(Image.open(
                f"images/{uploaded_image.name}").convert('L'))
            otsu_threshold = t.otsu_threshold(img)
            edited_img = t.apply_threshold(img, otsu_threshold)
            with edited_col:
                st.image(edited_img, use_column_width=True)

        elif apply and method == "Spectral":
            img = np.array(Image.open(
                f"images/{uploaded_image.name}").convert('L'))
            spectral_threshold = t.spectral_threshold(img)
            edited_img = t.apply_threshold(img, spectral_threshold)
            with edited_col:
                st.image(edited_img, use_column_width=True)

        elif apply and method == "Local":
            img = np.array(Image.open(
                f"images/{uploaded_image.name}").convert('L'))
            # optimal_threshold = t.optimal_threshold(img)
            edited_img = t.localthresholding(img, b_size)
            with edited_col:
                st.image(edited_img, use_column_width=True)


if __name__ == '__main__':
    main()
