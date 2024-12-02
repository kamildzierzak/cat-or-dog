import streamlit as st
import pandas as pd
from fastai.vision.all import load_learner
from PIL import Image

import pathlib

temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

model = load_learner("model.pkl")

def main():

	st.set_page_config(page_title="Is it a cat? Is it a dog? I'll tell you!")

	overview = st.container()
	upload = st.container()


	with overview:
		st.title("Cat 🐈 or Dog 🐕?")
		st.write("This is a simple web app that can tell you if the image you uploaded is a cat or a dog.")

	with upload:
		st.subheader("Upload image of a cat or a dog.")
		uploaded_image = st.file_uploader("Choose an file", type=["jpg", "jpeg", "png"])
		
		if uploaded_image is not None:
			image = Image.open(uploaded_image)
			st.image(image, caption="Uploaded image")

			pred, pred_idx, probs = model.predict(image)
			if pred == "cat":
				st.success("It's a cat!", icon="🐈")
			else:
				st.success("It's a dog!", icon="🐕")

			st.info(f"Probability: {probs[pred_idx]:.2f}")

if __name__ == "__main__":
    main()
