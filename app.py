import streamlit as st
import torch
import numpy as np
from models import *
from matplotlib import pyplot as plt

@st.cache
def load_model():
    model = VAE()
    model.load_state_dict(torch.load("./models/face_generatorE51v10.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

# model_load_state = st.text("Model Loading...")
model = load_model()
# model_load_state.text("Model Loaded")
# st.write("Model Architecture\n", model)
st.title("Face Generation with Variational Autoencoders")
st.header("Demo")

dial1 = st.slider("Feature 1", 0, 100, np.random.randint(0, 100)) / 100.0
dial2 = st.slider("Feature 2", 0, 100, np.random.randint(0, 100)) / 100.0
dial3 = st.slider("Feature 3", 0, 100, np.random.randint(0, 100)) / 100.0
dial4 = st.slider("Feature 4", 0, 100, np.random.randint(0, 100)) / 100.0
dial5 = st.slider("Feature 5", 0, 100, np.random.randint(0, 100)) / 100.0

z = [dial1, dial2, dial3, dial4, dial5] + [np.random.normal() for i in range(35)]
z = np.array(z)
z = torch.tensor(z, dtype=torch.float64)

with torch.no_grad():
    output = model.decode(z.float()).detach().squeeze()
    plt.figure(figsize=(4, 4))
    plt.imshow(output, cmap="gray")
    plt.axis("off")
    st.pyplot()
    #st.image(output.numpy(), width=None)
