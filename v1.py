import streamlit as st
import pandas as pd
import numpy as np
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn

device = 'cpu'

st.set_page_config(page_title="Flowers Classification",
                    page_icon="🌹", layout="centered",
                      initial_sidebar_state="collapsed",
                        menu_items={"Report a Bug":"mailto:makarbaderko@gmail.com",
                                    "About":"This small project was created by Makar Baderko.",})
st.title('Classifying Basic Flowers with Machine Learning')


st.markdown("### Upload your own image")

uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpeg", "jpg", "png"])

from typing import List, Tuple

from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    return (class_names[target_image_pred_label], target_image_pred_probs.max())

def sample_predict(image):
    model = torch.load("effNetb4_10epochs_v1_s2", map_location=torch.device('cpu'))
    # classes = ['astilbe',  'bellflower',  'black_eyed_susan',  'calendula',  'california_poppy',  'carnation',  'common_daisy',  'coreopsis',  'daffodil',  'dandelion',  'iris',  'magnolia',  'rose',  'sunflower',  'tulip',  'water_lily']
    classes = ['astilbe',  'bellflower',  'black eyed susan',  'calendula',  'california poppy',  'carnation',  'common daisy',  'coreopsis',  'daffodil',  'dandelion',  'iris',  'magnolia',  'rose',  'sunflower',  'tulip',  'water lily']
    flower_type, proba = pred_and_plot_image(model, image, classes)
    proba = "{:.2f}".format(proba*100)
    st.markdown(f"""<font size="4"> I'm {proba}% sure, that this is a {flower_type}. </font>""", unsafe_allow_html=True)

if uploaded_file:
    st.image(uploaded_file, width=200)
    sample_predict(uploaded_file)
    # model = torch.load("effNetb4_10epochs_v1_s2", map_location=torch.device('cpu'))
    # # classes = ['astilbe',  'bellflower',  'black_eyed_susan',  'calendula',  'california_poppy',  'carnation',  'common_daisy',  'coreopsis',  'daffodil',  'dandelion',  'iris',  'magnolia',  'rose',  'sunflower',  'tulip',  'water_lily']
    # classes = ['astilbe',  'bellflower',  'black eyed susan',  'calendula',  'california poppy',  'carnation',  'common daisy',  'coreopsis',  'daffodil',  'dandelion',  'iris',  'magnolia',  'rose',  'sunflower',  'tulip',  'water lily']
    # flower_type, proba = pred_and_plot_image(model, uploaded_file, classes)
    # st.text(f"I'm {proba*100}% sure, that this is a {flower_type}.")



st.markdown("### Sample images for predictions")
st.markdown("Image 1")
st.image("the-incomparable-waterlily-and-lotus-1403525-22-dc91986882b6494e96c142144817fff5.jpg", width=400)
if st.button(label="Predict!", key=1):
    sample_predict("the-incomparable-waterlily-and-lotus-1403525-22-dc91986882b6494e96c142144817fff5.jpg")
st.markdown("Image 2")
st.image("Unknown.jpeg", width=200)
if st.button(label="Predict!", key=2):
    sample_predict("Unknown.jpeg")
st.markdown("### Dataset")
st.markdown("Dataset which was used to train the model can be found [here](https://www.kaggle.com/datasets/l3llff/flowers)")
classes = ['astilbe',  'bellflower',  'black eyed susan',  'calendula',  'california poppy',  'carnation',  'common daisy',  'coreopsis',  'daffodil',  'dandelion',  'iris',  'magnolia',  'rose',  'sunflower',  'tulip',  'water lily']
classes_md = "\n* ".join(classes)
classes_md = '* ' + classes_md
st.markdown(f"""### Classes, on which the model was trained are: \n\n {classes_md}""")
st.markdown(f"### Author")
st.markdown("This website and the model were made by [Makar Baderko](https://www.makarbaderko.com)")
st.markdown(f"### Source code")
st.markdown(f"Code for model training, website code and everything else can be found [here](https://github.com/makarbaderko/flower-classififer)")