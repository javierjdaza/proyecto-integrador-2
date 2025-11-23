import streamlit as st 
import os
import json
from streamlit_option_menu import option_menu
import pandas as pd
st.set_page_config(page_title='Proyecto Integrador',page_icon=":brain:",initial_sidebar_state="collapsed")
import pickle
import streamlit as st
from PIL import Image

import torch.nn as nn
from torchvision import transforms as T, models
import torch
from model_utils.AgePredictorCORAL import CoralEfficientNetV2
from model_utils.AgePredictorRegression import EfficientNetV2Regression
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2
import gdown
import os

def download_models():
    os.makedirs("models", exist_ok=True)

    model_path1 = "models/EfficientNetLCORAL.pth"
    model_path2 = "models/EfficientNetLRegression.pth"
    file_id1 = "1L0CKDaVJtHdEtxvfYy-fG6Jk6fLrrXEI"  
    file_id2 = "1Ys_spKTYOTVanEeICC_wtxcdm8Pb4EAR"  
    url1 = f"https://drive.google.com/uc?id={file_id1}"
    url2 = f"https://drive.google.com/uc?id={file_id2}"

    if not os.path.exists(model_path1) or not os.path.exists(model_path2):
        print("Descargando modelo desde Google Drive...")
        gdown.download(url1, model_path1, quiet=False)
        gdown.download(url2, model_path2, quiet=False)
    else:
        print("Los modelos ya existen en disco")

    return model_path1, model_path2



class AgePredictorRegression:
    def __init__(self, weights_path = './models/EfficientNetLRegression.pth', device="cpu"):
        self.device = torch.device(device)

        # 1. Crear modelo
        self.model = EfficientNetV2Regression().to(self.device)

        # 2. Cargar pesos
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

        # 3. Definir transform para inference (val_transform)
        self.transform = T.Compose([
            T.Resize((244, 244)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    
    def load_image(self, image_path):
        
        img_pillow = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pillow)
        return img_array

    def retina_face(self, image_array):
        
        img_objs = DeepFace.extract_faces(
            img_path = image_array,
            detector_backend = 'retinaface', 
            align = True, 
            expand_percentage = 10,
            enforce_detection = False
        )
        
        area = img_objs[0]['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']
        face_crop = image_array[y:y+h, x:x+w]
    
        face_img = Image.fromarray(face_crop)
        return image_array, face_img, (x, y, w, h)
    
    def predict(self, image_path):
        
        
        original_img, img_array, _ = self.retina_face(image_path)
        img = self.transform(img_array).unsqueeze(0).to(self.device)  # (1,3,H,W)

        with torch.no_grad():
            output = self.model(img) 
            pred_age = output.item()

        pred_age = int(pred_age)

        return pred_age, img_array, original_img
    

class CoralAgePredictor:
    def __init__(self, weights_path='./models/EfficientNetLCORAL.pth', device="cpu", max_age=80):
        self.device = torch.device(device)
        self.max_age = max_age

        # 1. Crear modelo CORAL
        self.model = CoralEfficientNetV2(max_age=max_age).to(self.device)

        # 2. Cargar pesos
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

        # 3. Transform
        self.transform = T.Compose([
            T.Resize((244, 244)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def load_image(self, image_path):
        img_pillow = Image.open(image_path).convert("RGB")
        return np.array(img_pillow)

    def retina_face(self, image_array):
        
        img_objs = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend='retinaface',
            align=True,
            expand_percentage=10,
            enforce_detection=False
        )
        
        area = img_objs[0]['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']
        face_crop = image_array[y:y+h, x:x+w]
        face_img = Image.fromarray(face_crop)

        return image_array, face_img, (x, y, w, h)

    def predict(self, image_path):
        original_img, face_img, _ = self.retina_face(image_path)

        img = self.transform(face_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)  # (1, max_age)
            pred_age = self.predict_age(logits).item()  # convierte umbrales → edad

        return int(pred_age), face_img, original_img
    
    def predict_age(self, logits):
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)




st.markdown("<h1 style='text-align: center; color: #000000;'>Proyecto Integrador 2</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #000000;'>Maestría en Ciencia de Datos y Analítica</h3>", unsafe_allow_html=True)
st.write('---')
menu_selection = option_menu(None, ["Age Predictor", "The Team"], 
    icons=['robot', 'cup'], 
    menu_icon="cast", default_index=1, orientation="horizontal")
st.write('---')
st.write(' ')
st.write(' ')

if menu_selection == 'Age Predictor':
    # f1,f2 = st.columns(2,  vertical_alignment = 'center', gap = 'medium', border = True)
    # f3,f4 = st.columns(2,  vertical_alignment = 'center', gap = 'medium', border = True)
    # f5,f6 = st.columns(2, vertical_alignment = 'center', gap = 'medium', border = True)
    # with f1:
        # Widget de carga
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    regression_predictor = AgePredictorRegression(device="cpu")
    coral_predictor = CoralAgePredictor(device="cpu")
    # Mostrar la imagen
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        pred_age, img_array, original_img = regression_predictor.predict(img_array)
        pred_age_coral, img_array_coral, original_img_coral = coral_predictor.predict(img_array)
        c1,c2,c3 = st.columns(3)

        with c1:
            st.image(img_array, caption = f"EfficientNetV2Regression\nEdad Predicha: {pred_age}", use_column_width=True)

        with c3:
            st.image(img_array_coral, caption = f"EfficientNetV2CORAL\nEdad Predicha: {pred_age_coral}", use_column_width=True)
        


    



def put_img(name,role, img_file_name):
    st.image(f'./{img_file_name}.png', width = 'stretch')
    st.markdown(f"<h5 style='text-align: center; color: #000000;'>{name}<br>{role}</h5>", unsafe_allow_html=True)


if menu_selection == 'The Team':
    c1,c2,c3 = st.columns(3)

    with c1:
        put_img('Javier Daza', 'Ingeniero Industrial', './images/javier')
    with c2:
        put_img('Maria Sofia Uribe', 'Ingeniera Matematica', './images/sofia')
    with c3:
        put_img('Pablo Jimeno', 'Ingeniero de Procesos', './images/pablo')
        
    
