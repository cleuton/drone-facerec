# coding=utf-8
# Predict Face using CNN
# Copyright 2018 Cleuton Sampaio

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some code parts based on: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

import keras 
import keras.backend as K
import os, sys, random
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint
import numpy as np 
import cv2, dlib 
from keras.preprocessing import image as kimage
from sklearn.model_selection import train_test_split
from easytello.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, resizeAndPad

# Parâmetros: 

img_h, img_w = 64, 64 # Altura e largura das imagens
nb_class = 3  # Numero de classes
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./easytello/shape_predictor_68_face_landmarks.dat")



# Aux function conv layer: 

def conv3x3(input_x,nb_filters):
    # Prepara a camada convolucional
    return Conv2D(nb_filters, kernel_size=(3,3), use_bias=False,
               activation='relu', padding="same")(input_x)

# Create model and load weights: 

inputs = Input(shape=(img_h, img_w, 1))
x = conv3x3(inputs, 32)
x = conv3x3(x, 32)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = conv3x3(x, 64)
x = conv3x3(x, 64)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = conv3x3(x, 128)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
preds = Dense(nb_class, activation='softmax')(x)
model = Model(inputs=inputs, outputs=preds)

model.load_weights("./easytello/faces_saved.h5")

# Compila o modelo: 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Função de detecção de rostos e separação de imagens
# Retorna: vetor de detecçÕes e de imagens dos rostos já tratadas
def detectar(img): 
    detecs = detector(img, 1) # Vetor de detecção de rostos
    rostos = []
    s_height, s_width = img.shape[:2]

    for i, det in enumerate(detecs):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)
        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_height, s_width), flags=cv2.INTER_CUBIC)
        #print('ROTATED',type(rotated))
        #print(rotated)
        cropped = crop_image(rotated, det)
        #print('CROPPED',type(rotated))
        #print(cropped)
        error, squared = resizeAndPad(cropped, (img_h,img_w), 127)
        #print('SQUARED',type(rotated))
        #print(squared)        
        if error == 0:
            rostos.append(squared)

    return detecs, rostos        

# Função de nomear os rostos encontrados
# Retorna: Vetor com os nomes encontrados ou "desconhecido"
def mostraCateg(classe): 
    nome = "desconhecido"
    for idx, val in enumerate(classe[0]):
        if val == 1:
            nome = pessoas[idx]
            break
    return nome


# Função de classificação: 
# Retorna: Nomes encontrados ou "desconhecido"
def classificar(rostos):
    nomes = []
    for rosto in rostos: 
        im = kimage.img_to_array(rosto.T)
        im = np.expand_dims(im, axis = 0)    
        try:
            classe = model.predict(im, batch_size=1) 
            nomes.append(mostraCateg(classe))
        except Exception as ex: 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("ERROR predicting faces: ",ex,exc_type, fname, exc_tb.tb_lineno)
            raise ex
    return nomes


# Função de verificação: 
# Retorna: Rostos detectados e nomes reconhecidos
def verifica(imagem): 
    detecs, rostos = detectar(imagem)
    nomes = classificar(rostos)
    return detecs, nomes

# captura e executa predição: 

pessoas = ['Bill-Clinton',  'cleuton', 'Bill-Gates']

# Esta é a versão módulo, portanto você deve invocar esta função: 

def process_frame(frame):
    """Process an OpenCV image (a frame) trying to identify faces.

    Keyword arguments:
    frame -- OpenCV color image.
    
    Return: 
    new image with identified faces.
    """
    imagem = []
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s_height, s_width = imagem.shape[:2]
    detecs, nomes = verifica(imagem)
    #print("summary ",model.summary())
    for (i, rect) in enumerate(detecs):
        # Converte as marcas faciais e converte para um vetor numpy
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        try:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 255, 0), 2) 
            cv2.putText(imagem, nomes[i], (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("ERROR processing faces: ",error,exc_type, fname, exc_tb.tb_lineno)            
            pass
    return imagem


        

