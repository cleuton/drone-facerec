import tensorflow as tf
from easytello.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, resizeAndPad
import cv2,dlib
import keras
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.preprocessing import image as Kimage
import os, sys
import numpy as np

img_h, img_w = 64, 64

class Processor:
    def __init__(self,facesnames=[],modelname='faces_saved.h5'):
        self.img_h, self.img_w = 64, 64 # Altura e largura das imagens
        self.loaded = False
        self.modelname = modelname
        self.facesnames = facesnames
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./easytello/shape_predictor_68_face_landmarks.dat")
        self.loadmodel()

    def conv3x3(self,input_x,nb_filters):
        # Prepara a camada convolucional
        return Conv2D(nb_filters, kernel_size=(3,3), use_bias=False,
                activation='relu', padding="same")(input_x)

    def loadmodel(self):
        nb_class=len(self.facesnames)
        inputs = Input(shape=(img_h, img_w, 1))
        x = self.conv3x3(inputs, 32)
        x = self.conv3x3(x, 32)
        x = MaxPooling2D(pool_size=(2,2))(x) 
        x = self.conv3x3(x, 64)
        x = self.conv3x3(x, 64)
        x = MaxPooling2D(pool_size=(2,2))(x) 
        x = self.conv3x3(x, 128)
        x = MaxPooling2D(pool_size=(2,2))(x) 
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        preds = Dense(nb_class, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=preds)

        model.load_weights(self.modelname)

        # Compila o modelo: 

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
        self.model = model
        self.loaded = True

    # Função de nomear os rostos encontrados
    # Retorna: Vetor com os nomes encontrados ou "desconhecido"
    def mostraCateg(self,classe): 
        nome = "desconhecido"
        for idx, val in enumerate(classe[0]):
            if val == 1:
                nome = pessoas[idx]
                break
        return nome        

    def classificar(self,rostos):
        # Função de classificação: 
        # Retorna: Nomes encontrados ou "desconhecido"
        try:
            nomes = []
            for rosto in rostos: 
                #print(type(rosto))
                #print(dir(rosto))
                #print(rosto)
                im = Kimage.img_to_array(rosto.T)
                im = np.expand_dims(im, axis = 0)    
                classe = self.model.predict(im, batch_size=1) 
                nomes.append(self.mostraCateg(classe))
            return nomes     
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("###ERROR6: ",error,exc_type, fname, exc_tb.tb_lineno)  
            raise error        

    # Função de detecção de rostos e separação de imagens
    # Retorna: vetor de detecçÕes e de imagens dos rostos já tratadas
    def detectar(self,img): 
        try:
            detecs = self.detector(img, 1) # Vetor de detecção de rostos
            #print("Detectados: ",len(detecs))
            rostos = []
            s_height, s_width = img.shape[:2]

            for i, det in enumerate(detecs):
                shape = self.predictor(img, det)
                left_eye = extract_left_eye_center(shape)
                right_eye = extract_right_eye_center(shape)
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(img, M, (s_height, s_width), flags=cv2.INTER_CUBIC)
                #print('ROTATED',type(rotated))
                #print(rotated)
                cropped = crop_image(rotated, det)
                #print('CROPPED',type(rotated))
                #print(cropped)
                error, squared = resizeAndPad(cropped, (self.img_h,self.img_w), 127)
                #print('SQUARED',type(rotated))
                #print(squared)                   
                rostos.append(squared)
            return detecs, rostos
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("###ERROR5: ",error,exc_type, fname, exc_tb.tb_lineno)  
            raise error                

    def verifica(self,imagem_cv2): 
        # Função de verificação: 
        # Retorna: Rostos detectados e nomes reconhecidos
        try: 
            detecs, rostos = self.detectar(imagem_cv2)
            #print(rostos)
            nomes = self.classificar(rostos)
            return detecs, nomes
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("###ERROR4: ",error,exc_type, fname, exc_tb.tb_lineno)  
            raise error            

    def predict(self,imagem_cv2):
        try: 
            s_height, s_width = imagem_cv2.shape[:2]
            detecs, nomes = self.verifica(imagem_cv2)
            for (i, rect) in enumerate(detecs):
                # Converte as marcas faciais e converte para um vetor numpy
                print("Reconheceu!")
                x = rect.left()
                y = rect.top()
                w = rect.right() - x
                h = rect.bottom() - y
                cv2.rectangle(imagem_cv2, (x, y), (x + w, y + h), (255, 255, 0), 2) 
                cv2.putText(imagem_cv2, nomes[i], (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)    
            cv2.imshow('Resultado', imagem_cv2)  
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("###ERROR3: ",error,exc_type, fname, exc_tb.tb_lineno)  
            raise error         
