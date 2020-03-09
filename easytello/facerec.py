import tensorflow as tf
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, resizeAndPad
import cv2,dlib

class Processor:
    def __init__(self, model: str='faces_saved.h5', facesnames=[]):
        self.modelname = modelname
        self.facesnames = facesnames
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        loadmodel()

    def loadmodel(self):
        self.model = tf.keras.models.load(self.modelname)

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
        nomes = []
        for rosto in rostos: 
            im = image.img_to_array(rosto.T)
            im = np.expand_dims(im, axis = 0)    
            classe = self.model.predict(im, batch_size=1) 
            nomes.append(self.mostraCateg(classe))
        return nomes      

        # Função de detecção de rostos e separação de imagens
        # Retorna: vetor de detecçÕes e de imagens dos rostos já tratadas
        def detectar(self,img): 
            detecs = self.detector(img, 1) # Vetor de detecção de rostos
            rostos = []
            s_height, s_width = img.shape[:2]

            for i, det in enumerate(detecs):
                shape = self.predictor(img, det)
                left_eye = extract_left_eye_center(shape)
                right_eye = extract_right_eye_center(shape)
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(img, M, (s_height, s_width), flags=cv2.INTER_CUBIC)
                cropped = crop_image(rotated, det)
                squared = resizeAndPad(cropped, (img_h,img_w), 127)
                rostos.append(squared)

            return detecs, rostos                

    def verifica(self,imagem_cv2): 
        # Função de verificação: 
        # Retorna: Rostos detectados e nomes reconhecidos
        detecs, rostos = detectar(imagem)
        nomes = self.classificar(rostos)
        return detecs, nomes

    def predict(self,imagem_cv2):
        s_height, s_width = imagem_cv2.shape[:2]
        detecs, nomes = verifica(imagem_cv2)
        for (i, rect) in enumerate(detecs):
            # Converte as marcas faciais e converte para um vetor numpy
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            cv2.rectangle(imagem_cv2, (x, y), (x + w, y + h), (255, 255, 0), 2) 
            cv2.putText(imagem_cv2, nomes[i], (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)    
        cv2.imshow('Resultado', imagem_cv2)
        cv2.waitKey(0)    
