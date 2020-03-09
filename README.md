# Drone Facerec
[**Cleuton Sampaio**](https://github.com/cleuton), Março de 2020
My [**LinkedIn** profile](https://www.linkedin.com/in/cleutonsampaio/).

![](./im.png)

Uma demonstração de reconhecimento facial utilizando [**Keras**](https://keras.io/), [**ensorflow**](https://www.tensorflow.org/), python e o drone [**Tello**](https://store.dji.com/shop/tello-series), da **DJI**.

Este projeto é baseado em dois outros projetos do Github: 

1. [**FaceREC - CNN implementation of facial recognition**](https://github.com/cleuton/facerec_cnn), criado por [**mim**](https://github.com/cleuton);
2. [**EasyTello**](https://github.com/Virodroid/easyTello), criado por **Ezra Fielding**.

## Visão geral

Este software controla um drone Tello, capturando o vídeo streaming gerado por sua câmera de bordo. Ele intercepta cada frame e tenta reconecer rostos que aparecem na imagem, utilizando um modelo de rede [**Convolucional**](https://github.com/cleuton/FaceGuard/tree/master/CNN) que eu treinei, baseado em imagens do dataset  [**Labeled Faces in the Wild**](http://vis-www.cs.umass.edu/lfw/), e algumas fotos minhas. 

É uma prova de conceito do uso de inteligência artificial com drones e, por que não, de Internet das Coisas (IoT). Note que a performance pode ser baixa, afinal de contas, estou usando um laptop e um drone barato e simples. Com mais recursos, é possível obter melhor desempenho da solução.

## Antes de usar

Há algumas coisas que você precisa fazer antes de usar esta aplicação. Para começar, ela foi feita exclusivamente para o Drone Tello, mas se aplica a qualquer outro drone que possua uma API de programação python.

A primeira coisa a fazer é conseguir controlar seu drone usando apenas o [**easyTello**](https://github.com/Virodroid/easyTello) e depois tentar usar este projeto aqui. Por que? Bem, não é tão simples quanto parece...

Se você acabou de comprar seu drone Tello, precisa verificar qual é a versão do **Firmware** dele. Isso pode ser feito através da app [**Tello**](https://www.ryzerobotics.com/tello) (se não baixou, é melhor fazer logo).

Primeiramente, ligue o drone, conecte-se à rede WiFi dele com seu Smartphone (TELLO...), abra a app e clique no botão de configurações, como na imagem: 

![](./app1.jpg)

Depois, clique em **More**: 

![](./app2.jpg)

Clique no botão com três pontos: 

![](./app3.jpg)

Agora, verifique o número da versão do firmware. Se estiver como 1.3 (alguma coisa), é preciso atualizar para 1.4!

![](./app4.jpg)

O processo é feito em duas partes: Baixar o firmware para o Smartphone e atualizar no drone. Para baixar a nova versão do firmware, desconecte do WiFi do drone e conecte à Internet. Usando a app do Tello, clique no botão **Update** para baixar o firmware. Depois que baixar, conecte-se ao WiFi do drone e clique novamente no botão **Update**.

Depois do processo, verifique a nova versão do firmware. 

**Atenção**: Um indício que a API está desatualizada é quando o drone não reconhece o comando **streamon**!

Se você atualizou corretamente o firmware, eu recomendo que tente rodar o [**easyTello**](https://github.com/Virodroid/easyTello) original, seguindo as instruções no repositório dele. Com isso, você confirma que a API de controle está funcionando corretamente. Depois, pode baixar e rodar este repositório aqui. 

Isso feito, clone este repositório e crie um ambiente [**Anaconda**]() usando o script [**conta-env.yml**](./conda-env.yml): 

```
conda env create -f conda-env.yml
```

Ative o ambiente antes de usar este software: 

```
conda activate drone-facerec
```

## Criando um arquivo de modelo

Para fazer o reconhecimento facial, é necessário treinar uma rede neural. Eu não inserir o projeto [**FaceREC**](https://github.com/cleuton/facerec_cnn) aqui, nem mesmo o arquivo de modelo que eu treinei. Recomendo que você clone o projeto original e treine seu modelo, copiando o arquivo **HDF5** criado para a pasta **easyTello**, dentro deste projeto. 

Baixe e extraia o arquivo [**shape_predictor_68_face_landmarks**](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) file, e coloque no projeto FaceRec_CNN e dentro da pasta **easytello** deste projeto!

Gerar um modelo é simples: 

1. **Obtenha várias fotos do rosto da mesma pessoa**:
a) Use o [**Labeled Faces in the Wild**](http://vis-www.cs.umass.edu/lfw/) e baixe várias fotos da mesma pessoa, copiando para a pasta **raw** do projeto FaceRec_CNN;
b) Nomeie os arquivos neste padrão: fulano-de-tal.nnnn.jpg (não use espaços, numere fotos do mesmo sujeito, separando por pontos). Por exemplo: "bill-clinton.0001.jpg";
c) Tire várias fotos do seu rosto (e de quem mais deseja reconhecer) e salve na pasta **raw** seguindo a mesma nomenclatura do passo "b";
d) Cuide para que haja apenas um único rosto em cada foto de treinamento! Se houver mais de um rosto, corte a foto;

2. **Converta as fotos**: 
a) O script **trainCNN.py** vai rotacionar e cortar os rostos, transformando em imagens monocromáticas. Ele vai separar em fotos de treino e de teste (pastas "train" e "test") de acordo com a variável **train_test_ratio = 0.3**. Se mantiver em 30%, então 70% das imagens serão para treino e as outras para teste;
b) Anote as categorias encontradas! O programa exibirá um vetor com os nomes encontrados. Anote para mudar no script de predição (**predict.py**) e no script de reconhecimento deste projeto (**facerec.py**);
c) Se houver 4 pessoas na sua pasta **raw**, ele tem que separar 4 pessoas em **train** e 4 pessoas em **test**. Com poucas imagens, pode acontecer de ficarem menos pessoas em **test** e isso dará erro.

3. **Obtenha os nomes das pessoas e o arquivo de modelo**: 
a) Os nomes das pessoas são exibidos na console, após o treinamento. Copie esse vetor e altere no **facerec.py**;
b) O arquivo de modelo terá o nome **'faces_saved.h5'** copie-o para a pasta **easytello** deste projeto;


## Controlando o drone

O script [**teste.py**](./teste.py) controla o drone. Ele faz basicamente 2 coisas: coloca em modo comando e inicia a captura de vídeo. Mas você pode fazer muito mais! Pode fazer o drone decolar, ir para a frente, ou virar e depois pousar. Há alguns comandos comentados que você pode usar. Se quiser saber mais sobre os comandos que o Tello aceita, [**veja na documentação**](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf).

Certifique-se de haver colocado o arquivo **h5** na pasta easyTello! E mude o nome dentro do arquivo [**tello.py**](./easytello/tello.py).

**Atenção**: Se você fizer o drone decolar, tenha certeza de haver espaço para isso! Se ele bater no teto ou nas paredes, pode ficar danificado! E cuidado ao tentar usar o Tello no ambiente externo (não recomendável). Se ele se afastar por mais de 10 metros, pode ficar fora do alcance do WiFi e colidir com alguma coisa. Teoricamente, se ele perder o contato, ele pousa automaticamente. 









