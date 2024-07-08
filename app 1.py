# Código .py para activar una interfaz gráfica que permita dibujar un número y 
# que el modelo lo identifique en caso de que no funcione en el notebook
import gradio as gr 
import tensorflow as tf 
import numpy as np 

modelo = tf.keras.models.load_model("mnist_model.h5") # Es posible que a veces de error por la ubicación del modelo

def clasificar_imagenes(img):
    img = np.reshape(img, (28, 28, 1)).astype("float32") / 255
    predicciones = modelo.predict(img)
    digito_predicho = np.argmax(predicciones)
    return str(digito_predicho)

interfaz = gr.Interface(fn=clasificar_imagenes , inputs="sketchpad", outputs="label")
interfaz.launch()