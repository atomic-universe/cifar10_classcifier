import streamlit as st

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import os

def app():

    print("Method is called.")

    st.title("Cifar10 Classifier Model.")
    st.write("Demonstrate the Cifar10 model.")
    
    file = st.file_uploader("Please Upload Classfier Image.",type = ['jpg','png','jepg'] )


    if file:
        image = Image.open(file)
        st.image(image,width="stretch")

        resized_image = image.resize(size=(32,32))
        img_arr = np.array(resized_image) / 255

        img_arr = img_arr.reshape((1,32,32,3))


        model = tf.keras.models.load_model(os.path.join('model','cifar10_model.keras'))


        predictions = model.predict(img_arr)

        class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

        fig,ax = plt.subplots()
        
        ax.barh(np.arange(10), predictions[0], align='center')
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('CIFAR10 Predictions')

        st.pyplot(fig)

    else :
        st.text("You have not uploaded image yet.") 



print("Befor calling.")

# if __name__ == '__app__':
#     app()


app()