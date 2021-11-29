import sys
sys.path.append(".")

from game_class import tictactoe
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)

tf.random.set_seed(42)
import cv2



ttt=tictactoe()


df=ttt.combine_existing_data()
df.dropna(inplace=True)
df=df[(df.player!='0')]
df=df[(df.next_move!=0)].reset_index(drop=True)

df=df.drop(['player'],axis=1)
df['next_move']=df['next_move'].astype(int)
df['next_move']=df['next_move']-1    # to fix label indexing, now 1 is 0 and 9 is 8 

df=df[['1_played', '2_played', '3_played', '4_played', '5_played', '6_played',
       '7_played', '8_played', '9_played', '1_threat', '2_threat', '3_threat',
       '4_threat', '5_threat', '6_threat', '7_threat', '8_threat', '9_threat',
       '1_win', '2_win', '3_win', '4_win', '5_win', '6_win', '7_win', '8_win',
       '9_win', 'next_move']] 

x=np.array(df.drop('next_move',axis=1),dtype='float32')
y=np.array(df['next_move'])




class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        try:
            if(logs.get('val_acc')>VALIDATION_ACC_CAP):
                self.model.stop_training = True
        except:
            if(logs.get('val_accuracy')>VALIDATION_ACC_CAP):
                self.model.stop_training = True

VALIDATION_ACC_CAP = 0.62
callbacks=myCallback()

model = keras.Sequential([keras.layers.Dense(units=128,
                                             input_shape=(27,),
                                             activation='relu'),
                          keras.layers.BatchNormalization(),
                          keras.layers.Dropout(rate=0.3),
                          keras.layers.Dense(units=128,
                                             activation='relu'),
                          keras.layers.BatchNormalization(),
                          keras.layers.Dropout(rate=0.3),
                          keras.layers.Dense(units=9,
                                             activation='softmax')

                        ])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history=model.fit(x,
          y,
          epochs=100,
          validation_split=0.2,
          verbose=2,
#           callbacks=[callbacks]
                )
def model_summary():
    metric = history.history
    print(model.summary())
    fig, ax = plt.subplots(2, 1,figsize=(10,10))
    ax[0].plot(range(1,len(metric['accuracy'])+1),
             metric['accuracy'],
             label=f"Accuracy  : {round(metric['accuracy'][-1],2)}",
            )
    ax[0].plot(range(1,len(metric['accuracy'])+1),
             metric['val_accuracy'],
             label=f"Val Accuracy  : {round(metric['val_accuracy'][-1],2)}",
             linestyle="--"
            )   
    ax[0].legend()
    ax[0].set_title("ACCURACY")
    ax[1].plot(range(1,len(metric['accuracy'])+1),
             metric['loss'],
             label=f"Loss  : {round(metric['loss'][-1],2)}",
             linestyle="--"
            )
    ax[1].plot(range(1,len(metric['accuracy'])+1),
             metric['val_loss'],
             label=f"Val Loss  : {round(metric['val_loss'][-1],2)}",
             linestyle="--"
            )

    ax[1].legend()
    ax[1].set_title("LOSS")
    ax[1].set_xlabel("Epoch")
    plt.show()
    
    
model_summary()
plot_model(model=model,show_shapes=True)
img = cv2.imread('model.png',0)
cv2.imshow('model.png',img)

def save_model(name):
    global model
    model.save("Models/"+name)

