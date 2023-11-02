### 1026
```py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
custom_adam = optimizers.Adam()

import tensorflow as tf
data = pd.read_csv('Titanic-Dataset_1.csv')

#print(data[0:10])

for col in ['Age']:
    data[col].fillna(data[col].median(),inplace=True)
    
#print(data[0:30])
    #print(data.to_string())
Xo = data.drop("Survived", axis=1)
y = np.ravel(data['Survived'])

scalerX = StandardScaler().fit(Xo)
X = scalerX.transform(Xo)

#print(X)
#print(y)
#data Visualization
#colormap = np.array(['b', 'r'])
#plt.scatter(data["present"], data["midterm"], s=59, c=colormap[data["down"]])

#plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5,random_state=50)

# 建立模型 -------------------------------------------------
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([tf.keras.layers.Dense(4, activation = 'relu'),
                             tf.keras.layers.Dense(1, activation = 'sigmoid')])
#model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1),
 #                          tf.keras.layers.Dense(1)])
model_1.compile(loss = 'binary_crossentropy',optimizer = custom_adam,metrics = ['acc'])

checkpointer = ModelCheckpoint(filepath = 'd:/best_titanic_1026.hdf5',
                               monitor = 'val__loss',mode='min',
                               verbose=1, save_best_only=True)
hist = model_1.fit(X_train, y_train, epochs=100,
                   steps_per_epoch=100,
                   validation_data = (X_test, y_test),
                   callbacks=[checkpointer])
#model_1.fit(X, y, epochs = 10000, verbose = 0)
#-----------------------------------------------------------
```
