
import numpy as np
import tensorflow as tf
import pandas as pd

dataframe = pd.read_csv("./grade_dentist.csv")
dataframe = dataframe.dropna()
x_train = []
y_train = dataframe['합격여부'].values
for i, value in enumerate(y_train):
    if value == '불합격':
        y_train[i] = 0.0
    elif value =="합격":
        y_train[i] = 1.0
    else:
        y_train[i] = 0.0
y_train = tf.constant(np.array(y_train).astype(np.float32))
print(y_train.shape)
for i, row in dataframe.iterrows():
    x_train.append(row['총점'])
x_train = tf.constant(np.array(x_train).astype(np.float32))
print(x_train.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs = 5)
pr = model.predict([[210.0], [211.0], [212.0], [213.0]])
print(pr)