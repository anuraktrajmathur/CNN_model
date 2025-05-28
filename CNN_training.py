import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

path = os.path.dirname(__file__)
cur_path = os.path.join(path, 'GTSRB')


# Load train.csv
train_df = pd.read_csv(os.path.join(cur_path, 'Train.csv'))

# Extract the round signs 
round_signs = [0,1,2,3,4,5,7,8,9,10,15,16,17]
train_df = train_df[(train_df['ClassId'].isin(round_signs))]

# Add 'y_speed_limit_sign' column
train_df['y_speed_limit_sign'] = train_df['ClassId'].apply(lambda x: 1 if x in range(0,8) else 0) #0 is wrong

# Add the speed limit numbers instead of 'ClassId'
train_df.rename(columns={'ClassId':'Speed_limit'},inplace=True)

def assign_limits(x):
    if x not in range(6) and (x not in range(7,9)):
        return 6
    else: 
        return x
    
# train_df['Speed_limit'] = train_df['Speed_limit'].apply(assign_limits)

# Correct the path to images
# train_df['Path'] = train_df['Path'].apply(lambda x: cur_path + x)

features = []
classes = []
labels = []

for foldername in round_signs:
    folder_path = os.path.join(cur_path,'Train',str(foldername))
    images = os.listdir(folder_path)
    for image in images:
        try: 
            img = cv2.imread(folder_path+'\\'+image)
            img = cv2.resize(img, (30,30))
            img = np.array(img)
            features.append(img)
            classes.append(foldername)
        except:
            print("Error loading image")

for label in classes:
    labels.append(assign_limits(label))

features = np.array(features)
labels = np.array(labels)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

#Converting the labels into one hot encoding
num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train,num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

epochs = 10
batch_size = 32
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
# cnn_model.add(Dropout(rate=0.25))

cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
# cnn_model.add(Dropout(rate=0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(rate=0.5))
cnn_model.add(Dense(9, activation='softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

cnn_model.save(path + '\\CNN_model.keras')