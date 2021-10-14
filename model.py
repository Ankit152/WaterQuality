import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts


print("[INFO] All libraries are imported....")

data = pd.read_csv('./assets/waterProcessed.csv')
print("[INFO] Dataset loaded....")
y = data['Potability'].values
x = data.drop('Potability',axis=1)
y = to_categorical(y)
print(x.shape,y.shape)

print("[INFO] Dividing the dataset into train and test....")
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,random_state=42,stratify=y)
print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)
print("[INFO] Dividing is done....")

print("[INFO] Making the model....")

model = Sequential(name="WaterPotability")
model.add(Dense(128,input_shape=(9,),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))


checkpointer = ModelCheckpoint('WaterPotability.h5', save_best_only=True,monitor='val_loss',mode='auto')

print("[INFO] Model architecture is done....")
model.compile(loss = "categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
print("Model compiled....")
print("Training is starting....")
start = time.time()
hist = model.fit(xtrain,ytrain,batch_size=8,epochs=100,validation_data=(xtest, ytest),callbacks=[checkpointer])
print("Model training is over....")
print("Total Time taken: ",time.time()-start)
print("Model saved....")


# plotting the figures
print("[INFO] Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('./img/accuracy.png')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('./img/loss.png')
print("[INFO] Figures saved in the disk....")

model=load_model("WaterPotability.h5")
# testing the model
print("[INFO] Testing the model....")
print("[INFO] The result obtained is...\n")
model.evaluate(xtest,ytest)
