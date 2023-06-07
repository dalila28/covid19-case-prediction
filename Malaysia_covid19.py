"""
Malaysia Covid-19 Case Prediction
"""
# %%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os,datetime

#2.Data loading
PATH = os.getcwd()
PATH_TRAIN = os.path.join(PATH,"cases_malaysia_train.csv")
PATH_TEST = os.path.join(PATH,"cases_malaysia_test (1).csv")
df_train = pd.read_csv(PATH_TRAIN)
df_test = pd.read_csv(PATH_TEST)

# %%
#3. Data inspection
df_train.head()
# %%
df_train.tail()
# %%
df_train.info()
# %%
df_train.isna().sum()
# %%
#4.Data cleaning
#Fill up NA values with mean() for df_train
columns_to_mean = ["cluster_import", "cluster_religious", "cluster_community", "cluster_highRisk", "cluster_education", "cluster_detentionCentre", "cluster_workplace"]

df_train[columns_to_mean] = df_train[columns_to_mean].fillna(df_train[columns_to_mean].mean())
#%%
#check again
df_train.isna().sum()
#%%
#check NA value for df_test
df_test.isna().sum()
#%%
#Fill up NA values with mean() for df_test
df_test["cases_new"] = df_test["cases_new"].fillna(df_test["cases_new"].mean())
#%%
df_test.isna().sum()
# %%
#5. Feature selection
#we're selecting "cases_new as the feature and label"
df_train_new_case = df_train["cases_new"]
df_test_new_case = df_test["cases_new"]
#%%
#Replace non-numeric values with 0 after conversion to numeric
df_train_new_case = pd.to_numeric(df_train_new_case, errors='coerce').fillna(0)

# %%
#6. Data preprocessing
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_train_new_case_scaled = mms.fit_transform(np.expand_dims(df_train_new_case,axis=-1))
df_test_new_case_scaled = mms.transform(np.expand_dims(df_test_new_case,axis=1))
# %%
#7. Data windowing
window_size = 30   #use past 30days to predict 1day

X_train = []
y_train = []

for i in range(window_size,len(df_train_new_case_scaled)):
    X_train.append(df_train_new_case_scaled[i - window_size:i])
    y_train.append(df_train_new_case_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)
# %%
df_new_case_stacked = np.concatenate((df_train_new_case_scaled, df_test_new_case_scaled))
length_days = window_size + len(df_test_new_case_scaled)
data_test = df_new_case_stacked[-length_days:]

X_test = []
y_test =[]

for i in range(window_size,len(data_test)):
    X_test.append(data_test[i - window_size:i])
    y_test.append(data_test[i])

X_test = np.array(X_test)
y_test = np.array(y_test)
# %%
#8.Model development
from tensorflow import keras 
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model
# %%
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)
# %%
#9.Model compilation
model.compile(optimizer="adam",loss='mse',metrics=['mape'])
#%%
#10.Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorbaord_logs\msiacovid19"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#11.Model training
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=[tb])
# %%
#12. Model deployment
y_pred = model.predict(X_test)
# %%
#Perform inverse transform
actual_price = mms.inverse_transform(y_test)
predicted_price = mms.inverse_transform(y_pred)
# %%
#Plot actual vs predicted
plt.figure()
plt.plot(actual_price,color='red')
plt.plot(predicted_price,color='blue')
plt.xlabel("Days")
plt.ylabel("Malaysia Covid-19 Case Prediction")
plt.legend(['Actual','Predicted'])
# %%
