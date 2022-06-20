# In[1]:

import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout#ANN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten #CNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics

import sys
import numpy as np

try:
    root
except:
    root = os.getcwd()
# print(root)


# In[2]:

## Steuerungsparemeter:
save_model = False
save_figures = False
figs = {}

mode = 'convolution'
# mode = "dense"
implemented_modes = ['dense', 'convolution']
if mode not in implemented_modes:
    raise ValueError(f"mode {mode} is not implemented, valid options are {implemented_modes}")

num_out = 2
val_share = 0.25

epochs_restore = 2

epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])

batch_size = int(sys.argv[3])
num_filters = [int(sys.argv[4]), int(sys.argv[5])] # only applicable to convolution network
window = 5
pool = 3

# number of neurons in last ´len(num_hidden)´ hidden dense layers
num_hidden = [int(sys.argv[6])]


# In[8]:


os.chdir(root)
#Noise?
noisy = 'noisy'
noisy_options = ('noisy', 'clean')
if noisy not in noisy_options:
    raise ValueError(f"Noisy has to be one of the options: {noisy_options}")
#noisy = True
channels = ['eps', 'sig']
extra_channel = True
if extra_channel:
    channels.append(extra_channel)
num_channels = len(channels)
# inp_length = 201 #Zielanzahl der Datenpunkte = Anzahl der Neuronen im Input-Layer
# Channel 1: Spannung, Channel 2: Dehnung, Channel 3: 0/1 (Wert weg/vorhanden)
# path_training = f"training_data\\non-linear-shear-2P\\{noisy}\\{channels}"
path_training = "."
path_training = os.path.join(root, path_training)
os.chdir(path_training)
if noisy == 'noisy':
    noisy = True
else:
    noisy = False


# In[9]:


y_data_original = np.load("/home/paul/dev/bachelor-thesis/code/data/parameter.train")
num_out = y_data_original.shape[1]
num_all = y_data_original.shape[0]
x_data_original = np.load("/home/paul/dev/bachelor-thesis/code/data/prepared_data.train")
num_in = x_data_original.shape[2]

if num_channels == 1:
    x_data_original = x_data_original[:,0,:].reshape([x_data_original.shape[0],1,x_data_original.shape[2]])

print(0)

# split training and test data
x_train_unscaled, x_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(x_data_original, y_data_original, test_size=val_share, random_state=42)

print(1)
# In[12]:


# scale output data to a range of 0, 1
scaler = MinMaxScaler()
# fit scaler on training data
scaler.fit(y_train_unscaled)
# apply transform
y_train = scaler.transform(y_train_unscaled)
y_test = scaler.transform(y_test_unscaled)

print(2)
# In[13]:

x_train = x_train_unscaled
x_test = x_test_unscaled

# dimension ordering was optimized for pytorch, needs to be adjusted for tensorflow
x_train = np.swapaxes(x_train, 1, 2)
x_test = np.swapaxes(x_test, 1, 2)

print(3)
# In[14]:


# model build up

initializer = tf.keras.initializers.GlorotNormal()
model = Sequential()
# mode = 'dense'
if mode == "dense":
    
    x_train = x_train[:, :, 1]
    x_test = x_test[:, :, 1]
    for layer, num in enumerate(num_hidden):
        if layer == 0:
            model.add(Dense(num, input_dim=x_train.shape[-1], activation='relu', kernel_initializer=initializer)) # definition 1. Hidden-Layer und Definition der Anzahl Neuronen des input-layer mittels "input_dim" 
        else:
            model.add(Dense(num, activation='relu', kernel_initializer=initializer)) # Hidden num

    model.add(Dense(y_train.shape[1], activation="linear", kernel_initializer=initializer)) # Output Definition der Zielgrößen 
    
elif mode == "convolution":

    for layer, filters in enumerate(num_filters):
        if layer == 0:
            model.add(Conv1D(filters, window, #mit 32 Filtern und Maskengröße 5 (die gesamte Kurve wird mit Maske 5 32 mal "abgefahren" (jeweils anderer Filter))
          activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
          # The first argument of input_shape has to be the number of input points, the second argument has to be the dimension of each point
        else:
            model.add(Conv1D(filters, window, activation='relu'))
        model.add(MaxPooling1D(pool)) #Kondensieren auf pool Werte  

    model.add(Flatten())
    for neurons in num_hidden:
        model.add(Dense(neurons, activation='relu', kernel_initializer=initializer))
    
    model.add(Dense(y_train.shape[1], activation="linear", kernel_initializer=initializer))

print(4)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer=opt)

model.summary()
# keras.utils.plot_model(model, rankdir="LR", show_shapes=False)

print(5)

# use entire data set for gradient update if batch size is less than zero
if batch_size < 0:
    batch_size = y_train.shape[0]

print("batch size ", batch_size)

history = model.fit(x_train,y_train, validation_data=(x_test, y_test),
                    verbose=0,  epochs=epochs, shuffle=True, batch_size=batch_size)


print(6)

y_prediction_train = model.predict(x_train)
print(7)
G0_error_train = metrics.mean_squared_error(y_train[:,0], y_prediction_train[:,0])
print(8)
a_error_train = metrics.mean_squared_error(y_train[:,1], y_prediction_train[:,1])
print(9)
error_train = (G0_error_train + a_error_train) * 0.5
print(10)
y_prediction_test = model.predict(x_test)
G0_error_test = metrics.mean_squared_error(y_test[:,0], y_prediction_test[:,0])
a_error_test = metrics.mean_squared_error(y_test[:,1], y_prediction_test[:,1])
error_test = (G0_error_test + a_error_test) * 0.5
print(f"RESULT: {error_test}")