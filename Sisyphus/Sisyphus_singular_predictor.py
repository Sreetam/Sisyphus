# Recurrent Neural Network

#Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as skp
import time

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# dr is the divide ratio. It divides the dataset into dr:1 ratio
dr = 19

# f_param is the timestep function
f_param = 30

# Parameters of the RNN
num_of_middle_layers = 1
middle_layer_units = 30
middle_layer_dropouts = 0.2
RNN_epochs = 40

# The choice variable pos
pos = 4

# Importing the training set
dataset = pd.read_csv('BSE_indices_data.csv')#enter name of data file here

# Dropping the effective date. Its not needed for now
dataset = dataset.drop('Effective date ', 1)
k = len(dataset)

cols = dataset.columns.tolist()

# Now I will drop the base values. I want to work with real values
for i in cols :
    if i.find('(Base value)') != -1 :
        dataset = dataset.drop(i, 1)

cols = dataset.columns.tolist()

# Removing the nan or 0 values
for i in cols :
    for j in range(1,len(dataset)) :
        if dataset[i][j]==0 or np.isnan(dataset[i][j]) :
            dataset[i][j]=dataset[i][j-1]

print("\n\n!Achtung!")
print("The following are the choices of indexes:")
print(*cols, sep = "\n")
print("The index chosen is: ")
print(cols[pos])
print("If not, adjust the variable pos correctly")
print("pos should belong to [0, " + str(len(cols)) + ")")
print("\n\n")
time.sleep(5)

# Dividing the dataset into training set and testing set
dl = (int)((dr/(dr+1))*k)
training_set = dataset.iloc[0:dl, pos:(pos+1)].values
testing_set = dataset.iloc[(dl-1):, pos:(pos+1)].values

# Feature Scaling
sc = skp.MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with f_param timesteps and 1 output
X_train = []
y_train = []
for i in range(f_param, len(training_set)):
    X_train.append(training_set_scaled[i-f_param:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#                              no of rows........timestep.........features

#Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation. Its the input layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the middle layers
for i in range(0, num_of_middle_layers): 
    regressor.add(LSTM(units = middle_layer_units, return_sequences = True))
    regressor.add(Dropout(middle_layer_dropouts))

# Adding the penultimate LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# instead of adam we could also have used rmsprop as the optimizer
# This is a regression problem so we use mean_squared_error
# In case of classification we could have used binary cross entropy or categorical cross entropy
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = RNN_epochs, batch_size = 32)

# Getting the test index values
dataset_total = dataset.iloc[0:, pos:(pos+1)]
cdataset_total = dataset_total.columns.tolist()

dataset_total = dataset_total[cdataset_total[0]]

# Preparing the inputs
inputs = dataset_total[len(dataset_total) - len(testing_set) - f_param:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(f_param, (len(testing_set) + f_param)):
    X_test.append(inputs[i-f_param:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# exporting data
ts = np.reshape(testing_set, (1, len(testing_set)))
ps = np.reshape(predicted_stock_price, (1, len(predicted_stock_price)))
df = pd.DataFrame(list(zip(list(ts[0][0:]), list(ps[0][0:]))), columns=['real', 'prediction'])
df.to_csv("data/Prediction:_" + str(cols[pos]) + ".csv")

# Visualising the results
real_stock_price = testing_set
plt.plot(real_stock_price, color = 'red', label = 'Real Index')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Index')
plt.title('Prediction: ' + str(cols[pos]))
plt.xlabel('Time')
plt.ylabel('Index')
plt.legend()
plt.savefig('data/Magnified_Prediction_Chart:_' + str(cols[pos]) + '.png')
#plt.show()
plt.plot([0],[0])
plt.plot(real_stock_price, color = 'red', label = 'Real Index')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Index')
plt.title('Prediction: ' + str(cols[pos]))
plt.xlabel('Time')
plt.ylabel('Index')
plt.legend()
plt.savefig('data/Real_Prediction_Chart:_' + str(cols[pos]) + '.png')
#plt.show()
