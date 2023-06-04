import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout


DATA_DIR = 'data'
EXPERT_DATA_DIR = os.path.join(DATA_DIR, 'expert_data')
EXPERT_STATES_DIR = os.path.join(EXPERT_DATA_DIR, 'states')
PREPROCESSED_STATES_DIR=os.path.join(EXPERT_DATA_DIR, 'preprocessed_states')


# Load the data
X = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_states.npy"))
y = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_actions.npy"))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()



# Train the model
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Save the model
model.save('self_driving_car_model.h5')