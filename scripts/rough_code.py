import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout


DATA_DIR = 'data'
EXPERT_DATA_DIR = os.path.join(DATA_DIR, 'expert_data')
EXPERT_STATES_DIR = os.path.join(EXPERT_DATA_DIR, 'states')
PREPROCESSED_STATES_DIR=os.path.join(EXPERT_DATA_DIR, 'preprocessed_states')


# Load the data
X = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_states.npy"))
y = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_actions.npy"))

model = load_model('trained_model.h5')

loss, accuracy = model.evaluate(X, y)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))