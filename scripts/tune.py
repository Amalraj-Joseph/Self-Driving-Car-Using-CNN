import os
import numpy as np
from tensorflow.keras.models import load_model


DATA_DIR = 'data'
EXPERT_DATA_DIR = os.path.join(DATA_DIR, 'expert_data')
EXPERT_STATES_DIR = os.path.join(EXPERT_DATA_DIR, 'states')
PREPROCESSED_STATES_DIR=os.path.join(EXPERT_DATA_DIR, 'preprocessed_states')


# Load the data
X = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_states.npy"))
y = np.load(os.path.join(PREPROCESSED_STATES_DIR,"preprocessed_actions.npy"))


model = load_model('self_driving_car_model.h5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Save the model
model.save('self_driving_car_model.h5')