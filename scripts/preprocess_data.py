import os
import cv2
import numpy as np
import airsim
from utils.clear_directory import clear_directory

DATA_DIR = 'data'
temp=os.path.join(DATA_DIR, 'expert_data')
EXPERT_DATA_DIR = os.path.join(temp, 'specific_data')
EXPERT_STATES_DIR = os.path.join(EXPERT_DATA_DIR, 'states')
EXPERT_ACTIONS_DIR = os.path.join(EXPERT_DATA_DIR, 'actions')
#PREPROCESSED_STATES_DIR=os.path.join(EXPERT_DATA_DIR, 'preprocessed_states')
PREPROCESSED_STATES_DIR=os.path.join(temp, 'preprocessed_states')
#images_dir=os.path.join(EXPERT_DATA_DIR,"Images")

clear_directory(PREPROCESSED_STATES_DIR)

expert_data_files = sorted(os.listdir(EXPERT_STATES_DIR))

i=0

final_data=[]


for expert_data_file in expert_data_files:

    # Load the data
    X = np.load(os.path.join(EXPERT_STATES_DIR,expert_data_file))

    # Standardize the data
    X = X.astype('float32') / 255.0

    # Resize the images to (84, 84, 3)
    X_resized = []
    for img in X:
        img_resized = cv2.resize(img, (84, 84))
        X_resized.append(img_resized)
        final_data.append(img_resized)
    X_resized = np.array(X_resized)

    # Print the shape of the resized array
    print(X_resized.shape)

    #pre_processed_states_file = os.path.join(PREPROCESSED_STATES_DIR, "preprocessed_{}.npy".format(i))
    #np.save(pre_processed_states_file, np.array(X_resized))

    i += 1

actions=[]

expert_actions_files = sorted(os.listdir(EXPERT_ACTIONS_DIR))

for action_file in expert_actions_files:

    Y = np.load(os.path.join(EXPERT_ACTIONS_DIR,action_file))
    print(Y.shape)
    for element in Y:
        actions.append(element)

print(np.array(final_data).shape)
print(np.array(actions).shape)

final_states_file = os.path.join(PREPROCESSED_STATES_DIR, "preprocessed_states.npy")
np.save(final_states_file, np.array(final_data))

final_actions_file = os.path.join(PREPROCESSED_STATES_DIR, "preprocessed_actions.npy")
np.save(final_actions_file, np.array(actions))