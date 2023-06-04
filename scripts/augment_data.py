import os
import numpy as np


aug_dir="data/expert_data/augmented_data"
pre_actions=os.path.join(aug_dir,"actions")
pre_states=os.path.join(aug_dir,"states")
aug_actions=os.path.join(aug_dir,"augmented_actions")
aug_states=os.path.join(aug_dir,"augmented_states")



files = sorted(os.listdir(pre_states))

i=351

for file in files:

    X = np.load(os.path.join(pre_states,file))

    Xmirror = np.empty_like(X)


    for j in range(len(X)):
        
        mirrored_image = np.flip(X[j], axis=1)

        Xmirror[j]=mirrored_image
    np.save(os.path.join(aug_states,"expert_{}".format(i)),Xmirror)
    i += 1



# Define the mapping between array1 values and array2 values
mapping = {0.5: -0.5, -0.5: 0.5, 0.0: 0.0}

files = sorted(os.listdir(pre_actions))

i=351

for file in files:

    X = np.load(os.path.join(pre_actions,file))

    Xmirror = np.array([mapping[val] for val in X])

    np.save(os.path.join(aug_actions,"expert_{}".format(i)),Xmirror)
    i += 1



