import os
import cv2
import numpy as np
import airsim



aug_dir="data/expert_data/augmented_data"
pre_states=os.path.join(aug_dir,"states")
aug_states=os.path.join(aug_dir,"augmented_states")

images=os.path.join(aug_dir,"images")
original=os.path.join(images,"original")
original=os.path.join(images,"mirror")

files = sorted(os.listdir(pre_states))

X = np.load(os.path.join(pre_states,files[0]))
Xmirror=np.load(os.path.join(aug_states,files[0]))

print(X.shape)
print(Xmirror.shape)