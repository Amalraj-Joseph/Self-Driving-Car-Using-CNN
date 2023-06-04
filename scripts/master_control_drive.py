import airsim
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import keyboard

# connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
api_mode=True
client.reset()

car_controls = airsim.CarControls()

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
camera_img_size = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)

#load the model
model = load_model('self_driving_car_model.h5')

car_controls.throttle=0.5
car_controls.steering=0.0
car_controls.brake=0
car_controls.handbrake=False
car_controls.is_manual_gear=False
car_controls.manual_gear=1
car_controls.gear_immediate=False

client.setCarControls(car_controls)

while True:

    if keyboard.is_pressed('d'):

        client.enableApiControl(False)
    else:
        
        client.enableApiControl(True)
        img_resp = client.simGetImages([camera_img_size])[0]
        img = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8).reshape(img_resp.height,img_resp.width, 3)

        img_resized = cv2.resize(img, (84, 84))
        img_resized = img_resized.astype('float32') / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
            
        controls=model.predict(img_resized)[0]

        car_controls.steering=controls[0].item()
        client.setCarControls(car_controls)