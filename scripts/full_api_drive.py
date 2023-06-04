import airsim
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('self_driving_car_model.h5')

client = airsim.CarClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
car_controls = airsim.CarControls()

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
camera_img_size = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)

while True:
    
    img_resp = client.simGetImages([camera_img_size])[0]

    img = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8).reshape(img_resp.height,img_resp.width, 3)

    img_resized = cv2.resize(img, (84, 84))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    #print(img_resized.shape)

    controls=model.predict(img_resized)[0]

    '''
    car_controls.throttle=controls[0].item()
    car_controls.steering=controls[1].item()
    car_controls.brake=controls[2].item()
    car_controls.handbrake=bool(controls[3])
    car_controls.is_manual_gear=bool(controls[4])
    car_controls.manual_gear=int(controls[5])
    car_controls.gear_immediate=bool(controls[6])
    '''
    car_controls.throttle=0.65
    car_controls.steering=controls[0].item()
    car_controls.brake=0
    car_controls.handbrake=False
    car_controls.is_manual_gear=False
    car_controls.manual_gear=2
    car_controls.gear_immediate=False

    client.setCarControls(car_controls)
    #print(car_controls.throttle,car_controls.steering,car_controls.brake,car_controls.handbrake,car_controls.is_manual_gear,car_controls.manual_gear,car_controls.gear_immediate)