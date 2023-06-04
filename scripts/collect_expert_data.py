import airsim
import numpy as np
import os
from utils.clear_directory import clear_directory


# set the path to the directory where expert data will be saved
expert_data_dir = "./data/expert_data/manual_data"

states_dir=os.path.join(expert_data_dir,"states")
actions_dir=os.path.join(expert_data_dir,"actions")

#clear_directory(states_dir)
#clear_directory(actions_dir)

# set the number of expert demonstrations to collect
start=296
num_expert_demos = 300

# initialize AirSim client
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)

# set camera pose and image size for data collection
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
camera_img_size = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
# loop through and collect expert demonstrations
for i in range(start,num_expert_demos):

    print("Collecting expert demonstration {}...".format(i+1))
    # reset the car's position
    client.reset()

    expert_actions = []
    expert_states = []
    # set initial state
    state = client.getCarState()

    collision_info = client.simGetCollisionInfo()
    car_controls=client.getCarControls()

    while not car_controls.manual_gear==-1:

        # get camera image
        img_resp = client.simGetImages([camera_img_size])[0]

        img = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8).reshape(img_resp.height,img_resp.width, 3)
        # get car state
        
        expert_states.append(img)

        # get car controls
        car_controls=client.getCarControls()
        # add expert data to list

        
        #expert_actions.append((car_controls.throttle,car_controls.steering,car_controls.brake,int(car_controls.handbrake),int(car_controls.is_manual_gear),car_controls.manual_gear,int(car_controls.gear_immediate)))
        expert_actions.append((car_controls.steering))

        collision_info = client.simGetCollisionInfo()
    
    expert_actions_file = os.path.join(actions_dir, "expert_{}.npy".format(i))
    np.save(expert_actions_file, np.array(expert_actions))
    expert_states_file = os.path.join(states_dir, "expert_{}.npy".format(i))
    np.save(expert_states_file, np.array(expert_states))

