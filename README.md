AirSim RL and IL Project
This project utilizes AirSim, CNN, and reinforcement learning (RL) and imitation learning (IL) to train an agent to drive a car in a simulated environment. The project consists of several Python scripts and configuration files organized into the following structure:

lua
Copy code
project/
|-- data/
|   |-- expert_data/
|   |   |-- expert_0.npy
|   |   |-- expert_1.npy
|   |   |-- ...
|   |
|   |-- learner_data/
|   |   |-- learner_0.npy
|   |   |-- learner_1.npy
|   |   |-- ...
|   |
|   |-- preprocessed_data/
|       |-- expert_0_processed.npy
|       |-- expert_1_processed.npy
|       |-- ...
|
|-- models/
|   |-- cnn_model.py
|   |-- rl_model.py
|   |-- il_model.py
|
|-- scripts/
|   |-- collect_expert_data.py
|   |-- preprocess_data.py
|   |-- train_rl.py
|   |-- train_il.py
|   |-- evaluate_agent.py
|   |-- drive_car.py
|
|-- config/
|   |-- rl_config.yml
|   |-- il_config.yml
|
|-- logs/
|   |-- rl/
|   |-- il/
|
|-- README.md
Usage
Collect Expert Data
To collect expert data, run the collect_expert_data.py script. This script will start the AirSim simulator and allow you to control the car. During this time, the script will record your driving data and save it to the expert_data folder in the form of .npy files.

bash
Copy code
python scripts/collect_expert_data.py
Preprocess Data
To preprocess the collected expert data, run the preprocess_data.py script. This script will load the .npy files in the expert_data folder, preprocess the data, and save the preprocessed data to the preprocessed_data folder.

bash
Copy code
python scripts/preprocess_data.py
Train RL
To train the RL model, run the train_rl.py script. This script will load the preprocessed data, train the RL model, and save the trained model to the models folder. The script will also log the training progress to the logs/rl folder.

bash
Copy code
python scripts/train_rl.py
Train IL
To train the IL model, run the train_il.py script. This script will load the preprocessed data, train the IL model, and save the trained model to the models folder. The script will also log the training progress to the logs/il folder.

bash
Copy code
python scripts/train_il.py
Evaluate Agent
To evaluate the agent's performance, run the evaluate_agent.py script. This script will load the trained RL and IL models, start the AirSim simulator, and evaluate the agent's performance. The script will output the average reward over multiple episodes.

bash
Copy code
python scripts/evaluate_agent.py
Drive Car
To drive the car using the trained RL and IL models, run the drive_car.py script. This script will load the trained models and start the AirSim simulator. During this time, the script will use the models to predict the actions for the agent.

bash
Copy code
python scripts/drive_car.py
Configuration
The rl_config.yml and il_config.yml files contain the configuration settings for the RL and IL models, respectively. You can modify these files to change the hyperparameters and settings for the models.


!!!!
All data are wiped from project to reduce the size
