# Contents
  [1.Introduction](#Introduction)

  2.About this repository folders and files
  
  [3.Experiment](#Experiment)
    
# Introduction
The freshness of the data is important in UAV aided data collections.
The freshness of the data depends of the UAV's flight trajectory.
To decide the trajectory, Deep Reinforcement Learning is a good method and useful 
# About this repository folders and files

| **Name**  |  **Description**  |
| --  |  --  |
| Environment1  |  Train the model and execute the trained model about environment1  |
| Environment2  |  Train the model and execute the trained model about environment2  |
| Latency  |  The latency of the Neural Network model trained in **Environment2**  |
| Combination  |  Control UAVs with trained model in **Environment2**  |
| Results1  |  Experiment results of **Environment1**  |
| Results2  |  Experiment results of **Environment2**  |
| Results3  |  Experiment results of **Latency**  |
| Results4  |  Experiment results of **Combination**  |
| grouph.ipynb  |  Experiment grouph of **Environment1 & Environment2**  |


# Experiment
1.Make the directory to your google drive.

2.Copy the Experiment file .py to google colab and the change the diretory_name in the .ipynb file.

3.Execute the google colab file.

!Becareful!
If you clone this repository to google colab, it doesn't work due to tensorflow version. You have to use tensorflow-version-2.2. Tensorflow Version 2.5 is mainly used in our repository.
