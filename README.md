# E2E_learning_RC
Autonomous driving of RC car using end-to-end learning

## Abstract
This research thesis implements an autonomous remote-control car. It uses the images captured by one front facing camera as an input to the machine learning neural network and predicts the steering values of the car in real time. The throttle is a function of the steering value in this autonomous operation. The project has been carried out using several hardware components like the Raspberry Pi, Arduino, and a RC car along with its essential peripherals. The project involves the data collection phase where the steering values and the corresponding images were captured as the RC car is manually driven around a test track. This data is then processed and used to train a convolutional neural network. The steering values can then be predicted using the trained model. The predicted steering values will be used to determine the throttle in real time and subsequently drive the RC car autonomously around the complete test track as the final result.
![Overview](https://user-images.githubusercontent.com/57918108/133751903-36fc16c5-9c35-432a-b1d0-bc7d22de3c8e.JPG)
![RC car](https://user-images.githubusercontent.com/57918108/133752026-c4466b65-6d64-4647-829f-c19e57c6f06f.jpg)

## Steps
STEPS FOR DATA COLLECTION PHASE:
0.1 Turn on the Raspberry Pi and wait for about a minute for the OS to start. Then turn on the RC car switch. Then turn on the Remote control. Then connect to the VNC viewer to view the Pi. (this may take time when done the first time). Username : pi Password : raspberry
1. Use Geany tool and Open files : DataCollection_mode.py
csv_file_generator.py delete_recordings.py
2. Run the file DataCollection_mode.py. Ready to record. (RED LED lights up)
3. Press button on RC car to start recording session. (GREEN LED lights up)
4. Press button to stop trial. (RED LED lights up) Message on screen reads how many trials left in current session (configurable). Press button once again to start next trial. (GREEN LED lights up). All results of subsequent trails of same session will be stored in the same files(both cam images & readings).
5. You can terminate a session by : --- closing the terminal window --- ctrl + c --- completing all trials defined in a session.
6. Once terminated and you are happy with the trail run, run the file csv_file_generator.py Even if 1 trail of a session goes wrong. Please delete the session and start again.
7. Save the entire "cam_images" and "CSV_files" folders by copying them to your PC/Laptop. (Raspberry Pi file transfer).

STEPS FOR TRAINING PHASE:
1. Load all images in the "Dataset/images" folder. Transfer the "output.csv" file in the "Dataset" folder.
2. Use "Training.py" file to set training parameters. Run the file to begin training the model.
3. Once completed, the "Dataset/logs" folder will contain training logs for analysis on TensorBoard and "Dataset/models" folder will contain trained models from each epoch. Using the TensorBoard for analysis, select the best model with lowest epoch_loss. 

STEPS FOR AUTONOMOUS PHASE:
1. Load the trained model from the "Dataset/models" folder.
2. Run the Model converter code to generate a TFLite model. Save this new model in the Lite Models folder.
3. Open terminal window and run command “sudo pigpiod”. pigpiod is a utility which launches the pigpio library as a daemon.
4. Run the file “Autonomous_mode.py”. Please wait as the libraries take time to load. Once ready, the next set of instructions will be displayed on the terminal window.
5. As per the instructions, follow the steps to calibrate the ECS. Check for relay switch to be in closed position for PWM signals to be in a closed path.
6. Once calibration is completed, push the button on the RC car to begin autonomous mode.
7. To stop the car, press the wireless relay button on the remote.
8. Press the push button on the RC car to terminate the autonomous driving session.

## Credits
E2E learning implementation on Carla simulator by [tuannguyen](https://github.com/m4tice)
