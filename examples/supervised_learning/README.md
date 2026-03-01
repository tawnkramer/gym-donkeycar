# Supervised Learning

## Train the donkey by example. 

* Download the simulator
* Validate your setup
* Drive in the simulator
* Save data to a log
* Train a model
* Evaluate the model with the simulator


### Download the simulator

Download a [simulator binary](https://github.com/tawnkramer/gym-donkeycar/releases) for your platform.

Extract the zip file. Double click on the executable.

### Validate your setup

You may wish to try the sample model to validate your setup.

```python evaluate.py --model=models/example_model.h5```

Start the "Generated Track" environment. Click the "NN Control over Network" button.

### Drive in the simulator

Now you can try training your own model from your own data. 

In the simulator, press "Exit" to go the Menu screen.

### Save data to a log

Click the "log dir" button and specify a log dir to save data. You may choose the `gym-donkeycar/examples/supervised_learning/log` folder for example.

Select your prefered environment.

Click on "Auto Drive w Rec" or "Joystick/Keyboard w Rec". Record about 10K samples. View the lower left corner to see the log count.

Click "Stop" when done.

### Train a model

```python train.py --inputs=<path to log>*.jpg --model=<path to model>```

### Evaluate the model

```python evaluate.py --model=<path to model>```

Start the simulator. Select the environment. Click "NN Control over Network". The simulator will connect to your evaluator python process and send images to it. The evaluator will send control data back to the simulator.


