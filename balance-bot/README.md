## Balance Bot
This is a self-balancing 2 wheeled robot trained using Reinforcement Learning. The observation spaces, action spaces, reward computation and action steps are defined in the environment. The OpenAI baseline implementation of DeepQ Learning was used to find the optimal policy for making the robot balance on its own. 

### Installation
Download this folder as it is on your PC. Then open the folder on your terminal and run the following commands. 
<p align="center">
 <img  width="850" height="150" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/balance-bot/installation.png">
</p>

### Environment and Training
The model training using the DeepQ Network has been done in the balancebot_learning_deepq.py file. The environment used is available in the envs folder as balancebot_env.py file. The xml file of the 2 wheeled robot has also been given.

<p align="center">
 <img  width="300" height="400" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/balance-bot/balance-bot.png">
</p>

While training the model, you require tensorflow==1.14.0 version or any other tensorflow version which is compatible with OpenAI gym baselines.  

This has been included, because future work may include this 2-wheeled robot trained to move around and map the environment in-place of husky.
