# CS-370 Current/Emerging Trends

This repository showcases my work with artificial intelligence, neural networks, reinforcement learning, and deep Q-learning completed during CS-370 Current/Emerging Trends.

## Featured Project: Treasure Hunt Intelligent Agent

The main project in this repository is a reinforcement learning agent designed to navigate an 8x8 maze and locate a treasure.

Instead of programming a fixed route, I implemented a deep Q-learning approach that allowed the pirate agent to learn through repeated interaction with the environment.

### How It Works

* The 8x8 maze is represented using 64 state values.
* The neural network predicts Q-values for four possible actions: left, up, right, and down.
* Exploration allows the agent to test different actions during training.
* Exploitation allows the agent to use what it has already learned.
* Experience replay stores previous interactions so they can be reused during training.
* A target network helps stabilize the learning process.
* Rewards and penalties guide the agent toward successful behavior.

The final trained agent reached a rolling win rate of **1.000** and successfully completed the maze from every valid starting position.

## Skills Demonstrated

* Python
* Artificial Intelligence
* Machine Learning
* Neural Networks
* Reinforcement Learning
* Deep Q-Learning
* TensorFlow and Keras
* Algorithm Analysis
* Model Training and Evaluation
* Debugging and Problem Solving

## What I Learned

This project helped me understand the difference between programming a solution and creating an agent that can learn a solution. I approached the development process much like troubleshooting a real-world problem: test an idea, examine the results, make corrections, and continue improving the system until it performs reliably.

The project also strengthened my understanding of how exploration, exploitation, rewards, neural networks, and previous experience work together to produce learned behavior.

## Repository Contents

Project files and supporting documentation for the Treasure Hunt intelligent agent will be organized within this repository.
