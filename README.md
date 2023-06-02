# Experiments of Reinforcement Learning in Rust
Experiments to learn about Reinforcement Learning

## About

Four environments were implemented:
 - Blackjack
 - Frozen Lake
 - Cliff Walking
 - Taxi

The environment's implementation was based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) library for Python.

About the features implemented:
 - Only tabular policies: Basic Policy and Double Policy (for double learning)
 - Two action selections strategies: ε-greedy with uniform distribuition and Upper Confidence Bound (UCB).
 - Four policy update strategies: One-Step Sarsa, One-Step QLearning, One-Step Expected Sarsa, Sarsa with Eligibility Traces and QLearning with Eligibility Traces.
 - Three charts for visualizations about the agent's training: Episode's length, Rewards and Training error.
 - Visualizations for the states of each environment.
 - A small CLI program to change the parameters used on the training and generate the charts.

## Install
First install Rust:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Then compile the source code:

`cargo build -r`

## Usage of the CLI program
To run the experiments use:

`./target/release/reinforcement_learning 0`

Where '0' is the identifier for the BlackJack env, to see the other options use:

`./target/release/reinforcement_learning --help`
