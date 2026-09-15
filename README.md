# Inverted Pendulum

This project explores a classic control problem: keeping an inverted pendulum upright using a real mechanical setup and a small embedded controller. The goal is to compare traditional control methods, such as PID loops, with a reinforcement learning (RL) approach based on neural networks.

The physical system is built from 3D-printed parts and low-cost electronics. It is intentionally simple enough to be understandable, but rich enough to show the difference between a hand-tuned controller and a learned policy that adapts to the system over time.

## What is an inverted pendulum?

An inverted pendulum is a cart-and-pole system where the pendulum is balanced upside down. The challenge is to continuously move the base in the right direction so the pendulum does not fall.

This is a classic benchmark in control engineering and robotics because it is:

- easy to understand visually,
- hard enough to be interesting,
- and representative of many real-world balancing and stabilization problems.

In everyday terms, it is similar to balancing a broom on your hand or keeping a robot upright while it moves. The system is unstable by design, so the controller must react quickly and precisely.

## Why use reinforcement learning?

A traditional controller, like PID, follows a fixed set of rules. It is effective when the dynamics are well understood and the system behaves predictably.

Reinforcement learning takes a different approach. Instead of writing every rule by hand, the system learns through trial and error. It tries actions, sees the result, and gradually improves its policy to keep the pendulum balanced for longer periods.

This makes RL especially interesting for systems where the dynamics are nonlinear, noisy, or difficult to model precisely. In this project, the learned controller is compared against classical control strategies to see how well it performs in practice.

## Project goals

The project aims to:

- build a simple but real inverted pendulum prototype,
- test classical control methods on hardware,
- train a neural network policy using reinforcement learning,
- compare the performance of learned control against conventional approaches.

The result is a practical platform to study the gap between model-based engineering and data-driven control.

## Control experiments

### PID velocity control

A simple controller regulates the base velocity to keep the pendulum upright. This is a good baseline and shows how well a compact, transparent control law can perform.

Video:
[*PID velocity control*](https://youtube.com/shorts/lt8KjExcmE4?feature=share)

### PID double loop torque control

This version uses a more advanced control structure with separate loops for the position and torque behavior. It gives a stronger classical control baseline and highlights how tuned analytical control can handle the unstable dynamics.

Video:
[*PID double loop torque control*](https://youtube.com/shorts/xpPwe5tMBqU?feature=share)

### Neural network control with reinforcement learning

The RL policy uses observations from the system and learns a balancing strategy without requiring a manually designed control law for every operating condition.

Video:
[*Neural network control with reinforcement learning*](https://youtube.com/shorts/eyTFQ6OgbI4?feature=share)

## Video and model files

The project includes trained models and demonstration videos in the `model/` folder. These files are useful for:

- replaying the learned behavior,
- comparing control performance across methods,
- reviewing the trained policy in action.

If you want to see the system in motion before reading the code, start with the videos in `model/` and then look through the training and simulation files.

## Electronics used

- ESP32 DevKit
- SimpleFOC Mini
- AS5600 magnetic encoder
- Brushless DC motor: MiToot 2206/100T

## Hardware and design

The pendulum is built from 3D-printed parts and inexpensive components, making it a compact and accessible platform for experimentation. The combination of printed mechanics, encoder feedback, and a motor driver gives a realistic closed-loop system without requiring expensive industrial hardware.

This makes the design easy to reproduce and ideal for educational projects and embedded control experiments.

## Website and more information

For more projects, experiments, and technical notes, visit my website:

- [My website](https://aizej.com)
- [More robotics and control projects](https://aizej.com)

## Summary

This project is a simple but powerful demonstration of how classical control and reinforcement learning can be compared on the same physical problem. It is both a mechanical build and a control experiment, making it useful for learning, prototyping, and understanding how balancing systems behave in real life.

If you are new to RL, think of it as teaching the system by experience rather than by hard-coded instructions. The result is a controller that can learn to balance through repeated interaction with the environment.
