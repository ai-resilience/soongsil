 Backdoor Attack & Continual Learning Framework
 Overview

This repository provides a modular experimental framework for studying backdoor attacks and continual learning in deep neural networks.
It enables researchers to simulate various backdoor injection strategies, evaluate model robustness, and test defense or detection mechanisms in continual learning environments.

 Directory Structure
.
├── AccumulativeAttack/          # Implementation of accumulative (progressive) backdoor attacks
├── Agent/                       # Attack analysis utilities (recently removed analyze_attack.py)
├── Brainwash/                   # Brainwash-style backdoor attack implementation
├── avalanche/                   # Integration with Avalanche continual learning library
├── checkpoints_base_bn/         # Model checkpoints and BatchNorm-based configurations
├── chroma/                      # Feature or data storage module (color or embedding handling)
├── continual-learning-baselines/ # Baseline continual learning methods
├── labelflipping/               # Label flipping attack and detection modules
├── .gitignore                   # Git ignore rules
├── large_files.txt              # List of excluded large files

 Key Features
Category	Description
Backdoor Attacks	Implements multiple backdoor attack types such as AccumulativeAttack, Brainwash, and LabelFlipping.
Detection & Defense	Includes a detection module under labelflipping/; easily extendable for new detection algorithms.
Continual Learning	Supports continual learning experiments using avalanche and continual-learning-baselines.
Model Checkpointing	Stores model weights and configurations in checkpoints_base_bn.
Modularity	Each attack or defense method is isolated as an independent module for flexible experimentation.
