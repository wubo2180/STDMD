# STDMD: Spatio-Temporal Decoupled Meta-learning for Dynamic Graph Node Prediction

This repository provides a reference implementation for the method described in the patent:

**“A Spatio-Temporal Decoupled Meta-learning Method for Dynamic Graph Node Attribute Prediction”**  
（中文：一种基于时空解耦元学习的动态图节点属性预测方法）

The code is mainly used for **research validation and experimental reproduction**, supporting the development and verification of the proposed technical solution.

---

## 📌 Overview

Dynamic graphs are widely used to model complex systems such as traffic networks, social networks, and sensor networks, where node attributes evolve over time and are strongly coupled with dynamic structural changes.

This project focuses on a **spatio-temporal decoupled meta-learning framework**, which:

- Decouples node attribute prediction into **spatial structure learning tasks** and **temporal evolution learning tasks**
- Introduces a **meta-learning mechanism** with support/query task construction
- Enables fast adaptation to dynamic graph structure changes and evolving node attributes

The implementation follows a **process-oriented design**, emphasizing engineering feasibility rather than mathematical formulation.

---

## ✨ Key Features

- **Spatio-temporal task decoupling**  
  Separates spatial dependency modeling from temporal evolution modeling to reduce interference.

- **Meta-learning based adaptation**  
  Uses task-level support and query sets to enable fast parameter adaptation under dynamic graph changes.

- **Dynamic graph embedding**  
  Combines historical node states with current graph information for robust node representation learning.

- **General applicability**  
  Suitable for various dynamic graph scenarios, such as traffic flow prediction, epidemic modeling, and social network analysis.

---

## 🧠 Method Framework

The overall workflow includes:

1. Dynamic graph data collection and time slicing  
2. Spatio-temporal task decoupling  
3. Construction of support and query sets for spatial and temporal tasks  
4. Dynamic graph node embedding with historical state fusion  
5. Spatial-level adaptive update  
6. Temporal-level adaptive update  
7. Joint meta-update  
8. Node attribute prediction output  

> Note:  
> This repository focuses on **process-level implementation**.  
> Detailed algorithmic formulations are intentionally abstracted to align with patent protection requirements.

---

## 🗂 Repository Structure

```text
STDMD
├── .gitignore          # Git ignore rules for unnecessary files and directories
├── LICENSE             # License file for this project
├── README.md           # Project overview and usage instructions
├── requirements.txt    # Python dependencies required to run the project
│
├── main.py             # Main entry point: controls training and evaluation workflow
├── model.py            # Core model definition for spatio-temporal decoupled meta-learning
├── layers.py           # Basic neural network and graph-related layers
├── dataset.py          # Data loading and preprocessing for dynamic graph datasets
├── utils.py            # Utility functions (metrics, logging, helper methods)
│
├── baselines.py        # Baseline models for comparison experiments
├── basetest.py         # Evaluation and testing scripts for baseline methods
