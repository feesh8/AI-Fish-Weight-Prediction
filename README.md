# Fish Weight Predictor

## Overview

This project implements a predictive model to estimate the weight of a fish based on various physical measurements such as length, height, and width. The predictive model is developed using Python and leverages machine learning techniques.

## Dataset

The dataset `Fish.csv` contains the following columns:

- Species: Species of the fish.
- Weight: Weight of the fish (target variable).
- Length1, Length2, Length3: Length measurements of the fish.
- Height: Height of the fish.
- Width: Width of the fish.

## Usage

- Ensure you have Python and Jupyter Notebook installed on your system.
- Clone this repository: git clone https://github.com/feesh8/fish-weight-predictor.git
- Navigate to the project directory: cd fish-weight-predictor
- Run the Jupyter notebook: jupyter notebook `projet.ipynb`
- Follow the instructions in the notebook to explore the data, train the model, and make predictions.

## Result of the model study

| Model                            | Real MSE | Empirical MSE | Real R2 | Empirical R2 |
| -------------------------------- | -------- | ------------- | ------- | ------------ |
| Simple                           | 0.0051   | 0.0056        | 0.88    | 0.88         |
| Lasso Regulation                 | 0.0052   | 0.0057        | 0.88    | 0.88         |
| Polynomial expansion of degree 2 | 0.0009   | 0.0008        | 0.98    | 0.98         |
