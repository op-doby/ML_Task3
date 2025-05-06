# Assignment 3: Feedforward Neural Networks using PyTorch

This repository contains the implementation of various feedforward neural networks for MNIST classification using PyTorch, as required for Assignment 3 of the Applied Deep Learning course.

## Tasks Implemented

1. **MNIST Classification**: Implementation of a basic two-layer fully connected neural network for MNIST classification.
2. **Mitigating Pseudorandomness**: Adaptation of the code to produce consistent results across runs with different seed numbers.
3. **Validation Dataset**: Splitting the MNIST dataset into train, validation, and test sets to find the best model based on validation performance.
4. **Grid Search**: Implementation of a grid search over hyperparameters (hidden size, batch size, learning rate) to find the optimal model configuration.
5. **Feature Analysis**: Visualization of hidden features using t-SNE to analyze the model's learned representations.

## How to Run

1. Make sure you have the required dependencies installed:
```
pip install torch torchvision matplotlib numpy scikit-learn
```

2. Run the main script:
```
python main.py
```

This will execute all tasks sequentially and save the results in the `results` directory.

## Results

The results of each task will be saved in the `results` directory, including:
- Error plots for training and testing
- Visualizations of misclassified images
- t-SNE visualizations of hidden features and raw inputs

## Report

The report.pdf file (to be created separately) will contain detailed analysis of the results, including:
- Training and testing error graphs
- Analysis of model robustness to seed selection
- Validation performance analysis
- Grid search results table
- Feature analysis with t-SNE visualizations

## Code Structure

- `main.py`: Contains the implementation of all tasks
- `results/`: Directory containing all generated plots and results
