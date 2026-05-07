# MLP Companion Notebooks

These notebooks accompany `lecture_notes/16_mlp.pdf`.

## Proposed Sequence

1. `01_xor_and_linear_limitations.ipynb`: XOR, concentric circles, linear models versus MLPs.
2. `02_mlp_architecture_and_parameters.ipynb`: layer sizes, matrix shapes, and parameter counts.
3. `03_activation_functions.ipynb`: sigmoid, tanh, ReLU, Leaky ReLU, derivatives, saturation, and vanishing gradients.
4. `04_forward_pass_from_scratch.ipynb`: manual 2-2-1 forward pass from the lecture notes.
5. `05_losses_and_output_layers.ipynb`: output activations and loss functions for regression, binary, and multi-class tasks.
6. `06_backpropagation_from_scratch.ipynb`: manual backward pass and autograd check.
7. `07_training_loop_pytorch.ipynb`: PyTorch training loop with DataLoader and validation.
8. `08_optimization_regularization_debugging.ipynb`: initialization, optimizers, regularization, and debugging checks.
9. `09_mlp_sklearn_pipeline.ipynb`: scikit-learn MLP pipelines, scaling, early stopping, and tuning.
10. `10_case_study_tabular_mlp.ipynb`: applied tabular MLP case study.

## Implemented So Far

- `01_xor_and_linear_limitations.ipynb`
- `02_mlp_architecture_and_parameters.ipynb`
- `03_activation_functions.ipynb`
- `04_forward_pass_from_scratch.ipynb`
- `05_losses_and_output_layers.ipynb`
- `06_backpropagation_from_scratch.ipynb`
- `07_training_loop_pytorch.ipynb`
- `08_optimization_regularization_debugging.ipynb`

## Existing Notebooks To Incorporate Later

- `backprop.ipynb`: should be replaced by or folded into `06_backpropagation_from_scratch.ipynb`.
- `mlp_adult.ipynb`: can be converted into an optional applied notebook if network access is acceptable, or replaced by a local dataset.
- `MLP_for_cred_dataset.ipynb`: current local-file case study candidate for item 10.

## Dependency Note

- `07_training_loop_pytorch.ipynb` requires PyTorch for the training loop cells. If PyTorch is missing, the notebook remains executable and skips those cells with an explanatory message.

## Coverage Map

- Hidden layers and nonlinear decision boundaries: notebook 01.
- Architecture and parameter counting: notebook 02.
- Activation functions: notebook 03.
- Forward pass by hand: notebook 04.
- Loss/output compatibility: notebook 05.
- Backpropagation and chain rule: notebook 06.
- PyTorch implementation: notebook 07.
- Optimization, regularization, and debugging: notebook 08.
- scikit-learn implementation: notebook 09.
- Applied tabular workflow: notebook 10.
