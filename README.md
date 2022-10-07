# 4730-Project-I

4730 Machine Learning Fall 2022 Project I Repository

## Set up VENV

To set up a virtual environment, run the following command in the git directory:

```bash
python3 -m venv ./venv
```

## Installing Requirements

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

## Project Requirements

- A group of 2 to 3 students (3 or 4 for 4730)
- implement a CNN (not using preexisting network architecture or nulti-layer blocks)
- apply your model to MNIST dataset and compare results with your existing CNN implementation
- Write a shore report (4-6 pages) explainging your implementation and showing results
- You should keep the source from the project

## Notes on the dataset

- Turns out MNIST is not a site of datasets, it is a single dataset of images of handwritten digits.
- These images are 28x28 pixels, grayscale, and centered.

## Project Plan

### Phase 1: Planning

- Decide on a dataset and how we are going to implement it
- Take the dataset and implement an initial model using preexisting libraries (tensorflow, pytorch, etc.)

#### Results

- Using keras, we are able to import the dataset, and create a model that is able to classify the images with a >98% accuracy.
- This was done in a [jupyter notebook](phase_1.ipynb).

### Phase 2: Implementation

- Break up the project into smaller tasks, posting them to the issues tab
- We can have assigned or unassigned tasks
- **Make sure we are all properly using github branches etc**
- The easiest way to start will be to use a library like Keras or PyTorch, and test our implementation in a jupyter notebook

#### Tasks

**Note:** These tasks are subject to **heavy change**, this is simply a broken down feature list of the keras model. It does not make sense for us to implement this the same was keras does; there is a simpler, more direct way to hard code this model.

*Just pick tasks you are confident you can do and submit them on the issues tab, make sure you use branches*

- [ ] Need a function to convert each of the images into a numpy array
- [ ] Need a pooling function
- [ ] Need a convolution function
- [ ] Need a function to flatten the output of the convolution function
- [ ] Need to implement a dense layer, with a few different activation functions
  - [ ] relu
  - [ ] softmax
- [ ] Need to implement a loss function
  - [ ] Cross Entropy
- [ ] Need a fitting function that will train the model
- [ ] Need an opimization model, that will optimize the weights of the model
  - [ ] Adam