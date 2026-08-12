# Onboarding Exercise

_Originally authored by Adacus Green '27 with edits by John MacCormick, summer 2026._

In this module, you will learn how to train your first large language model and test its accuracy, and then you will learn how to test a model's ability to generalize to unseen data. The goal is to give you a hands-on experience with the process of training and evaluating a transformer model.

## Prerequisites
We assume you already have some familiarity with Python, command line interfaces such as bash or PowerShell, git, GitHub, and virtual environments. If you are not familiar with these topics, please spend a few days completing online tutorials.

Assuming you have this prerequisite knowledge, you should clone this repo, create a virtual environment, and install the required packages (e.g., `pip install -r requirements.txt`). A GPU is _not_ required for this module. If you have a GPU and you want to use it, check your CUDA driver version and install the corresponding version of the `torch` package.


## First Experiment: Train a memorization model

**Background:** In this experiment, you will train a model to learn how to capitalize short strings. The model will be trained on a small dataset of input-output pairs, where the input is a lowercase string and the output is the same string with all letters capitalized. The goal is to see how well the model can memorize this mapping.

### 1. Generate the data.
The training data for our transformer model will consist of a text file containing input-output pairs. Each line of the file will contain a lowercase string and its corresponding capitalized string, separated by an equals sign (`=`). For example, the first few lines of the file might look like this:
```
vfl=VFL
j=J
mg=MG
t=T
mgg=MGG
```
We create this training data using the `make_inputs_capital.py` script. First, run the following command in your terminal:
```
python make_inputs_capital.py 3 26 1000
```
The meanings of the three arguments are as follows:
- 3: The maximum length of the input string.
- 26: The character set size (utilizing the slice of the lowercase English alphabet from a to z).
- 1000: The number of dataset lines to generate.

The output will be stored in `inputs/capital.txt`. Open this file to observe the generated data. 


### 2. Prepare the data for training.
To transform the raw text of `inputs/capital.txt` into binary tokens and a vocabulary mapping required by the transformer, run the preparation script:
```
python prepare_inputs.py inputs/capital.txt
```
This creates three files in the `data` directory: `train.bin`, `val.bin`, `meta.pkl`. The files use a binary data format, so there is no point in inspecting them. However, it is useful to understand their contents:
- `train.bin`: Contains the training data in binary format.
- `val.bin`: Contains the validation data in binary format. Validation data is used to estimate the accuracy of the model on inputs that it has not seen before.
- `meta.pkl`: Contains the vocabulary mapping, which is a dictionary that maps each character to a unique integer index. This mapping is used to convert characters to tokens and vice versa.

By default, the `prepare.py` script splits the input data into a training set containing 90% of the data and a validation set containing the remaining 10%.


### 3. Train the model.
Now that the tokens are prepped, you can kick off the training routine. Run:
```
cd ..
python train_completions.py config/config_1char.py
```
By default, the model will run for 100 epochs (complete passes through the data) to learn the underlying sequence pattern. This may take about two minutes on a standard laptop.
- Train Loss: Represents how well the model is fitting the data it is actively studying.
- Val Loss: Represents how well the model generalizes to unseen validation data.
### 4. Test for accuracy.
Once training concludes, a model checkpoint named `completion_model.pth` will be saved inside the `out_1char/` directory. To evaluate its structural accuracy against your generated text, run:
```
python generate.py inputs/capital.txt
```
At the bottom, it will output the accuracy, split between total processed, total correct, and a final accuracy.

You can also test individual input strings, for example using `python generate_one.py rg` to see if the model correctly outputs `RG`. You can test any string of length 1-3 using this method.

## Second Experiment: Out-of-Distribution Generalization
How does an AI learn to solve problems it was never explicitly shown? In this experiment, you will test a model's ability to achieve true mathematical generalization, but this time, you'll need to apply the syntax you learned in the first module.
### 1. The Challenge: Generate a Sparse Math Dataset
Your goal is to generate 3,000 lines of addition problems modulo 100 using the make_inputs_add.py script.
Because a complete $100 \times 100$ addition table contains 10,000 total permutations, your model will only see 30% of the possible data during training.
Your Task: Note that `make_inputs_add.py` takes arguments `V` and `N`, where `V` is the modulus and `N` is the number of lines generated. Using this, create and run a command to generate `inputs/add.txt` based on the data presented earlier.
### 2. Prepare and Train the Data
Now, prepare your newly generated `inputs/add.txt` file for training and kick off the training routine just like you did in the first experiment.
Pro-Tip: Mathematical patterns take longer to learn than simple memorization. Before running the training script, open config_1char.py and locate the epochs variable, and increase it (e.g., set epochs = 200 or higher) to give the network enough time to discover the underlying arithmetic logic.
### 3. Exhaustive Evaluation
While you can test accuracy using generate.py on your input file, we want to see if the model actually understands addition globally. Try a few single test cases using `python generate_one.py`. We can test the model's conceptual understanding by sweeping every single possible combination from $0+0$ to $99+99$. To do this, use:
```
python generate_all.py
```
Note that the accuracy is higher than 30%. That is because the model is learning an underlying pattern and not just memorizing the data.


## Running the Test Suite

The project includes a small, refactor-focused pytest suite with fast and integration layers. Practice running the tests now so that you can use them later as you make further changes to this code base.

### Fast tests (recommended during active refactoring)
Runs CLI smoke checks and tiny data-logic checks only:

```
./venv/Scripts/python.exe -m pytest -q -m "not integration"
```

### Integration tests (end-to-end tiny pipeline)
Runs only integration checks (including tiny training/inference flow):

```
./venv/Scripts/python.exe -m pytest -q -m integration
```

### Full suite
Runs everything:

```
./venv/Scripts/python.exe -m pytest -q
```

