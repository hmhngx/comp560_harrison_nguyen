# Onboarding Exercise

_Originally authored by Adacus Green '27 with edits by John MacCormick, summer 2026._

In this module, you will learn how to train your first large language model, test its accuracy, and evaluate its ability to generalize to unseen data. The goal is to gain hands-on experience with the process of training and evaluating a transformer model.

## Prerequisites
We assume you already have some familiarity with Python, command-line interfaces such as bash or PowerShell, Git, GitHub, and Python virtual environments. If you are not familiar with these topics, please spend a few days completing relevant online tutorials.

We also assume you have read an informal account of how large language models and transformer models work. One recommended source is Chapter 10, "Generative AI: Unprecedented Scale, Surprising Simplicity," of the 2026 book [_Thinking AI: How Artificial Intelligence Emulates Human Understanding_](https://johnmaccormick.github.io/thinkingAI-web/). Dickinson students have free electronic access to this through the Dickinson Library. If you do not have access, contact a lab member for a copy of the chapter. 

Assuming you have this prerequisite knowledge, clone this repo, create a virtual environment, and install the required packages (e.g., `pip install -r requirements.txt`). A GPU is _not_ required for this module. If you have a GPU and wish to use it, check your CUDA driver version and install the corresponding version of the `torch` package.


## First Experiment: Train a memorization model

**Background:** In this experiment, you will train a model to capitalize short strings. The model will be trained on a small dataset of input-output pairs where the input is a lowercase string and the output is the same string capitalized. The goal is to observe how well the model memorizes this mapping.

### 1. Generate the data.
The training data for our transformer model consists of a text file containing input-output pairs. Each line of the file contains a lowercase string and its corresponding capitalized string, separated by an equals sign (`=`). For example, the first few lines of the file might look like this:
```
vfl=VFL
j=J
mg=MG
t=T
mgg=MGG
```
We create this training data using the `make_inputs_capital.py` script. Run the following command in your terminal:
```
python make_inputs_capital.py 3 26 1000
```
The three arguments are as follows:
- `3`: The maximum length of the input string.
- `26`: The character set size (using lowercase English letters `a` through `z`).
- `1000`: The number of dataset lines to generate.

The output will be saved in `inputs/capital.txt`. Open this file to examine the generated data. 


### 2. Prepare the data for training.
To transform the raw text of `inputs/capital.txt` into binary tokens and the vocabulary mapping required by the transformer, run the preparation script:
```
python prepare_inputs.py inputs/capital.txt
```
This creates three files in the `data` directory: `train.bin`, `val.bin`, and `meta.pkl`. These files use a binary data format and are not intended for direct inspection, but it is helpful to understand their roles:
- `train.bin`: Contains the training data in binary format.
- `val.bin`: Contains the validation data in binary format. Validation data is used to estimate the accuracy of the model on inputs it has not seen during training.
- `meta.pkl`: Contains the vocabulary mapping—a dictionary mapping each character to a unique integer index. This mapping is used to convert characters to tokens and vice versa.

By default, the `prepare_inputs.py` script splits the input data into a training set containing 90% of the data and a validation set containing the remaining 10%.


### 3. Train the model.
Now that the tokens are prepared, we can begin training. Review the configuration file for this experiment at `config/config.py`. Read through the comments for each parameter to familiarize yourself with them. Do not worry if some terminology is unfamiliar at this stage.

Once you have reviewed the configuration file, run the training script:
```
python -u train.py config/config.py
```
(The `-u` flag unbuffers output so that training logs print promptly on all platforms.)

By default, the model will run for 30 epochs to learn the underlying sequence pattern. (An epoch is one complete pass through the training dataset.) This step should take under a minute on a standard laptop.

The training process prints loss metrics for each epoch:
```
Epoch 1/30 | Loss: 4.0577
Epoch 2/30 | Loss: 1.4594
Epoch 3/30 | Loss: 0.1517
Epoch 4/30 | Loss: 0.0371
```
The loss measures how well the model fits the training data and should decrease steadily as training progresses. 

Periodically, the script prints a more detailed summary:
```
Epoch 10 summary | Time: 0.78s
  Train loss : 0.0067
  Val loss   : 0.0065
  Train acc  : token=100.00% seq=100.00%
  Val acc    : token=100.00% seq=100.00%
```
This summary displays performance on both the training and validation datasets. Loss serves as an indirect optimization measure, whereas the accuracy metrics are calculated directly from generated outputs. Accuracy is reported at both the token level (the percentage of individual characters predicted correctly) and the sequence level (the percentage of full output strings predicted correctly). For example, if the target output for input `abcd` is `ABCD`, but the model produces `ABCE`, 75% of the tokens are correct, but sequence accuracy is 0%.

### 4. Test for accuracy.

When training completes, a model checkpoint with a `.pth` extension is saved in the output directory. While this binary file cannot be read directly, we can load it into an evaluation script to generate outputs for a file of inputs:
```
python generate.py inputs/capital.txt
```
The script displays total processed samples, correct predictions, and the overall accuracy at the end of execution.

You can also test individual input strings. For instance, run `python generate_one.py rg` to verify whether the model outputs `RG`. You can test any string of length 1 to 3 using this script.

## Second Experiment: Out-of-Distribution Generalization

This experiment will investigate the ability of the model to predict outputs of small addition problems modulo 100. The model will be trained on a sparse subset of the complete addition table, and we will evaluate its ability to generalize to unseen combinations. Some typical lines of training data are as follows:
```
77+98=75
6+53=59
99+2=1
85+92=77
38+44=82
```
Note that in each case the correct output must be computed modulo 100. For example, $77 + 98 = 175 \equiv 75 \mod 100$.

### 1. Generate a sparse addition dataset.
Your goal is to generate 3,000 lines of addition problems modulo 100 using the `make_inputs_add.py` script.
Because a complete addition table modulo 100 contains $100 \times 100 = 10\text{,}000$ combinations, the model will observe only 30% of the total search space during training.

**Your Task:** The `make_inputs_add.py` script accepts arguments `V` and `N`, where `V` represents the modulus and `N` represents the number of dataset lines generated. Run the script to generate `inputs/add.txt` using $V = 100$ and $N = 3000$.

### 2. Prepare and train on the data.
Prepare the newly generated `inputs/add.txt` file and run the training pipeline as in the first experiment.

> **Tip:** Mathematical patterns take longer to learn than direct memorization. Before running `train.py`, open `config/config.py`, locate the `epochs` variable, and increase it (e.g., set `epochs = 50` or higher) to allow the network sufficient training steps to discover the arithmetic pattern.

### 3. Evaluate generalization.
You can check training-set performance using `generate.py` on your input file, but the model has already seen most of those examples. To test out-of-distribution performance, try a few standalone examples with `python generate_one.py`, or evaluate the model across every combination from $0+0$ to $99+99$ using the evaluation script:
```
python generate_all_additions.py
```
Notice that overall accuracy exceeds 30% (with default settings, accuracy reaches approximately 96%). This indicates that the model learned the underlying arithmetic rule rather than merely memorizing training examples.


## Running the Test Suite

The project includes a lightweight `pytest` suite with fast unit checks and slower integration checks. Get into the habit of running these tests so you can verify changes as you refactor the codebase.

### Fast tests (recommended during active refactoring)
Runs CLI smoke checks and fast data-logic tests only:

```
python -m pytest -q -m "not integration"
```

### Integration tests (end-to-end pipeline)
Runs integration checks (including lightweight training and inference passes):

```
python -m pytest -q -m integration
```

### Full suite
Runs all tests:

```
python -m pytest -q
```

