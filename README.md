# Onboarding Exercise

_Originally authored by Adacus Green '27 with edits by John MacCormick, summer 2026._

In this module, you will learn how to train your first large language model and test its accuracy, and then you will learn how to test a model's ability to generalize to unseen data. The goal is to give you a hands-on experience with the process of training and evaluating a transformer model.

## Prerequisites
We assume you already have some familiarity with Python, command line interfaces such as bash or PowerShell, git, GitHub, and virtual environments. If you are not familiar with these topics, please spend a few days completing online tutorials.

We also assume you have read an informal account of how large language models and transformer models work. One recommended source for this is Chapter 10, "Generative AI: Unprecedented Scale, Surprising Simplicity" of the 2026 book [Thinking AI: How Artificial Intelligence Emulates Human Understanding](https://johnmaccormick.github.io/thinkingAI-web/). Dickinson students have free electronic access to this through the Dickinson Library. If you do not have this access, contact a lab member for a copy of the chapter. 

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
Now that the tokens are prepped, we can kick off the training routine. Take a look at the configuration file that will be used for this: `config/config.py`. Read over the comments for each parameter to build an elementary understanding of them. Don't worry about unfamiliar terminology. This is just a superficial familiarization.

Once you understand the configuration file, use it to run the training script:
```
python -u train.py config/config.py
```
(The `-u` flag makes the output look better on some systems.)

By default, the model will run for 30 epochs to learn the underlying sequence pattern. (An epoch is a complete pass through the data.) This should take less than a minute on a standard laptop.

Let's examine the output of the training process. The output will show the loss at each epoch:
```
Epoch 1/30 | Loss: 4.0577
Epoch 2/30 | Loss: 1.4594
Epoch 3/30 | Loss: 0.1517
Epoch 4/30 | Loss: 0.0371
```
The loss is a measure of how well the model is fitting the training data. The loss should decrease over time as the model learns. 

Periodically the training script prints out a more detailed summary of training progress:
```
Epoch 10 summary | Time: 0.78s
  Train loss : 0.0067
  Val loss   : 0.0065
  Train acc  : token=100.00% seq=100.00%
  Val acc    : token=100.00% seq=100.00%
```
The training summary shows the model's performance on both the training and validation datasets. They lost as an indirect measure of performance, while the accuracy metrics Are calculated from actual inputs and outputs . They indicate the percentage of correct outputs on the training and validation data respectively . This accuracy can be computed at the token level or the sequence level. The token level indicates what percentage of individual tokens were correct, While the sequence level is the percentage of entire output sequences that were correct. For example, if the input `abcd` produces output `ABCE`, 75% All of the tokens are correct but the sequence as a whole is incorrect.

### 4. Test for accuracy.

Once training concludes, a model checkpoint named with the filename extension `.pth` will be saved in the output directory. This is a binary format that we cannot examine in more detail. However, we can load the model checkpoint into a new program and use it to generate new outputs based on a file of inputs.
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

