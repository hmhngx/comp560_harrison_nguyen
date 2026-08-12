# Phonebook experiment

This experiment is an introduction to the theme of "remembering and forgetting in micro-LLMs." The task for the the model is to memorize phone numbers provided in the format
```
    Ashra Dellis=333-512-9823
    Calio Blackham=644-383-9643
    Brie Brownby=403-662-3854
    Iano Carers=713-441-0043
```
The experiment is divided into two phases. In Phase A, we memorize a fixed number of phone numbers from dataset A. In Phase B we start with the model trained in phase A and memorize a new set of phone numbers from dataset B. We call this process _model adaptation_. In this case, the phase A model is adapted to dataset B. If the model is adapted to a completely new dataset, it typically forgets most or all of the facts it previously memorized. In the, this phenomenon is known as _catastrophic forgetting_ or simply _forgetting_. This experiment will lead us through a demonstration of the forgetting process.

## Instructions for the forgetting experiment

1. Make phone book (20 names), then split it into two halves (10 names each) for phase A and phase B:

```bash
cd phonebook/
py create_phonebook.py 20
cd inputs/
head -n 10 phonebook.txt > phonebookA.txt
tail -n 10 phonebook.txt > phonebookB.txt
# back to phonebook dir
cd ..
```


2. Prepare the Phase A and Phase B data separately. Notice that to compare the remembering and forgetting properties we Train and evaluate on separate data sets . In phase A, we train on phonebook A, but evaluate on phonebook B. In phase B, we train on phonebook B, but evaluate on phone book A.
```bash
py prepare_phonebook.py --train-input inputs/phonebookA.txt --eval-input inputs/phonebookB.txt --out-dir data_phase1
py prepare_phonebook.py --train-input inputs/phonebookB.txt --eval-input inputs/phonebookA.txt --out-dir data_phase2
```

3. Train Phase A model
```bash
# back to root dir
cd ..
py train_completions.py --data-dir phonebook/data_phaseA --out-dir phonebook/out_phaseA phonebook/config/phonebook.py
```
Examine the outputs. Do the accuracies on phone book A and phone book B correspond to your expectations?

4. Phase B: adapt Phase A model to phonebook B
```bash
 py train_completions.py --data-dir phonebook/data_phaseB --out-dir phonebook/out_phaseB --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py
 ```
Again examine the outputs, and decide if the accuracies on phone book A and phone book B correspond to your expectations.

## Next steps

You are ready to begin research on various aspects of the memory remembering and forgetting process. Here are a few things to try, in no particular order. Consult with the instructor and other students to get other suggestions for your initial research direction.
- Install matplotlib in your virtual environment. Make visualizations of the outputs of each step in the experiment above. Optionally, integrate this with wandb.com and visualize your experiments there.
- Make the very small by adjusting the configuration file `phonebook/config/phonebook.py`. (For example, reduce the number of layers, number of attention heads, or embedding dimension.) Determine the _capacity_ of your model by training on larger and larger phonebooks until it is impossible to get 100 percent accuracy. 
- Try different learning rates and other hyperparameters to see how they affect the remembering and forgetting process. For example, does a smaller learning rate reduce forgetting? Does a larger learning rate increase forgetting? What about the number of training epochs?
- Replace the phase B data set with data that contains both data set A and data set B, except the phase B is repeated k times. For example, if k=3, then the phase B data set would contain 3 copies of the phase B data set and 1 copy of the phase A data set. How does this affect forgetting? What value of k permits the fastest learning of 100 percent accuracy on all data?
- Reduce the tendency to forget by implementing a technique called _L2-SP regularization_. Guidance is available in the [transcript of an AI chat about regularization](./docs/L2-SP-regularization-chat.md). Optionally, check out the paper "Rethinking the Value of Network Pruning" by Li et al. (2022) for details. The idea is to add a regularization term to the loss function that penalizes the model for deviating too far from the original weights learned in phase A. This can help the model retain knowledge from phase A while learning new information in phase B.
- Consult with the instructor for various ideas about how to imitate knowledge consolidation, analogous to how a human might consolidate new knowledge with old knowledge while sleeping. The main idea is to store metadata as facts in the memorized knowledge base and use this to review previous knowledge periodically while training and while "sleeping." This falls into the general area of the neural network literature known as _replay methods_.


## AI Acknowledgement

Most of the code for this experiment was written by GitHub Copilot using GPT-5.3-codex. John MacCormick made some edits to the code. This README was mostly written by John MacCormick but with substantial AI completions.