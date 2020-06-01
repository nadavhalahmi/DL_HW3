r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=256, seq_len=64,
        h_dim=512, n_layers=3, dropout=0.5,
        learn_rate=0.001, lr_sched_factor=0.5, lr_sched_patience=2,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    start_seq = "ACT I."
    temperature = 0.25
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

The RNN is not capable of learning on large sequences as it will suffer from vanishing/exploding gradients.
Moreover, propagating the gradients will take a lot of time.

"""

part1_q2 = r"""
**Your answer:**

It shows memory longer than the sequence length thanks to the history of the states that the model remembers.
It is therefore, remembers results came from longer sequence, as the model learns from every input it gets. 
Given a sequence in a specific time $t$, the model's hidden layers will be updated with respect to this sequence and it 
will matter for the next sequences which will come in $time > t$.

"""

part1_q3 = r"""
**Your answer:**

We are not shuffling the order of batches when training because in comparison to other trainable models,
we need to keep the order of the batches, because the input is sequential, and each batch depends on the ones before it.
In other models, there's no meaning to the order of the batches, and therefore, these can be shuffled.

"""

part1_q4 = r"""
**Your answer:**

1. We lower the temperature in order to increase the chance of sampling the char(s) with the highest scores compared 
to the others. Since we have more classes than usual, we would like to increase the variance, and end up with less
uniform distribution.
2. When the temperature is very high, we won't be able to distinguish between possible target chars. They will get
very close scores, and the distribution will be close to uniform. It will increase the chance of getting more diverse
characters, but can lead to errors.
3. When the temperature is very low, we will end up with one character which is very likely to be chosen. In the extreme
case, we can end up with a sequence full of this char only. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 25
    hypers['z_dim'] = 10
    hypers['x_sigma2'] = 0.9
    hypers['learn_rate'] = 2e-4
    hypers['betas'] = (1-1e-1, 1-1e-3)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


