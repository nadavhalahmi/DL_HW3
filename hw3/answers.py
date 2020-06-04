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

It shows memory longer than the sequence length thanks to the model's state that represents the history the
model encountered. 
It is therefore, remembers results came from longer sequence, as the model learns from every input it gets. 
Given a sequence in a specific time, the model's hidden layers will be updated with respect to this sequence and it 
will affect the latter sequences processing.

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
characters, but can lead to errors. In the extreme case, we will end up with complete gibberish, made of many characters,
each in almost the same frequency. 
3. When the temperature is very low, we will end up with one character (or "structure")which is very likely to be chosen.
In the extreme case, we can end up with a sequence full of this char (or "structure") only. 

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
    hypers['h_dim'] = 10
    hypers['z_dim'] = 15
    hypers['x_sigma2'] = 0.0004
    # hypers['x_sigma2'] = 0.9
    hypers['learn_rate'] = 4e-4
    hypers['betas'] = (1 - 1e-1, 1 - 1e-3)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

$\sigma^2$ has a role of a regularization strength as we have in the loss function 2 terms: the regularization term and the 
data term. Therefore, $\sigma^2$ actually controls the importance of the regularization term: the bigger the sigma, the stronger regularization term). $\sigma^2$ also controls the amount of uncertainty that we had in the generation of an instance. Therefore, the bigger $\sigma^2$- the less we should trust our data. 

"""

part2_q2 = r"""
**Your answer:**

1. 

- The reconstruction lost is actually a data fitting term of the VAE loss and it tells how well the model generated 
points fit to the data.
- The KL diversion loss is actually a regularization term of the VAE loss and it tells came from mesuring the diversion
between the model posterior and the actual posterior, meaning it tells how much data is lost after the reconstruction process.
2. Without the KLV loss there will be difference between the distribiution we will get to the real distribution. 
Therefore it is there to keep overlapping. It is actually measures the similarities between 2 probabilties, in order to 
find the one that gives the distribution overlapping the real one.
3. Discontiuity is removed in the latent space, and that helps to improve generation tasks. It also allows interpolations between classes.

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
    hypers['batch_size'] = 64
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.25
    hypers['discriminator_optimizer'] = dict(type='Adam', weight_decay=0.015, betas=(1-1e-1, 1 - 1e-3), lr=2e-4)
    hypers['generator_optimizer'] = dict(type='Adam', weight_decay=0.015, betas=(1-1e-1, 1 - 1e-3), lr=2e-4)

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
