from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        modules.append(nn.Conv2d(in_size[0], 32, kernel_size=3, padding=1))
        modules.append(nn.Dropout(p=0.3))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        modules.append(nn.Dropout(p=0.3))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        modules.append(nn.Dropout(p=0.3))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.Conv2d(128, 1, kernel_size=3, padding=1))
        modules.append(nn.AvgPool2d(64))
        self.cnn = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.cnn(x).view(x.shape[0], 1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        modules.append(nn.ConvTranspose2d(z_dim, 64, kernel_size=featuremap_size,
                                          stride=featuremap_size, padding=0))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.Dropout(p=0.6))
        modules.append(nn.ConvTranspose2d(64, 32, kernel_size=featuremap_size,
                                          stride=featuremap_size, padding=0))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.Dropout(p=0.6))
        modules.append(nn.ConvTranspose2d(32, out_channels, kernel_size=featuremap_size,
                                          stride=featuremap_size, padding=0))
        modules.append(nn.Tanh())
        self.cnn = nn.Sequential(*modules)
        self.dsc_best_loss = None
        self.gen_best_loss = None
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(with_grad)
        samples = torch.normal(mean=0.0, std=1.0, size=[n, self.z_dim], device=device, requires_grad=with_grad)
        samples = self.forward(samples)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = torch.unsqueeze(z, dim=2)
        z = torch.unsqueeze(z, dim=3)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """

    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    labels = torch.empty(len(y_data), device=y_data.device).uniform_(data_label - label_noise / 2, data_label + label_noise / 2)
    generated_labels = torch.empty(len(y_data), device=y_data.device).uniform_(1 - data_label - label_noise / 2,
                                                         1 - data_label + label_noise / 2)
    losser = nn.BCEWithLogitsLoss(reduction='mean')
    loss_data = losser(y_data, labels)
    loss_generated = losser(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    losser = nn.BCEWithLogitsLoss(reduction='mean')
    loss = losser(y_generated, torch.tensor([data_label] * len(y_generated), dtype=torch.double, device=y_generated.device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======    
    x_scores = dsc_model(x_data).view(-1)
    samples = gen_model.sample(len(x_data), with_grad=True)
    samples_scores = dsc_model(samples).view(-1)
    dsc_optimizer.zero_grad()

    dsc_loss = dsc_loss_fn(x_scores, samples_scores)
    dsc_loss.backward(retain_graph=True)

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(samples_scores)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    
    
    if gen_model.dsc_best_loss is None or (dsc_losses < gen_model.dsc_best_loss and gen_losses < gen_model.gen_best_loss):
        gen_model.dsc_best_loss = dsc_losses
        gen_model.gen_best_loss = gen_losses
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
