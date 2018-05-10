import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def process_glove(pathname):
	pass # create glove embeddings for necessary words


class Dataset:
	def __init__(self, pathname):
		pass # import and clean dataset
		self.vocab_size = 0
		self.word2id = dict()
		self.id2word = list()


class WordGAN:
	def __init__(self, data, generator, discriminator, latent_distribution):
		if type(data) == Dataset:
			self.data = data
		else:
			self.data = Dataset(data)

		self.generator = generator # must take in latent_distribution, output sequence of word embeddings
		self.discriminator = discriminator # take in sequence of word embeddings, output binary value
		self.latent_dist = latent_distribution # must match input of generator

		self._init_embeddings()
		# create train operations for generator and discriminator
		# create loss operation as well

	def _init_embeddings(self):
		self.embeddings = nn.Embedding(self.data.vocab_size, self.embed_dim)
		# process glove embeddings, init new embedding if not found
		self.embeddings.weight = nn.Parameter(glove_embeddings)
		self.embeddings.requires_grad = False

	def _generate(self):
		pass
		# define generate to output a sequence of vectors representing the word embeddings of a sentence

	def _discriminate(self):
		pass
		# define descriminate to take a sequence of vectors and determine if it is proper english (fake or not)

	def _loss(self):
		pass
		# define as cross entropy between one hot word vector and dot product of predictions and word vector
		# find formulation that doesn't require going over entire vocab

	def train(self):
		pass # train model

	def save(self):
		pass # save model parameters

	def sample(self, latent_vectors=None):
		predictions = self.generator(latent_vectors)
		# compute dot produce, create one-hot vectors
		# input one-hot to data to create sentences
