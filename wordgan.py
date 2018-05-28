import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

# configuration dataclass
@dataclass
class Config:
	pickle_path: str
	pretrain_path: str
	embed_dim: int


class Dataset:
	def __init__(self, data_path, config):
		self.vocab_size = 0
		self.word2id = dict()
		self.id2word = list()
		self.embed_dim = config.embed_dim
		self.config = config

	def _init_pretrained_embeddings(self):
		try:
			# get pickle
			self.config.pickle_path
		except IOError:
			self.word_embeddings = np.empty(self.vocab_size, self.word_dim)
				with open(self.config.pretrain_path, 'r') as embedding_file:
					# pickle word embeddings
					remaining_words = set(self.word2id.keys())
					for line in embedding_file:
						# process line, see if in vocabulary
						tokens = line.split()
						assert len(tokens) == self.word_dim + 1
						try:
							index = self.word2id[tokens[0]]
							self.word_embeddings[index] = np.array(float(token) for token in tokens[1:])
							remaining_words.remove(tokens[0])
						except KeyError:
							continue

				for word in remaining_words:
					# initialize all unfound words
					self.word_embeddings[word] = np.random.uniform(self.word_dim)

				# pickle as pickle path


	def get_pretrained_embeddings(self):
		if not self.word_embeddings:
			self._init_pretrained_embeddings

		return self.word_embeddings


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
