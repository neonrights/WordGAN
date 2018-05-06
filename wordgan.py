import cntk
import numpy as np

def process_glove():
	pass # create glove embeddings for necessary words


class Dataset:
	def __init__(self, path):
		pass # import and clean dataset


class WordGAN:
	def __init__(self):
		pass

	def _predict(self):
		pass # define predict to output a sequence of vectors representing the word embeddings of a sentence

	def _loss(self):
		pass # define as cross entropy between one hot word vector and dot product of predictions and word vector
		# find formulation that doesn't require going over entire vocab
