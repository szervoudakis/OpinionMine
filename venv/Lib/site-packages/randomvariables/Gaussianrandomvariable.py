import numpy as np
from .Generalrandomvariable import RandomVariable, NonPositiveException, LessThanOneException 

class Gaussian(RandomVariable):

	def __init__(self, mu=0, sigma=1):
		RandomVariable.__init__(self, mu, sigma)

	def get_mean(self):
		return self.mean

	def get_variance(self):
		return self.stdev ** 2

	def get_stdev(self):
		return self.stdev

	def generate(self, n=1):
		if n < 1:
			raise LessThanOneException('x must be greater than or equal to 1')
		elif n == 1:
			return np.random.normal(self.get_mean(), self.get_stdev())
		else:
			return np.random.normal(self.get_mean(), self.get_stdev(), n)

	def __repr__(self):
		return "Gaussian({}, {})".format(self.get_mean(), self.get_variance())


