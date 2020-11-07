import numpy as np
from .Generalrandomvariable import RandomVariable, NonPositiveException, LessThanOneException 

class Exponential(RandomVariable):
	def __init__(self, rate=1):
		if rate <= 0:
			raise NonPositiveException('Value of rate parameter must be greater than zero')
		self.rate = rate
		RandomVariable.__init__(self, 1/rate, 1/rate)

	def get_mean(self):
		return self.mean

	def get_variance(self):
		return self.stdev ** 2

	def get_stdev(self):
		return self.stdev

	def get_rate(self):
		return self.rate

	def get_scale(self):
		return self.mean

	def generate(self, n=1):
		if n < 1:
			raise LessThanOneException('x must be greater than or equal to 1')
		elif n == 1:
			return np.random.exponential(self.get_mean())
		else:
			return np.random.exponential(self.get_mean(), n)

	def __repr__(self):
		return "Exponential({})".format(self.get_rate())
