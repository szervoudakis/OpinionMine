class NonPositiveException(Exception):
	pass

class LessThanOneException(Exception):
	pass

class RandomVariable:

	def __init__(self, mu=0, sigma=1):
		"""
		Generic random variable class
		
		Attributes:
		  mean (float) representing the mean value of the distribution
		  stdev (float) representing the standard deviation of the distribution
		"""
		if sigma <= 0:
			raise NonPositiveException('Standard deviation must be greater than zero')
		self.mean = mu
		self.stdev = sigma

