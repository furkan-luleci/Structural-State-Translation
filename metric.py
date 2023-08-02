from numpy import std
import numpy as np
import torch
import torch



# Define FID score
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), std(act1)
	mu2, sigma2 = act2.mean(axis=0), std(act2)
	# calculate sum squared difference between means
	diff = (mu1 - mu2)**2.0
	# calculate sqrt of product between cov
	stdmean = sigma1*sigma2
	# calculate score
	fid = diff + sigma1**2 + sigma2**2 - 2.0 * stdmean
	return fid
