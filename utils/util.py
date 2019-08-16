# %matplotlib inline
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def normalize(arr, input_bound=(0,255), output_bound=(-1,1)):
	input_lower_bound, input_upper_bound = input_bound 
	output_lower_bound, output_upper_bound = output_bound

	scale = (output_upper_bound - output_lower_bound) / (input_upper_bound - input_lower_bound)
	mid_in = (input_lower_bound + input_upper_bound) / 2
	mid_out = (output_lower_bound + output_upper_bound) / 2

	out = np.subtract(arr,mid_in)
	out = np.multiply(arr,scale)
	out = np.add(arr,mid_out)
	return out

def plot_loss(losses, label):
	l = []
	for epoch_loss in losses:
		# print('**** epoch_loss.size : {}'.format( len(epoch_loss)))
		total_epoch_loss = 0
		for mini_batch_loss in epoch_loss:
			# print('**** min_batch_loss.size : {}'.format(len(mini_batch_loss)))
			for sample_loss in mini_batch_loss:
				# print('**** sample_loss.size : {}'.format(len(sample_loss)))
				total_epoch_loss += sample_loss
		l.append(total_epoch_loss)

	counter = [ ix for ix in range(len(losses))]
	
	plt.subplot(3,1,1)
	plt.plot(l,marker='o',label=label)
	plt.legend(loc='upper right', borderaxespad=0.0)
	plt.legend(bbox_to_anchor=(1.04,0.5), loc='upper left', borderaxespad=0.0)
	plt.ylabel('Losses',fontsize=6,fontweight='bold')
	plt.xticks(counter)
	plt.grid(True)

def plot_accuracy(accuracies,num_data,label, ylabel, xlabel = None, subplot = (3,1,2)):
	counter = [ ix for ix in range(len(accuracies)) ]
	accuracies = [ acc/num_data for acc in accuracies]
	plt.subplot( subplot[0], subplot[1], subplot[2])
	plt.plot(accuracies,marker='o',label=label)
	plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', borderaxespad=0.0)
	plt.ylabel(ylabel,fontsize=6,fontweight='bold')
	if xlabel is not None:
		plt.xlabel('epochs')
	plt.xticks(counter)
	plt.grid(True)

def plot_accuracy_test(accuracies,num_data,label, subplot = (3,1,3)):
	counter = [ ix for ix in range(len(accuracies)) ]
	accuracies = [ acc/num_data for acc in accuracies]
	plt.subplot( subplot[0], subplot[1], subplot[2])
	plt.plot(accuracies,marker='o',label=label)
	plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', borderaxespad=0.0)
	plt.ylabel('Accuracy of test data after and before training',fontsize=6,fontweight='bold')
	plt.xticks(counter)

def plot_var_accuracy( var_acc, subplot = (1,1,1)):
	variances, accuracies = zip(*var_acc)
	plt.subplot( subplot[0], subplot[1], subplot[2])
	plt.plot( variances, accuracies, marker='o')
	plt.ylabel('Accuracy',fontsize=6,fontweight='bold')
	plt.xlabel('Variance',fontsize=6,fontweight='bold')
	plt.xticks(variances)

def pca_reduction(data, n_components):
	num_data = data.shape[0]
	reduced_data =  np.copy(data)

	reduced_data = reduced_data.reshape(num_data,-1)
	pca = PCA(n_components= n_components)
	reduced_data = pca.fit(reduced_data).transform(reduced_data).reshape(num_data,-1,1)
	
	return reduced_data


def data_normalization(data):
	normalized_data = np.zeros_like(data)

	for ix in range(data.shape[0]):
		normalized_data = ( data[ix] - data[ix].mean() ) / data[ix].std()

	return normalized_data

# def write_file('?')
