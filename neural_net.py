import random
import csv
from scipy.special import expit
from sklearn.metrics import log_loss

def sigmoid(x):
	return expit(x)

def loss( y_expected, y_predicted):
	return log_loss(y_expected, y_predicted, normalize=False)

# class neuron:

class network:

	def __init__(self, input_layer_len=784, hidden_layer_len=25, output_layer_len=10):

		# layers size
		self.input_layer_len = input_layer_len
		self.hidden_layer_len = hidden_layer_len
		self.output_layer_len = output_layer_len

		# initialize neurons
		self.nodes_input = input_layer_len * [1.0]
		self.nodes_hidden = hidden_layer_len * [1.0]
		self.nodes_output = output_layer_len * [1.0]

		# weights matrixes : random values
		self.weight_ih = [[random.uniform(-0.001, 0.001) for x in range(hidden_layer_len)] for y in range(input_layer_len)] 
		self.weight_ho = [[random.uniform(-0.001, 0.001) for x in range(output_layer_len)] for y in range(hidden_layer_len)]

		# auxiliary weight matrixes
		self.aux_wih = self.weight_ih
		self.aux_who = self.weight_ho

	def propagate(self, inputs):
		for i in range(self.input_layer_len):
			self.nodes_input[i] = inputs[i]

		for i in range(self.hidden_layer_len):
			value = 0.0
			for j in range(self.input_layer_len):
				value = value + self.nodes_input[j] * self.weight_ih[j][i]
			self.nodes_hidden[i] = sigmoid(value)

		for i in range(self.output_layer_len):
			value = 0.0
			for j in range(self.hidden_layer_len):
				value = value + self.nodes_hidden[j] * self.weight_ho[j][i]
			self.nodes_output[i] = sigmoid(value)

		return self.nodes_output

	def backpropagate(self, y_expected):

		output_deltas = [0.0] * self.output_layer_len
		for i in range(self.output_layer_len):
			output_deltas[i] = (y_expected[i] - self.nodes_output[i]) * loss(y_expected, self.nodes_output)

		hidden_deltas = [0.0] * self.hidden_layer_len
		for i in range(self.hidden_layer_len):
			error = 0.0
			for j in range(self.output_layer_len):
				error = error + output_deltas[j] * self.weight_ho[i][j]
			hidden_deltas[i] = error 

		return output_deltas, hidden_deltas

	def update_weights(self, y_expected, output_deltas, hidden_deltas, learning_rate, update=True):
		for i in range(self.hidden_layer_len):
			for j in range(self.output_layer_len):
				val = output_deltas[j] * self.nodes_hidden[i]
				self.aux_who[i][j] = self.aux_who[i][j] + (learning_rate * val)
				if update:
					self.weight_ho[i][j] = self.aux_who[i][j]
					#self.aux_who[i][j] = 0.0

		for i in range(self.input_layer_len):
			for j in range(self.hidden_layer_len):
				val = hidden_deltas[j] * self.nodes_input[i]
				self.aux_wih[i][j] = self.aux_wih[i][j] + (learning_rate * val)
				if update:
					self.weight_ih[i][j] = self.aux_wih[i][j]
					#self.aux_wih[i][j] = 0.0

	def test(self, data):
		for d in data:
			print d[1], '=>', self.propagate(d[0])

	def train(self, data, epochs=12, learning_rate=1, type='sgd', batch_size=10):
		
		size = len(data)
		print '----- ', type, ' ----- learning rate: ', learning_rate, ' ----- hidden layer size: ', self.hidden_layer_len , ' -----'
		if type == 'mini':
			print batch_size
		for i in range(epochs):
			error = 0.0
			count = 1
			for d in data:
				inputs = d[0]
				outputs = d[1]
				self.propagate(inputs)
				error = error + loss(d[1], self.nodes_output)
				output_deltas, hidden_deltas = self.backpropagate(outputs)

				if type == 'sgd':
					self.update_weights(outputs, output_deltas, hidden_deltas, learning_rate, True)
				elif type == 'mini':
					if count % batch_size == 0:
						self.update_weights(outputs, output_deltas, hidden_deltas, learning_rate, True)
					else:
						self.update_weights(outputs, output_deltas, hidden_deltas, learning_rate, False)
				elif type == 'gd':
					if count % size == 0:
						self.update_weights(outputs, output_deltas, hidden_deltas, learning_rate, True)
					else:
						self.update_weights(outputs, output_deltas, hidden_deltas, learning_rate, False)

				count = count + 1
				

			print 'epoch: ', i ,', erro: ', error/size
			

def _xor():
	data = [
		[[0, 0], [1, 0]],
		[[0, 1], [0, 1]],
		[[1, 0], [0, 1]],
		[[1, 1], [1, 0]]
	]
	n = network(2, 2, 2)
	n.train(data)
	n.test(data)

def mnist():
	file = open('data_tp1')
	reader = csv.reader(file)

	data = []
	labels = []

	for line in reader:
		ar = 10 * [0]
		ar[int(line[0])] = 1

		labels.append(int(line[0]))
		data.append([[float(x) for x in line[1:]], ar])

	print 'variando tipo'
	# n = network()
	# n.train(data, learning_rate=1, type='gd')
	# n = network()
	# n.train(data, learning_rate=1, type='sgd')
	n = network(hidden_layer_len=50)
	n.train(data, learning_rate=1, type='mini', batch_size=10)
	n = network(hidden_layer_len=50)
	n.train(data, learning_rate=1, type='mini', batch_size=50)

	# print 'variando learning rate'
	# n = network()
	# n.train(data, learning_rate=0.5, type='gd')
	# n = network()
	# n.train(data, learning_rate=10, type='gd')

	# print 'variando tamanho da camada oculta'
	# n = network(hidden_layer_len=50)
	# n.train(data, learning_rate=1, type='sgd')
	# n = network(hidden_layer_len=100)
	# n.train(data, learning_rate=1, type='sgd')

	#n.test(data)

if __name__ == '__main__':
    #_xor()
    mnist()

