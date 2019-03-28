from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import random
import os
import numpy as np
import tensorflow as tf

AtomData = [
['X','Nullium', 0.0],
['H','Hydrogen', 1.0],
['He','Helium', 2.0],
['Li','Lithium', 3.0],
['Be','Beryllium', 4.0],
['B','Boron', 5.0],
['C','Carbon', 6.0],
['N','Nitrogen', 7.0],
['O','Oxygen', 8.0],
['F','Fluorine', 9.0],
['Ne','Neon', 10.0],
['Na','Sodium', 11.0],
['Mg','Magnesium', 12.0],
['Al','Aluminum', 13.0],
['Si','Silicon', 14.0],
['P','Phosphorus', 15.0],
['S','Sulfur', 16.0],
['Cl','Chlorine', 17.0],
['Ar','Argon', 18.0],
['K','Potassium', 19.0],
['Ca','Calcium', 20.0],
['Sc','Scandium', 21.0],
['Ti','Titanium', 22.0],
['V','Vanadium', 23.0],
['Cr','Chromium', 24.0],
['Mn','Manganese', 25.0],
['Fe','Iron', 26.0],
['Co','Cobalt', 27.0],
['Ni','Nickel', 28.0],
['Cu','Copper', 29.0],
['Zn','Zinc', 30.0],
['Ga','Gallium', 31.0],
['Ge','Germanium', 32.0],
['As','Arsenic', 33.0],
['Se','Selenium', 34.0],
['Br','Bromine', 35.0],
['Kr','Krypton', 36.0],
['Rb','Rubidium', 37.0],
['Sr','Strontium', 38.0],
['Y','Yttrium', 39.0],
['Zr','Zirconium', 40.0],
['Nb','Niobium', 41.0],
['Mo','Molybdenum', 42.0],
['Tc','Technetium', 43.0],
['Ru','Ruthenium', 44.0],
['Rh','Rhodium', 45.0],
['Pd','Palladium', 46.0],
['Ag','Silver', 47.0],
['Cd','Cadmium', 48.0],
['In','Indium', 49.0],
['Sn','Tin', 50.0],
['Sb','Antimony', 51.0],
['Te','Tellurium', 52.0],
['I','Iodine', 53.0],
['Xe','Xenon', 54.0],
['Cs','Cesium', 55.0],
['Ba','Barium', 56.0],
['Lu','Lutetium', 71.0],
['Hf','Hafnium', 72.0],
['Ta','Tantalum', 73.0],
['W','Tungsten', 74.0],
['Re','Rhenium', 75.0],
['Os','Osmium', 76.0],
['Ir','Iridium', 77.0],
['Pt','Platinum', 78.0],
['Au','Gold', 79.0],
['Hg','Mercury', 80.0],
['Tl','Thallium', 81.0],
['Pb','Lead', 82.0],
['Bi','Bismuth', 83.0]]

ELEMENT_MODES = np.array([[ 0.00000000, 0.00000000, 0.00000000, 0.00000000],
[-3.1752536, -1.142872, -1.5934169, -2.4993973],
[-1.6680579, -2.399622, -0.30887717, -4.1619945],
[-5.486701, -2.0788946, -0.39364037, 2.938814],
[-2.4827032, -1.445508, -0.7483087, -0.20502234],
[-2.0425463, -0.9348759, -1.3872833, -0.5361312],
[-1.491054, -0.7777871, -1.728497, -1.2735122],
[-0.90035325, -1.6420275, -1.5011224, -1.5700817],
[-0.6509637, -0.8824361, -1.6013727, -2.2170765],
[0.34993896, -0.48677668, -0.8651852, -2.2496543],
[1.2620814, -3.197368, -0.32232887, -3.1073914],
[-4.995693, -1.9888805, 0.3074603, 3.2935553],
[-2.1174676, -1.7300301, -1.0488031, 1.1299887],
[-1.8376874, -0.8382999, -1.1075414, 1.03701],
[-1.5016971, -0.34238195, -1.1756057, 0.36012787],
[-0.86128306, -0.87700784, -1.478201, -0.5068094],
[-0.5057297, -0.42307985, -1.4144773, -0.8186628],
[0.22011341, -0.24248785, -0.657427, -0.63246554],
[1.0129275, -2.2348046, -0.55019075, -0.97796446],
[-6.8432403, -3.3034875, 1.8118958, 6.0466933],
[-3.3521194, -3.026552, -0.1183187, 3.7422986],
[-2.891919, -2.332284, -0.19537862, 3.2130826],
[-2.2785447, -1.9162403, -0.34983703, 2.647319],
[-2.0962763, -1.1506464, -0.35487393, 2.3892012],
[-2.4368107, -0.44457, 0.08601359, 1.7862847],
[-0.88687956, -1.1052301, -0.6379649, 1.4635943],
[-0.72981477, -0.54607224, -0.55103, 1.3071768],
[-0.60065734, -0.1956464, -0.5960953, 1.2143358],
[-0.5136049, 0.053226296, -0.6234979, 1.1415871],
[-0.5104966, 0.34388846, 0.015960256, 1.0114733],
[0.049403206, -0.6391534, -0.047429703, 1.0651706],
[0.027447658, -0.1964888, -0.15934445, 1.7807101],
[0.88784385, -0.19772023, 0.15759389, 1.1473469],
[1.1195555, -0.46827912, 0.32592872, 0.7290432],
[1.2601672, -0.10638335, 0.15066789, 0.40814894],
[1.4179097, 0.21814197, -0.07076641, 0.03931976],
[1.8894219, -1.2022529, 0.5307149, -0.150689],
[-5.9239745, -2.8030112, 2.5113463, 6.282557],
[-2.9968958, -2.3782315, 0.8338701, 4.434445],
[-2.442522, -1.5953307, 0.5200376, 3.6678998],
[-1.8460523, -1.0551751, 0.25150445, 3.1337967],
[-2.3060524, -0.3539464, 0.5700243, 2.5993133],
[-1.4924102, -0.1875788, 0.44804636, 2.029929],
[-0.8326731, -0.46050042, 0.209958, 2.3181612],
[-0.80834424, 0.2633914, 0.8186634, 2.0131528],
[-0.54405147, 0.3910988, 0.9381731, 1.9241306],
[-0.8924645, 0.96924984, 1.2748532, 1.1111994],
[0.05311054, 0.51894784, 1.1417516, 1.6077937],
[0.3905682, -0.82707053, 0.5625302, 1.7457047],
[0.584229, -0.4761517, 0.48003772, 2.6229346],
[1.3843664, -0.23806745, 0.7560801, 1.7684854],
[1.7226951, -0.13935018, 0.8676731, 1.0816072],
[1.8762771, 0.16291368, 0.65354854, 0.7105214],
[2.183301, 0.5345518, 0.37835196, 0.3463258],
[2.9667006, -1.2527919, 1.3462843, 0.37595224],
[-6.0408206, -2.508442, 3.6843417, 8.664445],
[-3.4308436, -2.469945, 2.0547187, 6.7617993],
[-0.7296025, -1.5791643, 3.4319758, 6.784385],
[0.065387115, -1.3340049, 2.7725933, 5.1155105],
[0.5279452, -0.835859, 2.3184357, 4.197991],
[1.131108, -0.7014899, 2.4800014, 4.318073],
[1.4884478, -1.300788, 2.4098122, 4.2357035],
[1.5781091, -0.32093132, 2.7768946, 3.8363068],
[1.7179236, 0.20936537, 2.9446495, 3.4869108],
[2.407444, 1.6401689, 3.827573, 1.9580134],
[2.5075088, 1.919955, 3.945945, 1.4942778],
[2.1382914, -0.7869838, 2.6290567, 2.4660778],
[4.1753864, 0.4612016, 3.7278538, 3.2893684],
[4.1273212, 0.2145474, 3.291612, 2.4639742],
[4.465515, 0.570441, 2.8771896, 2.0318608]], dtype=np.float64)


class NNModel(object):
	def __init__(self, data=None, name=None, hidden_layers=[128, 128], batch_size=32):
		self.data = data
		self.batch_size = batch_size
		self.hidden_layers = hidden_layers
		self.learning_rate = 0.001
		self.momentum = 0.9
		self.validation_ratio = 0.1
		self.test_ratio = 0.1
		self.max_steps = 1000
		self.tf_precision = eval("tf.float32")
		self.activation_function = tf.nn.softplus
		self.step = 0
		self.validation_freq = 5
		self.name = "ElpasEM_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.directory = "./"+self.name

		if name != None:
			self.name = name
			self.directory = "./"+self.name
			self.energy_mean = 0.0
			self.energy_std = 0.0

	def train(self):
		self.load_data(self.data)
		self.normalize()

		self.build_graph()
		for i in range(self.max_steps):
			self.step += 1
			self.train_step()
			if self.step%self.validation_freq == 0:
				validation_loss = self.validation_step()
				if self.step == self.validation_freq:
					self.best_loss = validation_loss
					self.save_checkpoint()
				elif validation_loss < self.best_loss:
					self.best_loss = validation_loss
					self.save_checkpoint()

		self.saver.restore(self.sess, tf.train.latest_checkpoint(self.directory))
		print("Best validation loss: ", self.validation_step())
		test_loss, test_errors = self.test()
		print("Test loss: ", test_loss)
		with open("test_errors.dat", "w") as f:
			for i in range(len(test_errors)):
				f.write(str(test_errors[i])+"\n")
		self.sess.close()

	def evaluate(self, ANs):
		self.build_graph()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(self.directory))
		ANs = np.array(ANs, dtype=np.int32).reshape(1,4)
		feed_dict = {self.Zs_pl: ANs}
		formation_energy = self.sess.run(self.energy_preds,  feed_dict=feed_dict)
		print("Formation energy (eV):", formation_energy)
		return

	def build_graph(self, restart=False):
		self.Zs_pl = tf.placeholder(tf.int32, shape=[None, 4])
		self.energy_pl = tf.placeholder(self.tf_precision, shape=[None])

		self.Z_map_idx = tf.constant([int(data[2]) for data in AtomData], dtype=tf.int32)
		self.gather_idx = tf.where(tf.equal(tf.expand_dims(self.Zs_pl, axis=-1), self.Z_map_idx))[:,-1]
		self.element_modes = tf.Variable(ELEMENT_MODES, trainable=True, dtype=self.tf_precision)
		self.feature_vec = tf.reshape(tf.gather(self.element_modes, self.gather_idx), [-1, 4*4])
		self.energy_std_tf = tf.Variable(self.energy_std, trainable=False, dtype=self.tf_precision)
		self.energy_mean_tf = tf.Variable(self.energy_mean, trainable=False, dtype=self.tf_precision)

		self.norm_energy_preds = self.build_network(self.feature_vec)
		self.energy_preds = (self.norm_energy_preds * self.energy_std_tf) + self.energy_mean_tf
		self.energy_loss = self.loss_op(self.energy_preds - self.energy_pl)
		self.train_op = self.optimizer(self.energy_loss, self.learning_rate, self.momentum)

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.saver = tf.train.Saver(max_to_keep = 3)
		if restart:
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.directory))
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def build_network(self, feature_vector):
		for i in range(len(self.hidden_layers)):
			if i == 0:
				layer = tf.layers.dense(inputs=feature_vector, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)
			else:
				layer = tf.layers.dense(inputs=layer, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)

		predictions = tf.layers.dense(inputs=layer, units=1,
				activation=None, use_bias=True)
		return tf.squeeze(predictions)

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def optimizer(self, loss, learning_rate, momentum):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def load_data(self, data_dict):
		self.Z_data = data_dict["ANs"]
		self.energy_data = data_dict["FEs"]
		self.num_cases = self.Z_data.shape[0]
		self.num_validation_cases = int(self.validation_ratio * self.num_cases)
		self.num_test_cases = int(self.test_ratio * self.num_cases)
		num_validation_test = self.num_validation_cases + self.num_test_cases
		self.num_train_cases = int(self.num_cases - num_validation_test)
		case_idxs = np.arange(int(self.num_cases))
		np.random.shuffle(case_idxs)
		self.validation_idxs = case_idxs[int(self.num_cases - self.num_validation_cases):]
		self.test_idxs = case_idxs[int(self.num_cases - num_validation_test):int(self.num_cases - self.num_validation_cases)]
		self.train_idxs = case_idxs[:int(self.num_cases - num_validation_test)]
		self.train_pointer, self.validation_pointer = 0, 0

	def normalize(self):
		self.energy_mean = np.mean(self.energy_data[self.train_idxs])
		self.energy_std = np.std(self.energy_data[self.train_idxs])

	def train_step(self):
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		num_batches = int(self.num_train_cases/self.batch_size)
		for ministep in range(num_batches):
			batch_data = self.get_train_batch()
			feed_dict = self.fill_feed_dict(batch_data)
			_, energy_loss = self.sess.run([self.train_op, self.energy_loss], feed_dict=feed_dict)
			train_energy_loss += energy_loss
		train_energy_loss /= num_batches
		train_energy_loss /= self.batch_size
		duration = time.time() - start_time
		print("step:", self.step, " duration:", duration, " train loss:", train_energy_loss)
		return

	def validation_step(self):
		start_time = time.time()
		validation_loss =  0.0
		validation_energy_loss = 0.0
		num_batches = int(self.num_validation_cases/self.batch_size)
		validation_energy_labels, validation_energy_outputs = [], []
		for ministep in range (num_batches):
			batch_data = self.get_validation_batch()
			feed_dict = self.fill_feed_dict(batch_data)
			energy_labels, energy_preds, energy_loss = self.sess.run([self.energy_pl,
					self.energy_preds, self.energy_loss],  feed_dict=feed_dict)
			validation_energy_loss += energy_loss
			validation_energy_labels.append(energy_labels)
			validation_energy_outputs.append(energy_preds)
		validation_energy_loss /= num_batches
		validation_energy_loss /= self.batch_size
		validation_energy_labels = np.concatenate(validation_energy_labels)
		validation_energy_outputs = np.concatenate(validation_energy_outputs)
		validation_energy_errors = validation_energy_labels - validation_energy_outputs
		duration = time.time() - start_time
		for i in [random.randint(0, num_batches * self.batch_size - 1) for _ in range(10)]:
			print("Energy label:", validation_energy_labels[i], " Energy output:", validation_energy_outputs[i])
		print("MAE  Energy (eV):", np.mean(np.abs(validation_energy_errors)))
		print("MSE  Energy (eV):", np.mean(validation_energy_errors))
		print("RMSE Energy (eV):", np.sqrt(np.mean(np.square(validation_energy_errors))))
		print("step:", self.step, " duration:", duration, " validation energy loss:", validation_energy_loss)
		return validation_energy_loss

	def test(self):
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		num_batches = int(self.num_test_cases/self.batch_size)
		test_energy_labels, test_energy_outputs = [], []
		batch_data = [self.Z_data[self.test_idxs], self.energy_data[self.test_idxs]]
		feed_dict = self.fill_feed_dict(batch_data)
		test_energy_labels, test_energy_outputs, test_energy_loss = self.sess.run([self.energy_pl,
				self.energy_preds, self.energy_loss],  feed_dict=feed_dict)
		test_energy_loss /= num_batches
		test_energy_loss /= self.batch_size
		test_energy_errors = test_energy_labels - test_energy_outputs
		duration = time.time() - start_time
		for i in [random.randint(0, num_batches * self.batch_size - 1) for _ in range(10)]:
			print("Energy label:", test_energy_labels[i], " Energy output:", test_energy_outputs[i])
		print("MAE  Energy (eV):", np.mean(np.abs(test_energy_errors)))
		print("MSE  Energy (eV):", np.mean(test_energy_errors))
		print("RMSE Energy (eV):", np.sqrt(np.mean(np.square(test_energy_errors))))
		return test_energy_loss, test_energy_errors

	def get_train_batch(self):
		if self.train_pointer + self.batch_size >= self.num_train_cases:
			np.random.shuffle(self.train_idxs)
			self.train_pointer = 0
		self.train_pointer += self.batch_size
		batch_data = []
		batch_data.append(self.Z_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		return batch_data

	def get_validation_batch(self):
		if self.validation_pointer + self.batch_size >= self.num_validation_cases:
			self.validation_pointer = 0
		self.validation_pointer += self.batch_size
		batch_data = []
		batch_data.append(self.Z_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		batch_data.append(self.energy_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		return batch_data

	def fill_feed_dict(self, batch_data):
		pl_list = [self.Zs_pl, self.energy_pl]
		feed_dict={i: d for i, d in zip(pl_list, batch_data)}
		return feed_dict

	def save_checkpoint(self):
		checkpoint_file = os.path.join(self.directory,self.name+'-checkpoint')
		self.saver.save(self.sess, checkpoint_file, global_step=self.step)
		return
