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
[-2.5513804, -5.7592616, -1.570019, 0.9597991],
[-6.5345106, -3.9986, -3.7074566, -2.3658702],
[4.8333983, -6.2240086, -1.249202, 2.0486069],
[0.4186223, -6.4531493, -4.4682946, 1.8849711],
[-0.15123788, -5.800419, -3.2940047, 1.1801078],
[-1.6375434, -5.0062633, -1.9952934, 0.09541747],
[-2.6509132, -4.8054466, -2.034238, -0.29223832],
[-2.7602823, -3.929046, -2.438075, -1.407692],
[-2.9046147, -2.915192, -2.8870792, -2.6655185],
[-6.717644, -2.0450015, -4.569536, -4.7079797],
[7.8395, -6.936753, -1.7457044, 2.894855],
[2.0202854, -7.6684084, -3.818847, 0.70305276],
[1.1409408, -5.037931, -2.8132944, 0.47489643],
[0.5356181, -6.216391, -3.0519676, -0.158249],
[-0.1490499, -4.4558043, -3.7785914, 0.43537363],
[-0.30983448, -4.256329, -2.8944316, -0.51772],
[-0.41789976, -3.4629006, -2.7958343, -1.5882387],
[-3.0685222, -1.2776725, -3.5703053, -4.6152663],
[7.193855, -7.6691976, 0.12255091, -0.42516446],
[6.2503614, -7.6132236, -3.0708182, 0.9849971],
[4.856036, -6.2476296, -3.094816, 1.3026085],
[3.4549632, -4.974286, -3.0995164, 1.3669995],
[3.4353325, -4.2819266, -3.1526618, 1.3151109],
[3.5385675, -2.9535408, -3.0437193, 2.0817711],
[2.3185253, -4.131144, -3.7341363, 1.2516202],
[2.2928872, -3.5851743, -3.8784347, 0.9306658],
[2.189301, -2.4109309, -4.183955, 0.39043185],
[2.1007168, -1.7268834, -4.7283397, 0.7649392],
[1.9699631, -0.20511246, -5.9426737, 1.8239678],
[3.3188124, -2.4396393, -7.4300914, 0.2110966],
[2.1510346, -1.6788414, -4.224724, -0.16901238],
[1.5373038, -1.2211224, -5.2692485, -0.027606336],
[1.4919189, -2.3792672, -6.103144, -0.6932203],
[0.74700624, -1.7150668, -4.8752184, -0.87878937],
[0.043626294, -1.5817119, -4.5046287, -1.6363539],
[-0.73498917, 1.1539364, -5.798006, -5.0569677],
[7.8886905, -6.6397185, 0.5447978, -1.1868057],
[6.7827973, -6.5068374, -2.2858627, -1.3052325],
[6.3289514, -5.159576, -2.5104373, -0.37375],
[6.126114, -4.421404, -2.9824169, 0.2491598],
[6.0238175, -2.3379898, -2.8541102, 1.5343702],
[4.908068, -0.8018897, -3.6160288, 1.3200582],
[4.410344, -2.3260436, -3.815118, 0.29119706],
[3.6250403, 0.090955615, -4.029447, 1.0440168],
[3.2576375, 0.2631651, -4.2128134, 1.0686506],
[2.9761908, 1.5848755, -5.985635, 3.0179186],
[2.849603, 0.81420475, -4.84298, 1.0887735],
[3.9495804, -1.3493524, -7.1853623, -0.41072518],
[2.5785937, -1.062681, -3.946417, -0.712771],
[2.2148595, -0.05350085, -4.8197865, -0.7984139],
[2.434557, -2.326516, -4.7613482, -2.3392107],
[2.3947463, -0.9634739, -5.3536787, -3.7211952],
[1.4822152, -0.22049338, -5.5291495, -4.9492083],
[0.8468896, 2.3932886, -6.90869, -7.023838],
[9.346741, -7.415296, 2.2813776, -3.3387027],
[8.314072, -7.4214783, -1.6558774, -3.8060603],
[8.014477, -3.045358, -2.7494197, -2.6347556],
[7.20436, -2.4130063, -3.817134, -2.4804966],
[6.4285126, -1.4230547, -4.279484, -1.7311608],
[7.0993123, -0.65722454, -4.505959, -2.0563915],
[7.172266, 1.0047804, -5.3870664, -4.0892243],
[7.4730906, 2.3568654, -5.492267, -3.8459082],
[7.457717, 2.9055762, -5.7015367, -3.6160223],
[6.4028955, 5.323797, -7.3247213, -2.356566],
[6.0816326, 5.6498637, -7.9215975, -2.1431234],
[6.001079, 2.5042822, -8.102415, -3.8793695],
[5.222268, 3.1952038, -5.6475205, -4.695193],
[4.7440853, 2.7182384, -6.2739787, -4.8635745],
[4.289927, 3.443812, -5.8423595, -5.493875]], dtype=np.float64)

class NNModel(object):
	def __init__(self, data=None, name=None, hidden_layers=[32, 32], batch_size=32):
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
