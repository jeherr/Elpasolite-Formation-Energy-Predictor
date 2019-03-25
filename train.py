import sys
import pickle
import network_model

def main():
	data = str(sys.argv[1])
	if sys.version_info.major > 2:
		data_dict = pickle.load(open(data, 'rb'), encoding='latin1')
	else:
		data_dict = pickle.load(open(data, 'rb'))
	model = network_model.NNModel(data=data_dict)
	model.train()

if __name__ == "__main__":
	main()
