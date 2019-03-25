import sys
import cPickle
import network_model

def main():
	data = str(sys.argv[1])
	data_dict = cPickle.load(open(data, 'rb'))
	model = network_model.NNModel(data=data_dict)
	model.train()

if __name__ == "__main__":
	main()
