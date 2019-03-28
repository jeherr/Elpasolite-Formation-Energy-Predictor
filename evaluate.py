import sys
import network_model

def main():
	a = sys.argv[1]
	b = sys.argv[2]
	c = sys.argv[3]
	d = sys.argv[4]
	model = network_model.NNModel(name="ElpasEM_Wed_Mar_27_16.36.09_2019")
	model.evaluate([a, b, c, d])

if __name__ == "__main__":
	main()
