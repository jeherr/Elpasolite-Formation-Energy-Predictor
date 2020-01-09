# Elpasolite Formation Energy Predictor
Neural network model to predict the formation energy for elpasolites of the crystal formula ABC<sub>2</sub>D<sub>6</sub>.

## Prerequisites
Required packages
```
numpy
tensorflow
```

## Getting Started
To download this project run the following command
```
git clone https://github.com/jeherr/Elpasolite-Formation-Energy-Predictor.git
```
ElpasEM_Thu_Mar_21_13.16.26_2019 contains an already trained network model which can be used to make predictions for any combination of main group elements up to Bismuth at any of the crystal lattice positions. To predict the formation energy (in eV) for any elpasolite run
```
python evaluate.py A B C D
```
where A, B, C, and D are integers corresponding to the desired atomic number at each crystal lattice position. For example
```
python evaluate.py 13 11 19 9
```
will print the prediction for the prototypical elpasolite AlNaK<sub>2</sub>F<sub>6</sub>.

## Training your own model
To train your own model run
```
python train.py train_set.pkl
```
which uses the same hyperparameters laid out in the paper and stops after 1000 epochs. The model is evaluated on the validation set after every 5 epochs and prints out a random sample of 10 true formation energies and the corresponding predictions, along with mean absolute errors, mean signed errors, and root mean square errors over the validation set. Only saves a new checkpoint if the evaluation loss is lower than the last saved checkpoint. After training finishes, the best checkpoint is reloaded and the errors are evaluated over the test set. Trained model create a new directory with the data and time the model was started. To use your own model with evaluate.py, replace the default model directory with the directory for your newly trained model in the following line.
```
model = network_model.NNModel(name="ElpasEM_Thu_Mar_21_13.16.26_2019")
```

## Citing this work
A publication for citation available at https://doi.org/10.1063/1.5108803.


## Acknowledgments
Thanks to the work linked below for the data set of elpasolites and formation energies used here to train the model.
https://doi.org/10.1103/PhysRevLett.117.135502
