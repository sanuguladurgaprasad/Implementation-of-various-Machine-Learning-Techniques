# Implementation-of-various-Machine-Learning-Techniques
The repository consists of various ML techniques implemented in MATLAB
# EMG.m (Expectation Maximization Alg)
EMG(flag: a binary indicator variable, image.bmp: file path to an image, k: scalar value of the number of clusters). Function EMG implements the standard EM algorithm if flag is 0, while implements the improved EM algorithm if flag is 1. The function prints to the Matlab workspace and returns in variables the following for a single image and value of k: (1) h: a n × k matrix where n is the number of pixels, (2) m: a k × d matrix where d = 3 denotes the RGB values, and (3) Q: a column vector of expected complete log-likelihood values. The outputs {h, m, Q} are defined in Section 7.4 of the "http://cs.du.edu/~mitchell/mario_books/Introduction_to_Machine_Learning_-_2e_-_Ethem_Alpaydin.pdf". The function also displays: (1) a single compressed color image for a single value of k and (2) a single plot for the expected complete log-likelihood function value vs iteration number for a single value of k.

# myKNN.m (K-NN ALg)
function return error percentage for the given train data, test data and k value

# myPCA.m (Principal Component Analysis)
dataMat is the input data matrix. The function returns the projection matrix 'W' and eigen values matrix 'eigVal'. If 'numComp' is non zero then it returns 'numComp' eigen vectors else it returns all the eigen vectors.

# myLDA.m (Linear Discriminant Analysis)
dataMat is the input data matrix. The function returns the projection matrix 'W' and eigen values matrix 'eigVal' if 'numComp' is non zero then we return 'numComp' eigen vectors else we return all the eigen vectors

# MultiLayer Perceptron with 1 hidden layer
# mlptrain.m
Train a MLP: mlptrain(train data.txt: path to training data file, val data.txt: path to validation data, m: number of hidden units, k: number of output units). The function returns in variables the outputs (z: a n × m matrix of hidden unit values, w: a m×(d+1) matrix of input unit weights, and v: a k × (m + 1) matrix of hidden unit weights). The function must prints the training and validation error rates for the given function parameters.

# mlptest.m
Test a MLP: mlptest(test data.txt: path to test data file, w: a m×(d+1) matrix of input unit weights, v: a k × (m + 1) matrix of hidden unit weights). The function returns in variables the outputs (z: a n × m matrix of hidden unit values). The function also prints the test set error rate for the given function parameters.
