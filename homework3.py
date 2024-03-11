#Citations:
# -	https://www.analyticsvidhya.com/blog/2021/11/study-of-regularization-techniques-of-linear-model-and-its-roles/
# -	https://towardsdatascience.com/understanding-regularization-in-machine-learning-d7dd0729dde5
# -	https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
# -	Added code comments

import argparse
import numpy as np

import sklearn.metrics.pairwise as pairwise

def read_data(filepath):
    Z = np.loadtxt(filepath)
    y = np.array(Z[:, 0], dtype = int)  # labels are in the first column
    X = np.array(Z[:, 1:], dtype =float)  # data is in all the others
    return [X, y]

def save_data(filepath, Y):
    np.savetxt(filepath, Y, fmt = "%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 3",
        epilog = "CSCI 4360/6360 Data Science II",
        add_help = "How to use",
        prog = "python homework3.py -i <input-data> -o <output-file> [optional args]")

    # Required args.
    parser.add_argument("-i", "--infile", required = True,
        help = "Path to an input text file containing the data.")
    parser.add_argument("-o", "--outfile", required = True,
        help = "Path to the output file where the class predictions are written.")

    # Optional args.
    parser.add_argument("-d", "--damping", default = 0.95, type = float,
        help = "Damping factor in the MRW random walks. [DEFAULT: 0.95]")
    parser.add_argument("-k", "--seeds", default = 1, type = int,
        help = "Number of labeled seeds per class to use in initializing MRW. [DEFAULT: 1]")
    parser.add_argument("-t", "--type", choices = ["random", "degree"], default = "random",
        help = "Whether to choose labeled seeds randomly or by largest degree. [DEFAULT: random]")
    parser.add_argument("-g", "--gamma", default = 0.5, type = float,
        help = "Value of gamma for the RBF kernel in computing affinities. [DEFAULT: 0.5]")
    parser.add_argument("-e", "--epsilon", default = 0.01, type = float,
        help = "Threshold of convergence in the rank vector. [DEFAULT: 0.01]")

    args = vars(parser.parse_args())

    # Read in the variables needed.
    outfile = args['outfile']   # File where output (predictions) will be written. 
    d = args['damping']         # Damping factor d in the MRW equation.
    k = args['seeds']           # Number of (labeled) seeds to use per class.
    t = args['type']            # Strategy for choosing seeds.
    gamma = args['gamma']       # Gamma parameter in the RBF kernel
    epsilon = args['epsilon']   # Convergence threshold in the MRW iteration.
    # For RBF, see: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel

    # Read in the data.
    X, y = read_data(args['infile'])

    # Create affinity matrix via RBF kernel from sklearn using X and gamma
    A = pairwise.rbf_kernel(X, gamma=gamma)

    # Calculate the diagonal degree matrix D from the sum of rows in A (subtract epsilon)
    D = np.diag(np.sum(A, axis=1) + epsilon)

    # Find weighted transition probability matrix W
    W = np.zeros(A.shape)
    #Fill in W using Aij/Dij formula
    for i in range(len(X)):
        for j in range(len(X)):
            if D[i, i] != 0:
                W[i, j] = A[i, j] / D[i, i]

    # Create seed vectors (must be able to use RANDOM and DEGREE seed selection)
    # desired strategy defined in t variable
    # Initialize seed vectors list
    seeds = []
    #seed indices:
    seed_indices = []
    # Pull class labels from y matrix, ignore -1 "label"
    unique_labels = [label for label in np.unique(y) if label != -1]

    # Loop through each label and select seeds either randomly or by degree to be used in MRW
    if t == "random":
        for label in unique_labels:
            # Pull indices for corresponding y label
            indices = [i for i, y_label in enumerate(y) if y_label == label]
            # Pick random seed
            class_seed_index = np.random.choice(indices, k)
            # Add to seeds list
            seeds.append(X[class_seed_index])
            # Track seed vector indices
            seed_indices.append(class_seed_index)
    # Loop through class labels and select seeds by highest degree
    elif t == "degree":
        for label in unique_labels:
            # Pull indices for corresponding label
            indices = [i for i, y_label in enumerate(y) if y_label == label]
            # Define class seeds
            class_seeds = X[indices]
            # Pull rows in A corresponding to the selected indices
            selected_A_rows = A[indices]
            # Calculate degrees (sum of rows) for the selected A rows
            degrees = np.sum(selected_A_rows, axis=1)
            # Create a list of tuples pairing degrees with original indices to keep track of degree-index pair
            degrees_with_indices = [(degree, index) for degree, index in zip(degrees, indices)]
            # Sort degrees_with_indices in descending order based on degree values
            degrees_with_indices.sort(reverse=True)
            # Extract the highest degree and its corresponding index in X
            highest_degree, highest_degree_index = degrees_with_indices[0]
            # Use the corresponding index in X to select the row with the highest degree
            highest_degree_seed = X[highest_degree_index]
            # Add the highest degree seed to the seeds list
            seeds.append(highest_degree_seed)
            # Track seed indices
            seed_indices.append(highest_degree_index)

    #Initialize ranking vectors
    ranking_vectors = []
    #Perform iteration through each class
    for label in unique_labels:
        # Create U seed vector with a length equal to rows in X for each class
        U = np.zeros(len(X))
        #Set correspoinding U elements to 1 for specific class
        for i in range(len(y)):
            if y[i] == label:
                U[i] = 1
        # Normalize U so that sum of terms is 1
        normalized_U = U / np.sum(U)
        # Start MRW iterations for R until epsilon convergence threshold is met
        # print(normalized_U)
        # Perform MRW iterations until convergence based on epsilon
        converged = False
        #Initialize ranking vector R (anything other than 0s)
        R = normalized_U 
        #Loop
        while not converged:
            # Perform MRW iteration
            R_new = (1-d) * normalized_U + np.dot(W,R) 
            # Check for convergence based on squared difference
            squared_diff = np.sum((R - R_new) ** 2)
            if squared_diff < epsilon:
                converged = True
                # Append the final ranking vector to the list
                ranking_vectors.append(R_new)
                break
            else:
                #Update R if  threshold not met
                R = R_new
                
    # Generate prediction labels for unlabeled data
    for i in range(len(y)):
        if y[i] == -1:
            #Define number of classes
            number_classes = len(unique_labels)
            #Pull corresponding values from each of the ranking vectors
            ranking_vector_values =  [ranking_vectors[j][i] for j in range(number_classes)]
            #Pull class label from indexed row in the values list
            predicted_label = np.argmax([ranking_vector_values])
            #Assign label in y
            y[i] = int(predicted_label)

    #Save output to "output.npy"
    save_data(outfile, y)