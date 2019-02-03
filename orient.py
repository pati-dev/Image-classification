#!/usr/bin/env python3
# orient.py: The answer to "Which way is up?" for a corpus of vectorized images.
# Ankit Mathur (anmath), Nitesh Jaswal (njaswal), and Nishant Jain (nishjain)
# December 2018

###################################### Report ######################################
### Storing train and test data
# All three techniques listed below read-in the data as two dictionaries:
    #1 Feature data:
        # key: image_name; value: list_of_features (length: 192)
    #2 Label data:
        # key: image_name; value: correct_label (0, 90, 180, 270)

### K-nearest neighbors
# Results:
    # Accuracy: 73.17%  Training time: ~ 9 sec  Testing time: ~2 min
# KNN, being an unsupervised ML technique, does not need any training, and hence can be directly implemented to the testing dataset. However, the key element in the implementation of this technique is to find a way to deal efficiently with the 192-dimensional space storing the vectors in the input training data.
# The initial implementation of our solution computed Manhattan distance of every train image (~40,000 in total) from the test image in a 192-dimensional space and picked the 'k' nearest train images to classify the test image into one of the four orientations. Since there are ~1,000 test images to classify, our initial implementation computed approximately (40,000 * 1,000 * 192 =) 10^9 numbers in total, which took ~60 minutes. :(
# In order to speed up calculations, we tried multiple modifications such as:
    #1 using cosine relation (spatial.distance.cosine) instead of Manhattan distance
    #2 using numpy to carry put vector arithmetic
    #3 using map() along with arithmetic operators from the operator module
# While modifications #1 and #2 failed to improve running time, #3 significantly improved the running time - from ~60 minutes to ~20 minutes. Cheers to map()!
# However, all further efforts to optimize the running time further by modifying how Python carries out vector algebra failed miserably, until we realized we were looking at the wrong place.
# In the next iteration of our implementation, we made a critical design decision: to carry out Principal Component Analysis(PCA) on the 192-dimensional features. As you'll see, PCA has been carried out from scratch using no additional library, apart from numpy that played a key role in simplifying the vector algebra involved in PCA. The PCA implemented in this code makes use of a Python feature that, although subtle, is really crucial for our code to work:
    # Dict keeps insertion order.
# This means that since we are inserting each train and test data point into Python dictionaries one after another, these dictionaries order the keys in this order and then remember this order.
# The aforementioned assumption is critical to the implementation of our solution because the PCA involved follows a 7-step process:
    #1 Extract values (lists) from the dict storing features data from both train and test files
    #2 Store these lists in numpy arrays without meddling with their order
    #3 Normalize train and test features
    #4 Use the normalized train features to compute 192 eigen values
    #5 Pick, say, top 20 of these values and export the corresponding 192 x 20 matrix of eigen vectors in the model file
    #6 Reduce the 192-dimensional features data for both train and test into a 20-dimensional space by taking the dot product of normalized features with the matrix of eigen vectors exported in step #5
    #7 Map each vector in the reduced matrix of features with the dict keys as in step #1
# Thus, PCA takes dicts of 192-dimensions as inputs and returns reduced dicts of 20-dimensions as outputs.
# This implementation substantially reduced the running time - from ~20 minutes to less than 4 minutes (depending on the value of 'k' and number of reduced dimensions('dim'). AWESOME!
# The obvious outstanding question, hence, is what values of 'k' and 'dim' work best for our data? To make a well-informed decision here, we tracked accuracies and running times at different combinations of values of 'k' and 'dim', both ranging from 5 to 50. We observed that KNN performed best for a 20-dimensional space (dim=20) and reached the maximum accuracy of ~73% at k=15. Cherry on the top was that it took ~2 minutes to run! ^o^
# A detailed analysis of the variation of accuracy with 'k' and 'dim' can be seen in the report.pdf file.
### Adaboost
# Results:
    # Accuracy: ~70%  Training time: ~4 min Testing time: ~15 sec
# While implementing Adaboost initially, we decided to use the reduced 20 dimensions that we had gotten from the KNN implementation and then generate all possible pairs of features to compare. However, there was a major gap in our understanding of the technique. Since Adaboost is a binary classification technique, we were supposed to implement 1-vs-all methodology for all four classes or orientations. However, we ended up creating combining all classes into one and then comparing them to learn the best classifier for a given pair of features. Due to this, our accuracy never went beyond 49%, which was way lower than the expected accuracy of ~70% as per the Piazza posts.
# In the next iteration of our implementation, we filled the gaps in our understanding of Adaboost and recreated the entire code for Adaboost from scratch by correctly implementing the 1-vs-all classifiers and randomly generated pairs from the 192-dimensional feature space for each classifier to learn upon. We also used numpy for all matrix calculations that helped us reduce the total time taken for training from ~30 minutes for 200 random samples to ~1 minute! What's better was that our accuracy jumped from 49% in the previous implementation to ~72% in the correct implementation for 6000 random pairs.
# For identifying the right number of random pairs to be generated, we tested our program for multiple numbers of pairs and found out that the accuracy sort of converged to ~70% after 1000 pairs. A detailed analysis of this variation can be seen the report.pdf file.

import sys
import random
import math
import operator as op
import numpy as np
import collections
import traceback
from time import time

def read_data(filename, t_param):
    global size
    print("size =", size)
    data_labels = {}
    data_features = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[:size]:
            values = [ i for i in line.split() ]
            img_name = ( values[0] + '_' + values[1] ) if t_param == 'train' else  values[0]
            label = values[1]
            features = values[2:]

            data_labels[img_name] = label
            data_features[img_name] = []
            for i in range(0, len(features) ):
                data_features[img_name] += [ int(features[i]) ]
    return data_labels, data_features

def normalize(features):
    print("##### Normalizing features...")
    feature_array = np.array( [ feature for feature in features.values() ] )
    feature_means = np.mean(feature_array, axis=0)
    norm_features = np.asmatrix( [ np.subtract(feature, feature_means) for feature in features.values() ] )
    return norm_features

def eig_decomp(features, dim):
    print("##### Decomposing features into eigen vectors...")
    covar = np.cov(features.T)
    eig_values, eig_vecs = np.linalg.eig(covar)
    sorted_indices = eig_values.argsort()[::-1]
    eig_values = eig_values[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]
    eig_vecs = eig_vecs[:, :dim]
    return eig_vecs

def update_features(orig_features, trans_features):
    print("##### Transforming features...")
    i = 0
    mod_features = {}
    for key in orig_features:
        mod_features[key] = trans_features[i].real.tolist()[0]
        i += 1
    return mod_features

def write_model_knn(model_file, k, dim, eig_vecs, mod_train_features, train_labels):
    print("##### Writing data into the model file...")
    with open(model_file, 'w') as outfile:
        print(k, dim, file=outfile)
        for vec in eig_vecs:
            for val in vec:
                print(val, file=outfile, end=' ')
            print('', file=outfile)
        for key, val in mod_train_features.items():
            print(key, train_labels[key], file=outfile, end=' ')
            for item in val:
                print(item, file=outfile, end=' ')
            print('', file=outfile)
    return None

def read_model_knn(model_file):
    print("##### Reading data from the model file...")
    with open(model_file, 'r') as infile:
        lines = infile.readlines()

        k, dim = list( map(int, lines[0].split() ) )

        eig_vecs = [ lines[l].split() for l in range(1, 193) ]
        eig_vecs = np.array( [ [ complex( item.rstrip("'") ) for item in vec ] for vec in eig_vecs ] )

        train_labels = {}
        mod_train_features = {}
        for line in lines[193:]:
            values = [ i for i in line.split() ]
            img_name = values[0]
            label = values[1]
            features = values[2:]

            train_labels[img_name] = label
            mod_train_features[img_name] = []
            for i in range(0, len(features) ):
                mod_train_features[img_name] += [ float(features[i]) ]
    return k, dim, eig_vecs, train_labels, mod_train_features

def predict_knn(mod_test_features, mod_train_features, train_labels):
    print("##### Predicting test data...")
    with open('output.txt', 'w') as outfile:
        print('', end='', file=outfile)
    counter = 0
    pred_labels = {}
    for test_img, test_features in mod_test_features.items():
        counter += 1
        dist = []
        img = []
        for train_img, train_features in mod_train_features.items():
            dist += [ sum( map(abs, map(op.sub, test_features, train_features) ) ) ]
            img += [train_img]
        dist, img = np.array(dist), np.array(img)
        sorted_indices = dist.argsort()
        dist, img = dist[sorted_indices], img[sorted_indices]
        nn = img[:k]
        nn_labels = [ train_labels[neighbor] for neighbor in nn ]
        freq = collections.Counter(nn_labels)
        label = freq.most_common(1)[0][0]
        pred_labels[test_img] = label
        # write result to output file
        with open('output.txt', 'a') as outfile:
            print(test_img, label, file=outfile)
        print(counter, "images predicted...")
    return pred_labels

def compute_accuracy(test_labels, pred_labels):
    correct_count = 0
    tot_count = len(test_labels)
    for test_img, test_label in test_labels.items():
        correct_count += 1 if pred_labels[test_img] == test_label else 0
    acc = round( correct_count/tot_count*100, 2 )

    return acc

def getRandomPairs():
    global n
    print("n =", n)
    return [ list( np.random.choice(192, 2) ) for i in range(n) ]

def getRandomFeaturePairs(n_features, tree_depth):
    selected_features = np.random.choice(192, n_features, replace=False).tolist()
    feature_pairs = []
    i = 0
    while(i < tree_depth):
        pair = np.random.choice(selected_features, 2, replace=False).tolist()
        # import pdb; pdb.set_trace()
        if(pair not in feature_pairs):
            feature_pairs += [pair]
            i += 1
    return(selected_features, feature_pairs)

def sliceData(n_data_slice, train_features, train_labels, selected_features):
    x = 38000 - n_data_slice
    start_point = np.random.choice(x, 1)[0]
    features = np.array( [ feature for feature in train_features.values() ] )
    labels = np.array( [ label for label in train_labels.values() ] )
    features = features[start_point:start_point+n_data_slice, :]
    labels = labels[start_point:start_point+n_data_slice]
    # print("X:", x, "Size: ", features.shape[0])
    return(features, labels)

def forestCalcEntropy(n_total, labels_predict):
    n_i = np.sum(labels_predict.astype(int))
    # Adding a very small number in the log to prevent math domain error if encountered
    return(-(n_i/n_total)*math.log((n_i/n_total + 10**-18), 2) + -((n_total - n_i)/n_total)*math.log(((n_total - n_i)/n_total + 10**-18), 2))

def forestCalcWeight(my_forest, n_trees, n_data_slice):
    wt_trees = np.zeros([n_trees, 1])
    for i in range(n_trees):
        sum = 0
        for node in my_forest[i].values():
            # import pdb; pdb.set_trace()
            sum += np.sum(node[3])
        wt_trees[i] = sum/n_data_slice
    # Normalize weights
    wt_trees = np.multiply(wt_trees, 1/np.sum(wt_trees))
    return(wt_trees)

def getDecisionTree(feature_pairs, features, labels, tree_depth):
    decision_tree = {}
    labels = labels.astype(int)
    labels = np.reshape(labels, [features.shape[0], 1])
    node_input_data_filter = np.ones([features.shape[0], 1], dtype='int')
    node_input_data, filtered_labels = features, labels
    for i in range(0, tree_depth):
        pair_entropy = []
        # Filter the data received by the next node in the decision tree based on the output of previous node
        # Adding a constant so that values pf pixel intensity that are already 0 are not removed
        # import pdb; pdb.set_trace()
        node_input_data = np.multiply(node_input_data + 1, node_input_data_filter)
        # Removing the rows and the added constant
        node_input_data = np.ma.compress_rows(np.ma.masked_equal(node_input_data, 0)) - 1
        filtered_labels = np.multiply(filtered_labels + 1, node_input_data_filter)
        filtered_labels = np.ma.compress_rows(np.ma.masked_equal(filtered_labels, 0)) - 1
        for pair in feature_pairs:
            node_output = (node_input_data[:,pair[0]] > node_input_data[:,pair[1]]).astype(int)
            # Reshape to keep shape consistent i.e. (N,1) instead of (N,)
            node_output = np.reshape(node_output, [node_input_data.shape[0], 1])
            min_entropy, min_entropy_orient, min_entropy_predict = math.inf, '', []
            for orientation in [0, 90, 180, 270]:
                node_output_label = np.where(node_output==1, orientation, -1)
                labels_predict = (filtered_labels == node_output_label)
                entropy = forestCalcEntropy(node_input_data.shape[0], labels_predict)
                (min_entropy, min_entropy_orient, min_entropy_predict) = (entropy, orientation, labels_predict) if entropy < min_entropy else (min_entropy, min_entropy_orient, min_entropy_predict)
            pair_entropy += [(min_entropy, min_entropy_orient, pair, min_entropy_predict)]
        min_entropy_node = min(pair_entropy)
        decision_tree[i] = min_entropy_node
        # Generate mask to filter input data for next node
        node_input_data_filter = np.logical_not(min_entropy_node[3]).astype(int)
        # Exclude the parent node from re-occouring as a child
        feature_pairs.remove(min_entropy_node[2])
    return(decision_tree)

def adaboost_stumps(train_labels, train_features):
    print("##### Learning decision stumps...")
    pairs = getRandomPairs()
    features = np.array( [ feature for feature in train_features.values() ] )
    labels = np.array( [ label for label in train_labels.values() ] )
    wt_stumps = {}
    pairs_stumps = {}

    for orientation in ['0', '90', '180', '270']:
        # wt_stumps is a hash that stores the weight of each learner (for a pair of features) for each orientation
        wt_stumps[orientation] = []
        # pairs_stumps is a hash that stores the pairs of features that need to be compared for each learner for each orientation
        pairs_stumps[orientation] = []
        # create a vector to store the orientation being learned
        labels_vec = np.array( [orientation] * len(labels) )
        # create a boolean vector to store whether a training image has the orientation being learned
        orient_labels = labels == labels_vec
        # initialize uniformed weights for each image
        wt_exemplars = [ 1/len(train_features) for feature in train_features ]
        for pair in pairs:
            # create a boolean vector (bool) to store the hypothesis
            diff = features[ :, pair[0] ] - features[ :, pair[1] ]
            bool = diff > 0
            # map the boolean vector (bool) with the correct orientation of each image
            correct_vec = bool == orient_labels
            # store the sum of weights of correctly classified exemplars
            correct_sum = sum( np.where( correct_vec, wt_exemplars, [0]*len(wt_exemplars) ) )
            # compute error
            error = 1 - correct_sum
            # compute the weight of stump
            wt_stump = math.log(correct_sum/error, 10)
            # check if the weight of stump is positive
            if wt_stump > 0:
                # store the weight of the stump in the wt_stumps var and the corr pair in pairs_stumps var
                wt_stumps[orientation] += [wt_stump]
                pairs_stumps[orientation] += [pair]
                # adjust weights of correct exemplars
                reduced_wt = np.multiply( wt_exemplars, [error/correct_sum]*len(wt_exemplars) )
                wt_exemplars = np.where(correct_vec, reduced_wt, wt_exemplars)
                # normalize exemplars weights
                s = [sum(wt_exemplars)] * len(wt_exemplars)
                wt_exemplars = np.divide( wt_exemplars, s )

    return wt_stumps, pairs_stumps

def write_model_adaboost(model_file, wt_stumps, pairs_stumps):
    print("##### Writing data into the model file...")
    with open(model_file, 'w') as outfile:
        for key, value in wt_stumps.items():
            print(key, file=outfile, end=' ')
            for i in range( len(value) ):
                print(pairs_stumps[key][i][0], pairs_stumps[key][i][1], value[i], file=outfile, end=' ')
            print('', file=outfile)
    return None

def read_model_adaboost(model_file):
    print("##### Reading data from the model file...")
    wt_stumps = {}
    pairs_stumps = {}
    with open(model_file, 'r') as infile:
        for line in infile:
            data = line.split()
            label = data[0]
            wt_stumps[label] = []
            pairs_stumps[label] = []
            for i in range(1, len(data[1:]), 3):
                p1, p2, w = data[i:i+3]
                wt_stumps[label] += [ float(w) ]
                pairs_stumps[label] += [ [ int(p1), int(p2) ] ]
    return wt_stumps, pairs_stumps

def predict_adaboost(test_features, wt_stumps, pairs_stumps):
    print("##### Predicting test data...")
    with open('output.txt', 'w') as outfile:
        print('', end='', file=outfile)
    cntr = 0
    pred_labels = {}
    for test_img, test_features in test_features.items():
        cntr += 1
        outputs = {}
        for key, pairs in pairs_stumps.items():
            output, counter = 0, 0
            for pair in pairs:
                p1, p2 = pair
                w = wt_stumps[key][counter]
                result = 1 if ( test_features[p1] - test_features[p2] ) > 0 else -1
                output += w * result
                counter += 1
            outputs[key] = output
        label = max(outputs, key=lambda x: outputs[x])
        pred_labels[test_img] = label
        # write result to output file
        with open('output.txt', 'a') as outfile:
            print(test_img, label, file=outfile)
        print(cntr, "images predicted...")
    return pred_labels


def write_model_forest(model_file, my_forest, wt_trees):
    print("##### Writing data into the model file...")
    with open(model_file, 'w') as outfile:
        for tree_num, tree in my_forest.items():
            print(tree_num, wt_trees[tree_num,0], file=outfile, end='\n')
            for node_num, node in tree.items():
                print(node_num, node[1], node[2][0], node[2][1], file=outfile, end='\n')
            print('', file=outfile)
    return None

def read_model_forest(model_file):
    print("##### Reading data from the model file...")
    my_forest, wt_trees = {}, []
    idx_tree = 0
    with open(model_file, 'r') as infile:
        # import pdb; pdb.set_trace()
        for line in infile:
            data = line.split()
            if not data:
                continue
            elif(len(data) == 2):
                idx_tree = int(data[0])

                wt_trees += [float(data[1])]
                my_forest[idx_tree] = {}
            else:
                pair = [float(data[2]), float(data[3])]
                # import pdb; pdb.set_trace()
                my_forest[idx_tree][int(data[0])] = (int(data[1]), pair)
    wt_trees = np.array(wt_trees)
    wt_trees = np.transpose(np.reshape(wt_trees, [wt_trees.shape[0], 1]))
    return (my_forest, wt_trees)


def predict_forest(test_features, my_forest, wt_trees):
    features = np.array( [ feature for feature in test_features.values() ] )
    n_trees = len(my_forest.keys())
    tree_depth = len(my_forest[0].keys())
    for i in range(n_trees):
        node_input_data_filter = np.ones([features.shape[0], 1], dtype='int')
        node_input_data = features
        for j in range(tree_depth):
            try:
                node_input_data = np.multiply(node_input_data + 1, node_input_data_filter)
                # Removing the rows and the added constant
                node_input_data = np.ma.compress_rows(np.ma.masked_equal(node_input_data, 0)) - 1
                orientation = my_forest[i][j][0]
                pair = my_forest[i][j][1]
                # import pdb; pdb.set_trace()
                node_output = (node_input_data[:,pair[0]] > node_input_data[:,pair[1]]).astype(int)
                # Reshape to keep shape consistent i.e. (N,1) instead of (N,)
                node_output = np.reshape(node_output, [node_input_data.shape[0], 1])
                node_output_label = np.where(node_output==1, orientation, node_output)
                # Generate mask to filter input data for next node
                node_input_data_filter = np.logical_not(node_output).astype(int)
            except Exception as e:
                print(traceback.format_exception(*sys.exc_info()))
                # import pdb; pdb.set_trace()

    return(None)
def knn():
    global k, dim
    if t_param == 'train':
        print("\n########## Training K-nearest Neighbors ##########\n")
        print("k =",k)
        print("dim =",dim)

        print("##### Reading train data...")
        # read and store train data
        train_labels, train_features = read_data(t_file, t_param)

        # carry out eigen decomposition of the train features and transform them
        norm_train_features = normalize(train_features)
        eig_vecs = eig_decomp(norm_train_features, dim)
        trans_train_features = norm_train_features.dot(eig_vecs)

        # update features of train data to store transformed data
        mod_train_features = update_features(train_features, trans_train_features)

        # write out the parameters (k and dim), eigen vectors, and updated train data into the model file
        write_model_knn(model_file, k, dim, eig_vecs, mod_train_features, train_labels)

    elif t_param == 'test':
        print("\n########## Testing K-nearest Neighbors ##########\n")

        # read the model file to obtain the parameters (k and dim), eigen vectors, and updated train data
        k, dim, eig_vecs, train_labels, mod_train_features = read_model_knn(model_file)

        # read and store test data
        print("##### Reading test data...")
        test_labels, test_features = read_data(t_file, t_param)

        # normalize test features and transform as per the eigen vectors calculated previously
        norm_test_features = normalize(test_features)
        trans_test_features = norm_test_features.dot(eig_vecs)

        # update features of test data to store transformed data
        mod_test_features = update_features(test_features, trans_test_features)

        # predict test datapoints
        pred_labels = predict_knn(mod_test_features, mod_train_features, train_labels)

        # compute accuracy
        acc = compute_accuracy(test_labels, pred_labels)
        print("\nAccuracy:", acc, "%")

        # t = round( (time()-t0), 2)
        # with open('output-file.txt', 'a') as outfile:
        #     print(size, acc, t, file=outfile)
    else:
        print("Incorrect parameter.")
        return None

    return None

def adaboost():
    if t_param == 'train':
        print("\n########## Training Adaboost ##########\n")
        # read and store train data
        print("##### Reading train data...")
        train_labels, train_features = read_data(t_file, t_param)
        # learn decision stumps
        wt_stumps, pairs_stumps = adaboost_stumps(train_labels, train_features)
        # write out the parameters into the model file
        write_model_adaboost(model_file, wt_stumps, pairs_stumps)

    elif t_param == 'test':
        print("\n########## Testing Adaboost ##########\n")
        wt_stumps, pairs_stumps = read_model_adaboost(model_file)
        # read and store test data
        print("##### Reading test data...")
        test_labels, test_features = read_data(t_file, t_param)
        # predict test datapoints
        pred_labels = predict_adaboost(test_features, wt_stumps, pairs_stumps)
        # compute accuracy
        acc = compute_accuracy(test_labels, pred_labels)
        print("\nAccuracy:", acc, "%")

        # t = round( (time()-t0), 2)
        # with open('output-file.txt', 'a') as outfile:
        #     print(size, acc, t, file=outfile)

    else:
        print("Incorrect parameter.")
        return None
    return None

# Design Decision: Slice a continuous part of the dataset because we want one decision tree to
# encounter every possible orientation of a particular image
def forest():
    if t_param == "train":
        my_forest = {}
        n_trees = 5
        n_features = 6
        tree_depth = 5
        n_data_slice = 20000

        train_labels, train_features = read_data(t_file, t_param)
        print("Training Model...")
        for i in range(0, n_trees):
            (selected_features, feature_pairs) = getRandomFeaturePairs(n_features, tree_depth)
            (features, labels) = sliceData(n_data_slice, train_features, train_labels, selected_features)
            my_forest[i] = getDecisionTree(feature_pairs, features, labels, tree_depth)
        wt_trees = forestCalcWeight(my_forest, n_trees, n_data_slice)
        # Write data to model file
        print("Training Done")
        write_model_forest(model_file, my_forest, wt_trees)
    elif t_param == "test":
        test_label, test_features = read_data(t_file, t_param)
        (my_forest, wt_trees) = read_model_forest(model_file)
        predict_forest(test_features, my_forest, wt_trees)
        print("Forest: \n",my_forest)
        print("Weights: \n",wt_trees)
        # import pdb; pdb.set_trace()

    return None

def main(t_param, t_file, model_file, model):
    if model == 'nearest':
        knn()
    elif model == 'adaboost':
        adaboost()
    elif model == 'forest':
        forest()
    elif model == 'best':
        knn()
    else:
        print("Unknown model.")

t0 = time()

global k, dim, n, size

t_param, t_file, model_file, model = sys.argv[1:5]

# with open('output-file.txt', 'w') as outfile:
#     print("size acc t", file=outfile)

if __name__ == '__main__':
    # initialize size of training data
    size = 40000

    # initialize parameters for KNN
    k = 15  # nearest neighbors
    dim = 20    # number of eigen vectors to be chosen for reduction

    # initialize parameter for Adaboost
    n = 6000

    main(t_param, t_file, model_file, model)

    # print("\nTime taken: <", round( (time()-t0)/60, 0), "minutes" )
    print("\nTime taken:", round( (time()-t0), 2), "seconds" )
else:
#     for i in range(20, 51, 5):
#         k = i
#         for j in range(5, 51, 5):
#             dim = j
#             t0 = time()
#             for t_param, t_file in zip( ['train', 'test'], ['train-data.txt', 'test-data.txt'] ):
#                 main(t_param, t_file, model_file, model)
#             print("\nTime taken: <", round( (time()-t0)/60, 0), "minutes" )

    # initialize parameters for KNN
    k = 15  # nearest neighbors
    dim = 20    # number of eigen vectors to be chosen for reduction

    # initialize parameter for Adaboost
    n = 1000

    for i in range(4000, 40001, 4000):
        size = i
        t0 = time()
        for t_param, t_file in zip( ['train', 'test'], ['train-data.txt', 'test-data.txt'] ):
            main(t_param, t_file, model_file, model)
            print("\nTime taken:", round( (time()-t0), 2), "seconds" )
