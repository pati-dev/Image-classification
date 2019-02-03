# Image-classification
Predicting the correct orientation of an image using three different Machine Learning algorithms, without using the existing scikit-learn packages.

This program implements and tests several different classifiers: k-nearest neighbors, AdaBoost, and decision forests. It uses rescaled versions of the images where each image was rescaled to a very tiny “micro-thumbnail” of 8 × 8 pixels, resulting in an 8 × 8 × 3 = 192 dimensional feature vector. The text files have one row per image, where each row is formatted like:
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...
where:
• photo id is a photo ID for the image.
• correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be wrong because of noise; this is just a fact of life when dealing with data from real-world sources.
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc., each in the range 0-255.

The training dataset consists of about 10,000 images, while the test set contains about 1,000. For the training set, each image was rotated 4 times to give four times as much training data, so there are about 40,000 lines in the train.txt file.

The program is run like this:
./orient.py train train_file.txt model_file.txt [model]
where [model] is one of nearest, adaboost, forest, or best.
This program uses the data in train file.txt to produce a trained classifier of the specified type, and saves the parameters in model file.txt.
For testing, the program runs like this:
./orient.py test test_file.txt model_file.txt [model]
where [model] is again one of nearest, adaboost, forest, best.
