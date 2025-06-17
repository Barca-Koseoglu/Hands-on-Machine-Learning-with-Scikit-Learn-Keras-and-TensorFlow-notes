# Classification

Classification is used to classify. So A or B. Yes or No. True or false. Good, bad, or average. 1,2,3,4,5,6,7,8,9, or 10. Regression is to predict a value. That's the difference.

## Datasets

You can use:

```python

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

fetch_openml gives datasets, for example the MNIST dataset which is a dataset of a bunch of pictures of hand written numbers.

Generated datasets are usually returned as an (X, y) tuple containing the input data and the targets, both as NumPy arrays. Other datasets are returned as sklearn.utils.Bunch objects, which are dictionaries whose entries can also be accessed as attributes.

By default, fetch_openml gives a pandas DataFrame, but since the MNIST dataset has images, it's better to set as_frame=False to get the data in a NumPy array instead. Each image is 28x28 pixels btw.

Remember, always create a test set and set it way aside.

## Binary classifiers

One model you can use it stochastic gradient descent classifier, or SGDClassifier. It is very good with large datasets and online learning. The way it works is that for a bunch of models, by default an SVM (the models can be picked by setting the loss function to things like "hinge_loss" or "log_loss"), and instead of doing whatever it did to optimize the loss function, it uses stochastic gradient descent, which is just gradient descent except it pick a bunch of random values.

## Performance evaluation

Oh boy.

Evaluating classifiers is normally more tricky than evaluating regression. :(

A good way to evaluate a model in general is to use cross-validation. Using the cross_val_score() function, we can see the accuracy of the model. Imagine, instead of figuring out 10 seperate numbers, we just want to see if a number is 5 or not. Running this model with this purpose:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

We get about 95% accuracy (ratio of correct predictions to all predictions) on average for all the splits. Woah, did we just do THAT well right off the bat?

Hold your horses, buddy, my sole purpose is to ruin your hopes and dreams.

A DummyClassifier is a classifier that just classifies every image to the most frequent class. So, in this case, non-fives. The accuracy of this model across all cross-validation splits... is 90% for everything... since only about 10% of the pictures are 5s, it guarantees to be right 90% of the time when TRAINING. It will do HORRENDOUSLY if ever given a five.

This is why accuracy is generally not preferred to evaluate performance, especially when dealing with **skewed** datasets, i.e. when some classes are much more frequent than others. A better way to evaluate it is to look at the **confusion matrix** (CM).

### Confusion matrixes

The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B, for all A/B pairs. For example, to know the number of times the classifier confused images of 8s with 0s, you would look at row #8, column #0 of the confusion matrix.

To compute it, you need a set of predictions to actually be compared to the targets. Using the test set is not good practice, so it's better to just use the cross_val_predict function and store it as a variable. Just like cross_val_score, it performs k-fold cross-validation but instead it returns the predictions rather than the score. Now we get clean (meaning out-of-sample, a good thing) predictions to use. By the way, it returns the predictions for each thing not three different sets of predictions.

![image](https://github.com/user-attachments/assets/50a9db7b-29d4-4553-be8a-0065442750ac)

This is what it looks like. By default, each row and column is sorted in ascending order. 687 were false positives, 1891 were false negatives (slightly more severe), 53892 were true positives, and the rest are true negatives. A perfect classifier would only have non-zero values from the top left to the bottom right, aka it's main diagonal.

### Precision

Confusion matrices give us lots of information, but sometimes, we wan't more concise, straightforward pieces of info. That's why some people like using the accuracy of positive predictions: (true positives)/((true positives)+(false positives)). This is called precision.

A stupid way to have perfect precision is to make a classifier that always makes negative predictions except for a single predictions it's extremely confident about. If that one prediction is correct, then the classifier has 100% precision. That's why, people who like useful information tend to also use **recall**.

### Recall

Recall, also called sensitivity or the 'true positive rate (TPR)' is the ratio of positive instances that are correctly detected by the classifier, so (true positives)/((true positives)+(false negatives))
