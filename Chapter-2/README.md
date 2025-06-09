# End-to-End Machine Learning Project

This checklist can guide you through your Machine Learning projects. There are eight main steps:

1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to gain insights.
4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.
5. Explore many different models and shortlist the best ones.
6. Fine-tune your models and combine them into a great solution.
7. Present your solution.
8. Launch, monitor, and maintain your system

## Data pipilines:

Sequences of data processing components. They run asynch, each part takes in data and gives it to the next. They're mostly self-contained, meaning only data is transfered between them no other funny business. If part of it breaks, the downstream flow should be chill until it gets fixed.

## A performance measure: RMSE

Root mean squared error measures the typical error (actual minus prediction) by giving a heavier emphasis on large, outlying errors, "punishing" the system for it.

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

Squaring it is good for both doing the punishing and dealing with negatives.

#### Reminder to self in case I forget

to start up jupyter notebooks just do "python -m jupyterlab", remember not to pluarlize.

### Ploting some data using a histogram for the whole dataset:

dataframe.hist(bla bla bla)

plt.show()

### train-test-split

Splits up your data into training and testing data:

train_set, test_set = train_test_split(dataframe, test_size=0.2 (sets test set size as a percent), random_state=42 (for consistent results))

Although this is normally good, stratified sampling is where it's at. Statified sampling is when you sample by dividing a dataset into equally represented subgroups when creating a sample, for example the test set. A really good example of this is when you take the data of 100 americans and the general american population is 51% female and 49% male. Your sample should have that ratio of female and male samples. These subsets are called strata. So here's a more solid example: income. Let's say there's like five different groups of different percentiles, from first to 20th, 21st to 40th, etc. the first to 20th would be considered a stratum. So to do stratified sampling, define the strata, get their population ratios right, then use them in your test group. Or just use the stratified sampling method.

Like this:
![image](https://github.com/user-attachments/assets/9d6fae90-eda6-41e5-a913-5342f0e7a7f5)

You can also specify the stratification of a column in the train_test_split function

### Geographcial data visualization: 
```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2 (controls opacity))

plt.show()
```
### Correlation: 

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }
         { \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \cdot \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

How correlated two things are. Not something crazy but it will guide us in our endeavors. It's normalized covariance. Basically covariance (the measure of if something is positively or negatively related) divided by the standard deviation of X times the standard deviation of Y. We divide because it normalizes, as mentioned. Like for height and weight, we divide by the spread of the data to bring it from a scaled value, like cm, to just a unit value. It's kind of weird to explain but it works.

This might miss out on non-linear relationships though.

### Scatter matrix

scatter_matrix(dataframe[attributes])
plt.show()

Makes a len(attributes) x len(attributes) matrix of correlations.

#### FYI: change up attribute combinations

If you have rooms per block, turn that into rooms per house. That's definetly more informative than rooms per block.

## Data cleaning

If there's missing data you can delete the indexes, delete the columns, or fill it in with something, called **imputation**.

Use the SimpleImputer class by setting the strategy. The benefit yo just notmally imputing is that it stores the values of whatever you picked for each feature, like the mean. You can use it on a bunch of other sets this way and it wont compute the set's own mean. .fit() gives it the values to store, .transform() actually changes the values of a set.

### For text and categorical things

Machines don't understand things like POSITIVE or NEGATIVE, so we prefer to convert them into numbers. We can use something called OrdinalEncoder, but machines learning algorithms might think theres a relationship between the distances of the numbers. Like if there's five categories, it might realize number 5 and 1 are further apart than 3 and 4. It might be fine for things like good average bad and great, but it's bad for other things. One solution to this is to use OneHotEncoder(). This helps because it turns the values into binary representations, where there's a bunch of zeroes and only one 1 to represent a specific value. So now it might be able to understand "oh, the houses NEAR the sea are more expensive than the houses INLAND. They're NOT similar to each other just because their values are close." Once again kinda hard to explain. The 1 is 'hot' and the zeroes are 'cold'. It's also a sparse matrix. One Hot Econder remembers things too.

If theres a LOT of possible categories, don't use onehotencoder. Instead, use some other encoder.

## Feature Scaling and transformations

Without scaling, most models will be biased toward ignoring the median income and focusing more on other less important things. There's two common ways to get all attributes to have the same scale: minimax scaling and standardization.

minimax scaling (aka normalization) is very simple: take x as the thing you want to scale from the dataset X. Scale it by doing (x - min(x))/(max(x) - min(x)). This works because if x is the minimum, it's zero, and if x is the max, the denominator and numerator are the same, so x is 1. Anything between the min and max is between 0 and 1. Very smart but easy way to scale values, although definetly prone to problems like large outliers skewing results.

Code for it:
```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
```
Now, here it shows it from -1 to 1. This is found using the general formula for minmax (I said minimax a lot confusing it for something else, the correct wording is MIN-MAX) which is the formula to scale it from [a,b]: a + ((x-min(x))*(b-a))/(max(x)-min(x)). It's pretty intuitive once again, using very simple variable manipulation.


### Standardization

Standardization first subtracts the mean value (so standardized values have a zero mean), then it divides the result by the standard deviation (so the values have standard deviation equal to 1: literally unit spread). This technique doesn't restrict values to a specific range, but it's much less affected by outliers. Imagine we have a datset of values from 0-15, but we accidentaly get 100 in there. Min-max would make that value equal to 1 and the rest of the value from 0 to 0.15, but standardization doesn't get affected as much as this.

Code:
```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
```
### Heavy tails

Weird subtitle, but very very VERY important topic. When a feature's distribution has a heavy tail (aka when the values far from the mean aren't exponentially rare), both min-max and standardization will scale most values to a small range. Before we scale the values, we should transform them to get rid of that heavy tail, and try to make the distributions kinda symmetrical. A common way to do this is to replace the feature with it's square root (raise it to a power between 0 and 1 if you want more or less power than that). If it has a VERY long and heavy tail, like a dinosaurs, then replace the feature with it's logarithm. The best way to understand this is to visualize it:

![image](https://github.com/user-attachments/assets/a94a7d71-8e37-42f1-ada2-016b22f96414)

Another approach is to *bucketize* the feature. This means chopping the distribution to roughly equal-sized buckets and replacing each feature value with the index of the bucket it belongs to, kind of similar to stratified sampling.

Here's a very simple way to understand it, according to the GOAT, chatgpt: 

![image](https://github.com/user-attachments/assets/8da23b04-61ad-425f-a008-b85fcf0ffe17)

When a feature has multimodel distrbution (two or more peaks in the data, called modes), it can be helpful to bucketize it, but treat thebuckets as categories rather than numerical values. You can use OneHotEncoder to encode te indices of these buckets. This now allows models to understand more easily different rules for different ranges of this feature value. For example, maybe for some house ages, houses built around 35 years ago are weird and ugly, so they're cheaper than most other houses.

### The worst thing this book has tried explaining since chapter 1: the radial basis function

One more way to approach a multimodel distribution is to add a feature for each of the modes (or at least the main ones) that represent the **similarity** between that node and the other value points. This is typically computed using a radial basis function (RBF), which is any function that depends on the distance between a fixes point and an input. Normally, Gaussian RBF is used. This is what it looks like:

$$
K(x, x') = \exp(-\gamma \|x - x'\|^2)
$$

Where x` is the fixed point, x is your input, the y looking thing basically controls if the definition of closeness is more strict (higher y) or more loose (smaller y), the swquaring is there to "penalize" higher values and prevent negatives, we negate y to make it into a fraction for exp(), which is just "e to the power of whatever's in the parentheses". There we go.

An example of using could be when you want to find the similarity of other points with the median house value with some other houses. If that age group is correlated with lower prices, the feature might actually help.

![image](https://github.com/user-attachments/assets/0a5b2dc8-eaab-4cc0-808a-91ca05b05c0f)

The lines are the rbf functions plotted, with each corresponding to a different value of gamma, the y looking thing.

A machine learning engineer and data scientist must have the ability to figure out relavent and important data from other bits of data. The models are just a small part of their jobs. I would know. I am a machine learning engineer. That makes 500K a year.

In my dreams.

## Target value manipulation

Target value might also need to be transformed and changed, like if they have a heavy tail, we could choose to replace it with its logarithm. But... wouldn't we be predicting the logs of the target values, not the actual target values?

Precisely!

To revert this, you have to exponentiate the prediciton. The transformers of Skitlearn, luckily, do have an inverse_transform() method that makes it easy to comput the inverse. Doing everything manually is the bane of everyone's existence, though, so instead there's something called TransformedTargetRegressor, where you just construct it and give it the regression model you're using with the original labels. It'll automatically scale the labels, train the model on those labels, then revert the predicitons to the original scale.

Like this:
```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
```
#### A quick note on SciKit-Learn's design

All objects share very consistent and simple intefaces, whic makes it easy to use.

**Estimators**: Anything that can estimate some parameters based on a dataset is called an estimator, for example, SimpleImputer. The estimation itself is done using .fit() and it takes a dataset as a parameter (two for supervised learning since it needs the labels). Any other parameter is considered a hyperparameter, like the strategy for SimpleImputer.

**Transformers**: Some estimators can also transform a dataset. They're called transformers. Transformation is done by the .transform() method with the dataset to transform as a parameter. Transformation generally relies on learned parameters, as is the case with SimpleImputer. Every transformer has a convenient .fit_transform() method, which is the same as calling them step by step except it might be faster.

**Predictors**: Some estimators can make predictions. They're called predictors. Linear Regression is a predictor. It has a .predict() method that takes a dataset of new instances and returns its predictions. .score() is used to measure the quality of the predictions.

### Custom Transformers

Skitlearn does give you a bunch of useful transformers, but, as is everything, you cannot be reliant on others. You have to write your own.

For transformers that don't need training, you can just write a function that takes a NumPy array as input and outputs the transformed array.
```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

log_pop = log_transformer.transform(X)
```
Inverse_func is optional but you can use it in case you plan to use it somewhere, like in TransformedTargetRegressor.

You can also modify hyperparameters using kw_args.

Custom transformers are also useful when combining features, like taking the ratios between input features.

### Trainable transformers

FunctionTransformer is cool and all, but what if we wanted it to be trainable? For this, we need to write a whole custom class. We have to make it learn some parameters in the fit() method and use them later in the transform() method.

You can get fit_transform() for free by simply adding TransformerMixin as a base class. BaseEstimator is also inherited to use scikit-learn things like set_params().

Here's a Standard Scaler clone:
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True): # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None): # y is required even though we don't use it
        X = check_array(X) # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
        return self # always return self!

    def transform(self, X):
        check_is_fitted(self) # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
```

Takes some work, but it's very cool!

A custom transformer can use other estimators in its implementation. For example, here's some code that uses KMeans clustering in the fit() method to find the main clusters in the training data, then uses rbf_kernel() in the transform method to measure how similar each samle is to each cluster center.

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

### A quick aside on what KMeans clustering is
KMeans clustering is a way to "group together" data using a number of center points, called centroids (K). The data isn't actually grouped together; the centroids are moved to the means of the data points that are "assigned" to them based on their Euclidean Distance, which is how far they are from that centroid using the pythagorean theorem, i.e. sqrt((x-centroid)^2 + (y-centroid)^2 ...). They use this algorithm for every point and every centroid, and the points that are closer to a centroid than other centroid are "assigned" to that centroid. Then, the means of the assigned points are taken and the centroid is relocated there, and this process continues however many times you want or until there is no change in any centroids. The centroids' original positions are randomly assigned, so there is some potential for very stupid looking clusters, but there are techniques that I am currently not familiar with that are used to fix this problem, including KMeans++.

So our complicated looking transformer fits the data to a KMeans Clustering algorithm, uses weights to... well... weigh the data, then measures the similarity between each district and the cluster centers. Here's a cool graph depicting what this looks like:

![image](https://github.com/user-attachments/assets/e959d026-7ce0-407f-87fe-f40282b27f76)

## Transformation pipelines

There are MANY, things to do when data preprocessing. I signed up for this sh** because of how cool AI was, and now I'm sifting through house prices and biomedical records. I STILL LOVE IT.

Because there are so many steps to this data preprocessing, and those steps need to be executed in the right order, we can use **PIPELINESSSS**.

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```

This scared the CRAP out of me when I first learned about it, but it's actually very straightforward: It's just a step-by-step process. First, the 'impute' named action is done, where SimpleImputer is applied, then 'standardize' uses StandardScaler on the data. Then we're done! A pipline just puts everything in one place so it's easy to use, to evaluate, to tweak and change, and to use in the future for other projects.

