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

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2 (controls opacity))

plt.show()

### Correlation: 

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }
         { \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \cdot \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

How correlated two things are. Not something crazy but it will guide us in our endeavors. It's normalized covariance. Basically covariance (the measure of if something is positively or negatively related) divided by the standard deviation of X times the standard deviation of Y. We divide because it normalizes, as mentioned. Like for height and weight, we divide by the spread of the data to bring it from a scaled value, like cm, to just a unit value. It's kind of weird to explain but it works.

This might miss out on non-linear relationships though.

### Scatter matrix

scatter_matrix(dataframe[attributes)
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

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

Now, here it shows it from -1 to 1. This is found using the general formula for minmax (I said minimax a lot confusing it for something else, the correct wording is MIN-MAX) which is the formula to scale it from [a,b]: a + ((x-min(x))*(b-a))/(max(x)-min(x)). It's pretty intuitive once again, using very simple variable manipulation.


### Standardization

Standardization first subtracts the mean value (so standardized values have a zero mean), then it divides the result by the standard deviation (so the values have standard deviation equal to 1: literally unit spread). This technique doesn't restrict values to a specific range, but it's much less affected by outliers. Imagine we have a datset of values from 0-15, but we accidentaly get 100 in there. Min-max would make that value equal to 1 and the rest of the value from 0 to 0.15, but standardization doesn't get affected as much as this.

Code:

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

### Heavy tails

Weird subtitle, but very very VERY important topic. When a feature's distribution has a heavy tail (aka when the values far from the mean aren't exponentially rare), both min-max and standardization will scale most values to a small range. Before we scale the values, we should transform them to get rid of that heavy tail, and try to make the distributions kinda symmetrical. A common way to do this is to replace the feature with it's square root (raise it to a power between 0 and 1 if you want more or less power than that). If it has a VERY long and heavy tail, like a dinosaurs, then replace the feature with it's logarithm. The best way to understand this is to visualize it:

![image](https://github.com/user-attachments/assets/a94a7d71-8e37-42f1-ada2-016b22f96414)

Another approach is to *bucketize* the feature. This means chopping the distribution to roughly equal-sized buckets and replacing each feature value with the index of the bucket it belongs to, kind of similar to stratified sampling.

Here's a very simple way to understand it, according to the GOAT, chatgpt: 

![image](https://github.com/user-attachments/assets/8da23b04-61ad-425f-a008-b85fcf0ffe17)

When a feature has multimodel distrbution (two or more peaks in the data, called modes), it can be helpful to bucketize it, but treat thebuckets as categories rather than numerical values. You can use OneHotEncoder to encode te indices of these buckets. This now allows models to understand more easily different rules for different ranges of this feature value. For example, maybe for some house ages, houses built around 35 years ago are weird and ugly, so they're cheaper than most other houses.

### The worst thing this book has tried explaining since chapter 1: the radial basis function

