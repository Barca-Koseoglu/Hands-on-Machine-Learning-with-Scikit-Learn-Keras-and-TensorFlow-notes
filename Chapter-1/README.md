# The Machine Learning Landscape

## What is machine learning?

Machine Learning is the science (and art) of programming computers so they can learn from data.

A more general definition: Machine Learning is the field of study that gives computers the ability to learn
without being explicitly programmed.

A more enigneering-oriented one: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

## But why?

Imagine a you're trying to build spam detection software. Using tradiitonal programming techniques, you might notice a bunch of buzzwords most spam emails use and write a detection algorithm for them.

![image](https://github.com/user-attachments/assets/1019dc49-4ba9-4278-8a4f-6f8b7e7b5611)

Your program will probably be a list of complex rules, so it's pretty hard to maintain. Also, not using machine learning proves futile when spam emailers realize they're buzzword doesn't work anymore so they change it, but your normal program doesn't detect it or change by itself. Machine learning does exactly that, and that's why it's simpler and better to use.

![image](https://github.com/user-attachments/assets/5d3d65a4-5004-42e7-bc55-eaac0f5fee1b)

Another example is speech detection. Let's say you want to find a way to differentiate between someone saying one and two. You can't just use something like an algorithm to see if someone says something more high pitched because millions of people can say it in many of different ways. So the best solution is to write an algo that learns by itself.

Last but not least, ML helps us learn too. If we look at what ML has learned, we can use it to find unsuspected correlations or trends and that'll lead to a better understanding of the problem.

Applying ML techniques to dig into large amounts of data can help discover patterns that were not immediately apparent. This is called data mining.

![image](https://github.com/user-attachments/assets/e3108b1f-51f6-4c6a-9d92-eee232c1c6a6)

To summarize, Machine Learning is great for:

• Problems for which existing solutions require a lot of fine-tuning or long lists of
rules: one Machine Learning algorithm can often simplify code and perform bet‐
ter than the traditional approach.

• Complex problems for which using a traditional approach yields no good solu‐
tion: the best Machine Learning techniques can perhaps find a solution.

• Fluctuating environments: a Machine Learning system can adapt to new data.

• Getting insights about complex problems and large amounts of data.

## Types of machine learning systems

There are a lot of different types of ML systems, so there are ways to classfiy them.

• Whether or not they are trained with human supervision (supervised, unsuper‐
vised, semisupervised, and Reinforcement Learning)

• Whether or not they can learn incrementally on the fly (online versus batch
learning)

• Whether they work by simply comparing new data points to known data points,
or instead by detecting patterns in the training data and building a predictive
model, much like scientists do (instance-based versus model-based learning)

### Supervised/Unsupervised learning

4 types: supervised, unsupervised, semisupervised, and reinforcement learning

In supervised learning, the training set you feed to the algo has the desired solutions, called labels.

![image](https://github.com/user-attachments/assets/2b8d4ff7-9b63-44e9-8969-1d7b9a6ef119)

Classification is a good example of this. The spam filter is trainged with many example emails along with their class (spam or ham) and it must learn how to classify new emails.

Another task is to predict a target numeric value, like the price of a car, given a set of features (mileage, age, brand...) called predictors. It's called regression, and to train it you need many examples of cars including both ther predictors and labels, aka their prices.

![image](https://github.com/user-attachments/assets/d4e8271b-ea53-4798-801f-4bc0b4759dbb)

Here are some of the most important supervised learning algorithms:

• k-Nearest Neighbors

• Linear Regression

• Logistic Regression

• Support Vector Machines (SVMs)

• Decision Trees and Random Forests

• Neural networks

### Unsupervised leanring

In unsupervised leanring, the training data is unlabeled, so the system tries to learn without a teacher.

Here are some of the most important unsupervised learning algorithms:

• Clustering

  — K-Means
  
  — DBSCAN
  
  — Hierarchical Cluster Analysis (HCA)
  
• Anomaly detection and novelty detection

  — One-class SVM
  
  — Isolation Forest
  
• Visualization and dimensionality reduction

  — Principal Component Analysis (PCA)
  
  — Kernel PCA
  
  — Locally Linear Embedding (LLE)
  
  — t-Distributed Stochastic Neighbor Embedding (t-SNE)
  
• Association rule learning

  — Apriori
  
  — Eclat

For example, say you have a lot of data about your blog’s visitors. You may want to run a clustering algorithm to try to detect groups of similar visitors. At no point do you tell the algorithm which group a visitor belongs to: it finds those connections without your help. For example, it might notice that 40% of your visitors are males who love comic books and generally read your blog in the evening, while 20% are young sci-fi lovers who visit during the weekends. If you use a hierarchical clustering algorithm, it may also subdivide each group into smaller groups. This may help you target your posts for each group.

Visulatization algorithms are also good examples. You feed them a lot of complex and unlabeled data and they output a 2D or 3D representation of it that can be easily plotted.

Another task is dimenstionality reduction, which means to simplify the data without losing too much info. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called feature extraction.

Another important task is anomaly detection, for example detecting unusual credit card transactions to prevent fraud or catching manufacturing defects. The system is shown mostly normal instances during training, so it learns to recognize them; then, when it sees a new instance, it can tell whether it looks like a normal one or whether it is likely an anomaly.

A similar task is novalty detection; it aims to detect all instances that look different from all instances in the training set. This requires a very clean training set devoid of any instance that you want to detect.  For example, if you have thousands of pictures of dogs, and 1% of these pictures represent Chihuahuas, then a novelty detection algorithm should not treat new pictures of Chihuahuas as novelties. On the other hand, anomaly detection algorithms may consider these dogs as so rare and so different from other dogs that they would likely classify them as anomalies.

![image](https://github.com/user-attachments/assets/469023bc-9f83-44bc-90bd-78a9c6173d2a)

Another common unsupervised task is association rule learning, where the goal is to dig into large amounts of data and discover interesting relations between attributes. For example, suppose you own a supermarket. Running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also tend to buy steak. Thus, you may want to place these items close to one another.

### Semisupervised learning

This is for algorithms that deal with data that's partially labeled.

![image](https://github.com/user-attachments/assets/5ad17717-a76d-4f6a-947f-1e50b1ae3ad6)

Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person and it is able to name everyone in every photo, which is useful for searching photos.

### Reinforcement learning

The learning system is called an agent, and it can observe the environment, select and perform actions, and get reward in return, or even penalties in the form of negative rewards. It must learn by itself what the best strategy is, called policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![image](https://github.com/user-attachments/assets/1ba132cd-464e-420e-9492-724ea18c0d6b)

Many robots implement this to learn how to walk. It can beat people in board games like Go after analying millions of games and then playing many games against itself.

## Batch and Online Learning

Learning from a stream of incoming data

### Batch learning

In batch learning, the system can't learn incrementally and needs all the data. This takes a bunch of time and resources so people typically do it offline. It's trained first then launched into production and runs without learning anymore. This is called offline learning.

If you want it to know about ne data, you have to train a new version from scratch. Thankfully, the whole process of training, evauating and launching an ML system can be automated pretty easily.

A better option for this is online learning

### Online learning

In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly.

![image](https://github.com/user-attachments/assets/ea2d379a-6d9c-4a5d-a9f6-8338d8e89324)

This saves a huge amount of space because once an online learning system has learned about new data instances, it does not need them anymore, so you can discard them.

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data (you don’t want a spam filter to flag only the latest kinds of spam it was shown). Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points (outliers).

![image](https://github.com/user-attachments/assets/cac9fd3b-9c0c-49d6-98b0-9a90d83b755c)

One problem we could encounter is if bad data is given to the system. It it's bad, it'll gradually decline its performance. The best way to reduce this risk is to monitor the system closely and switch learning off if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

## Instance-based versus model-based learning

Machine learning systems can generalize. A lot of ML tasks are about making predictions.

### Instance-based learning

The most trivial form of learning; by heart.

If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users—not the worst solution, but certainly not the best.

Instead of this, though, you could flag emails that are very *similar* to the known spam emails. A very basic *measure of similarity* would be words they have in common.

This is called instance-based learning: the system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples.

![image](https://github.com/user-attachments/assets/17129657-e71e-4f94-8aac-ab04514ee8e1)

### Model-based learning

Another way to generalize is to build a model of these examples then use it to make predicitons. This is called model-based learning.

![image](https://github.com/user-attachments/assets/4e9545be-9461-4954-8fcc-d5848c62d9c7)

Let's say you want to know if money makes people happy. You take some life satisfaction data and GDP per capita data for a bunch of countries, then plot them.

![image](https://github.com/user-attachments/assets/a12cbd47-20d0-4478-86c4-1e470a6f4dd3)

Although this data is noisy (a little random), it still looks like there's a linear correlation. Now you decide to model life happiness as a linear function of GDP per capita. This step is called model selection; you select a linear model of life satifaction with just one attribute, GDP per capita.

life_satisfaction = θ<sub>0</sub> + θ<sub>1</sub> × GDP_per_capita

Using the parameters, you can model any linear function.

![image](https://github.com/user-attachments/assets/e876b716-7ecc-4eed-818b-84c462d35e4f)

Before you use your model, you need to set the parameters. But how do you do this? You need a specifc performance measure. Either define a utitlity fucntion that measures how good it is, or a cost function to describe how bad it is. For linear regression problems, people normally use cost functions.

Now, the linear regression algorithm comes in: feed it your training examples and it finds the parameters that best fit your data.

![image](https://github.com/user-attachments/assets/494acf7e-70d7-4ea2-9eaa-674c05df6e78)

So the steps are studying the data, selecting a model, training it on the training data, and finally applying the model to make predictions on new cases. This is what a typical machine learning project looks like.

## Main challenges of machine learning

Since our main task is to select a learning algorithm and train it on data, the things that can go wrong are a 'bad algorithm' and 'bad data.'

### Insufficient quantity of training data

For humans to learn what something is, you just need them to see it once. Like an apple. After they see it, they can recognize an apple that comes in different colors and shapes. ML is not really that genius yet. It takes a ton of data for most ML algorithms to work properly. Even for some simple problems, you might need thousands of examples, and for problems like speech or image recognition, you might need millions.

### Nonrepresentive training data

In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. This is true whether you use instance-based learning or model-based learning.

![image](https://github.com/user-attachments/assets/a670e9fd-5e0c-4e43-821a-52af7e2be230)

Looking at the last example, we didn't include some countries (the red dots). After including them, we get the solid line. The dotted line represents our previous model.

So you can't make accurate representations.

It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

### Poor-quality data

Having your data filled wiht errors, outliers and noise will make it harder for the system to detect any patterns. Cleaning your data is often well-worth the effort.

### Irrelevant features

Garbage in, garbage out. The system will only learn with relevant features.

A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called feature engineering, involves the following steps:

• Feature selection (selecting the most useful features to train on among existing features)

• Feature extraction (combining existing features to produce a more useful one— as we saw earlier, dimensionality reduction algorithms can help)

• Creating new features by gathering new data

### Overfitting the training data

Imagine you visit a country and a taxi driver rips you off. You might be tempted to say that all tax drivers are thieves. Overgeneralizing is something we do and something algorithms can do too. This is called overfitting: it means the model performs but doesn't generalize well.

Complex models can detect subtle patterns, but if it's noisy or too small, then it can actually detect patterns in the noise itself. For example, feed the life satisfaction model more attributes like the names of the countries. It see that countries that have a w in their name have life satisfaction over 7, but how would this generalize to Rwanda or Zimbabwe? This pattern is by pure chance, but there's no way the model can know that.

Constraining a model to make it simpler is called regularization. The linear model earlier has two parameters, θ<sub>1</sub> and θ<sub>1</sub>. That gives the model two degrees of freedom to adapt to the training data. If we forced one of them to be 0, we would've had a hard time fitting it correctly. If we picked one of them to be modified but small, we would have freedom between one and two degrees. You want to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.

![image](https://github.com/user-attachments/assets/642bec88-2655-409d-8b3d-0eac097df083)

The amount of regularization applied can be controlled by a hyperparameter. A hyperparameter is a parameter of a learning algorithm. Tuning hyperparameters is important.

### Underfitting the training data

Underfitting is the opposite of overfitting, happening when your model is too simple to learn. The reality of life is more complex than the model we had for life satisfation, so it's bound to be innacurate.

Here are the main options for fixing this problem:

• Select a more powerful model, with more parameters.

• Feed better features to the learning algorithm (feature engineering).

• Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).

## Testing and Validating

The only way to know a model works well is to test it out. If you put your model into production and observe it, you might see how well it performs, but customers won't be happy if it works badly. Instead, split the data into a training set and a test set. Train your model using the training set, and test it using the test set. The error rate on new cases is called the generalization error, aka out-of-sample error. By evaluating the model on the test set, you get an estimation for this error.

If the training error is low but the generalization error is high, then you're overfitting the data.

It's actually common to use 80% of the data and use the other 20 % to test it. Of course it depends on the size, but that's just a general statement.

### Hyperparameter tning and model selection

What if you want to evaluate a model but you have to pick between a linear model and a polynomial model? One option is to just test both and compare how well they generalize.

Suppose the linear model generalizes better. Now, you want to apply some regularization to avoid overfitting. What value will you assign the regularization hyperparameter? One option is to train 100 different models with 100 different values for it. After doing this, you find one with low error and send it off to production. But a higher error rate comes out. So what happened?

The problem was that you measured the generalization error multipe times on the test set and adapted the model and hyperparameters to produce the best model **for that particular set**. This means the model isn't likely to perform well on new data.



