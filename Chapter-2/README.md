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

## Working with real data

Normally, it's beneficial to use actual datasets to experiment.

• Popular open data repositories

— UC Irvine Machine Learning Repository

— Kaggle datasets

— Amazon’s AWS datasets

• Meta portals (they list open data repositories)

— Data Portals

— OpenDataMonitor

— Quandl

• Other pages listing many popular open data repositories

— Wikipedia’s list of Machine Learning datasets

— Quora.com

— The datasets subreddit

## Look at the big picture

Your first task is to use California census data to build a model of housing prices in the state. This data includes metrics such as the population, median income, and median housing price for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). We will call them “districts” for short. Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

## Frame the problem

What exactly is the business objective of this model? Knowing the objective is important to determine how the problem is framed, which algorithm will be selected, which performance measure you'll use, and how mch effort is need tweaking it.

Imagine you work at a company. Your boss tells you the model's output will be fed to another ML system along with many other signals (pieces of info fed to a machine learning system). Getting this correct is critical; it greatly affects our revenue (no pressure).

![image](https://github.com/user-attachments/assets/7ed64c2c-684e-4a8a-89a5-14791a244756)

