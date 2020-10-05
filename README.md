# Project 4

In this project, we used NFL data from fivethirtyeight and made a regression model to predict the scores of a team. I approached this problem by training different models on a given teams data, and then predicting scores for that team.

Some of the dataset modifications were:
* A ScoreDif variable to indicate the caliber of team they were up against.
* Analyzing which variables actually added information to the models, and which ones made it worse.

## Regression Methods
We attempted to run two different regression methods for our predictor: linear regression and gradient boosting.

### Linear Regression
In this model, we trained a linear regression model on a team's data and created a model that would predict the scores of the team. This model had fairly low accuracy, and analyzing the data you can understand why because a team's score doesn't typically follow a linear model. This lead us to try a different method that would explain more fully the change in score.

### Gradient Boosting

For the gradient boosting model, I was hopeful we would be able to build a relatively accuracy model. Using the same inputs as in the linear regression model, we randomly chose a different team and trained the model. In this model we saw a large increase in accuracy. This model did a much better job explaining the variation in the team's scores.
