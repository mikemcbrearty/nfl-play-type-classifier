# nfl-play-type-classifier

Neural network classfier that predicts whether an NFL play will be a run, pass, or kick given a game situation.

I wrote this because I'm learning about neural networks, and I was curious how well a dense classifier would peform at predicting play types.

Also, in some game situations, when a play doesn't work out, there are armchair quarterbacks who might say "Aha, in this situation, obviously a team should run. And in that situation, obviously a team should pass". Is it really so obvious?

## Background

There are several existing projects that attempt to predict NFL play type, [ref](https://rahuljain28.github.io/NFLPredictions/), [ref](https://dspace.mit.edu/bitstream/handle/1721.1/113120/1016455954-MIT.pdf?sequence=1), [ref](https://medium.com/analytics-vidhya/i-can-sort-of-predict-an-nfl-pass-situation-now-a291eb3c1243).

They each achieve an accuracy of ~70-75% at predicting play types, using models including multi layer perceptrons and xgboost.

## Data set

This project uses the [Detailed NFL Play-by-Play Data 2009-2018](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016) dataset from Kaggle.

This dataset includes ten years of data, and is easily accessible.

## Data analysis

I limited the scope of the analysis to plays of type pass, run, punt, or field_goal. Ie. excluding kickoffs, extra point attempts, etc.

For those plays:
|play_type|frequency|
|---|---|
|pass|52.9%|
|run|37.6%|
|punt|6.8%|
|field_goal|2.7%|

So, a baseline would be to predict that every play is a pass, which would give an accuracy of 52.9%.

As input to the classifier, I chose a handful of columns from the dataset that seemed relevant based on my intuition as a football fan. I focused on data about down, field position, game time, and score.

## Model

This project uses a dense neural network classifier. I split the dataset into 60% training data, 20% validation data, and 20% test data. With this, the model implemented with Keras achieved 71% accuracy. (I also implemented a similar model with PyTorch, as well as an XGBoost model. Both of these also achieved 71% accuracy.)

## Results

71% accuracy is a significant improvement over the baseline. And, it's comparable to results achieved by similar work.

Analyzing the performance of the model against the validation dataset, segmenting by the type of play, the model correctly predicts the type of play as follows:

|play_type|accuracy|
|---|---|
|pass|71.8%|
|run|65.6%|
|punt|98.2%|
|field_goal|89.2%|

The accuracy of the model at predicting pass and run plays drives the overall accuracy of the model. Analyzing cases where the model incorrectly predicts the play type, there are some scenarios that stand out.

|play_type|predicted play_type|down|scenario|
|---|---|---|---|
|run|pass|1st down|offensive team is trailing in the 4th quarter|
|run|pass|1st down|offensive team has been penalized and is facing 1st & 15, etc.|
|run|pass|2nd down|2nd & long (eg. 7+)|
|pass|run|1st down|offensive team is leading or trailing by at most two scores, and more than a quarter game time remaining|
|pass|run|2nd down|2nd & <7|
|pass|run|2nd down|offensive team leading in the 4th quarter with less than 5 minute remaining|
|pass|run|3rd down|3rd & 1|
|pass|run|3rd down|offensive team leading in the 4th quarter with less than 5 minute remaining|

Looking at these scenarios, the predicted play types are reasonable given the situations. However, the alternative play type is not necessarily unreasonable either. And, for an offense, if there's some uncertainty about which type of play they'll run in a given situation, then that can make them more difficult to defend, which is to their advantage.

## Run notes

- Download the [Detailed NFL Play-by-Play Data 2009-2018](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016) dataset from Kaggle, and unzip.
- Extract the features using `extract_play_data.sh`. Requires sqlite3.
- Run `keras_model.py`, etc. to train a model.


## Coda

To my surprise, the model strongly tends to predict `pass` in situations where the offensive team is trailing in the fourth quarter. This includes predicting `pass` when it's 2nd and goal from the 2 yard line with 24 seconds remaining in the 4th quarter trailing by 4 points with a timeout.. ie. correctly predicting the Seahawks call from Super Bowl 49.. You can [watch](https://www.youtube.com/watch?v=U7rPIg7ZNQ8) how that worked out. (Personally, I think they should've run there)
