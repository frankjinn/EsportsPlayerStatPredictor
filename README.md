# Esports Player Stat Predictor
## Description
This model predicts a player's stat (Kills, Deaths, etc.) based on historical data using linear regression. Currently tuned for CS2, this model does better or
at least equal to predictions made by commercial online esports betting companies in terms of MSE. Usually, this model outperforms 
commercial online esports predictions in terms of the precision of the prediction, usually having a smaller STD error. 
The relevant files are stored in pickled data frames in the cache.

## Tests folder
In the tests folder you can see the outcomes of experiments run with different models, and compared to the previous model this was based off of.
This model tends to do an average of 25% better than the previous model in terms of MSE. This model also includes features (namely STD), that the
previous model did not have. 

It also predicts new statistics that the previous model does not predict, these predictions do not have a comparison with the previous model.

## Running
- To replicate the results, run the Jupyter notebook file. Although marked Rocket League, it is tuned for CS2. 
- Use the non-overtime matches initially, this is later joined with the overtime matches.
- The bias is calculated first, then added onto the model, so the model will be run at least twice. The first time will use non-overtime training data, to predict overtime data labels. The bias is taken and added to the prediction. The second time, the model should be tuned to predict overtime data.

## About
This code is from a small work sample from a prior consulting internship. Many features, such as the initial pull from the database into a data frame, and some features were given to me. My contribution was mainly feature analysis, creating new statistics to predict, new evaluation metrics, and bug fixes with the code.
