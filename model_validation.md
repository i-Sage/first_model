# What is Model Validataion

The relevant measure of model quality is predictive accuracy. In other words, will the 
model's predicitons be close to what actually happens.

There are many metrics for summarizing model quality, be we'll start with one called
**Mean Absolute Error (MAE)**. Lets break down this metric starting with the last word,
error. 

The prediction error for each house is:

|> Error = actual - predictetd

With the MAE metric, we can take the absolute value of each error, and then take the 
averate of those absolute errors. This is our measure of model quality. In plain english,
it can be said as:

|> On average, our predictions are off by about X

To calculate MAE, we first need a model
