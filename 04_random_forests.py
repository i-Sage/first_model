"""
    Decision trees leave you with a difficult decision. A deep tree with lots of leaves 
will overfit because each prediction is coming from historical data from only the few
houses at its leaf. But a shallow tree with few leaves will perform poorly because it
fails to capture as many distinctions in the raw data.

    A way to reduce the errors is through random forests.

The random forest uses many trees, and it makes a prediction by averaging the predictions
of each component tree. It generally has much better predictive accuracy than a single
decision tree and it works well with default parameters.
"""
