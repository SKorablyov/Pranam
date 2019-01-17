

# Thoughts
Monitor saturations (as in the figure 6 - I should print activations)
batch size dependence
Add best network instead on average (average was only reasonably good for Schwefel)
Try softsign instead of tanh (and other nonlinerities)
Start with mean gradient --> progress everything is for the best gradient
Hierarchical training
Biases could be needed for easier fast shift to 0
Do for the convolution
Embed 2 layers instead of 1 (and try convolutions in that case?)

I could add an online batch norm?


# Deep Thoughts
Covariance could be useful



# Work

## step0
cfg12-19 just trying to capture the signal if nonconvex opt helps at all

For some reason with Xavier initializer does not work
(while it's is the best initialization strategy for a normal net)

## step1
1. report (variable stats + losses in tensorboard)
Embedding [1,64,None] is different from [64,64,None] and should not be 12-27
