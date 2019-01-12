"""
# temporary file to experiment with writing an optimizer
# in this case optimizer will have to replicate the nodes to get something done


# optimize any network with a bigger fully-connected network
# and if that does not work, I will need variations
"""

import tensorflow as tf

# function_network(data=data, schema=schema)
# return loss, metrics

# pranam_optimizer(function_network=function_network,embed_shape)
#    loss,_ = function_network
#    trainable_variables = _get_gradient(loss) # build a list of trainable variables (could be a graph)
#    embedding = build_perceptron(embed_shape)

#    for i in range(embed_shape[0]):
#        loss, metric = replace_graph(function_network)
#  train_step = tf.train.AdamOptimizer(losses)

# return losses,metrics,train_step

