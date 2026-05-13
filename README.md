
first lecture:

loop for a neural net


get y preds
find loss
reset the gradients to zero
loss.backward

then update the values acc to learning rate
repeat

regression problem at the end

second lecture:
usage of softmax activation function
usage of torch.multinomial to sample for both cases, inference
use negative log likelihood for loss function instead of mean squared error because of classification problem


bigram table 2 methods:
first, creating a heatmap using count and normalizing and making a row add up to 1
    create table relating to the relationship between 2 characters, consequent ones
    normalize the rows to add up to 1
    profit
    .sum(1,keepdims=True) is an essential thing for things to go right


second
    How does PyTorch's broadcasting rule apply to the operation probs = counts / counts.sum(1, keepdims=True)?
        expands 27x1 to 27x27 replicating across rows
    create dataset  
    initialize weights
    change input to one hot encoded vector if character is b then [0,0,0,1,0,0,....]
    create logits
    log logits and normalize them as well
    take loss
    clear weights
    loss.backward()
    update weights
    repeat
    

third lecture:
    this time instead of using bigram model to predict the next letter given context of just one letter, the task is to predict the next letter given 3 letters of context, in hopes that the prediction is more accurate.
    a lot of tensor manipulation here
    torch.tensor.view to change the actual metadata of the tensor instead of creating new tensor in memory which is efficient
    made using paper about multilayer perceptron 

    steps:
        first step is always preparing the dataset for training, at the end of the lecture the dataset is split into three differnt sections, one is the training set, then the dev or validation set and the last one is test set where the actual test is done and the benchmark and error rate is calculated. typically in 80, 10,10 ratio. the dev or validation is for testing with the hyperparameters
        we can know the model is optimum for the current paramters when the training accuracy and the test accuracy drifts away from each other, ie the model is starting to overfit on training data.
        another thing, the concept of learning rate decay is also introduced here. and setting the learning rate is also a skill. first check using large and small learning rates and make a range of them, for small interations. then check the graph or the log graph and where the model performs the best or the loss graph is most changed and is converging to the actual value. after finding the optimal value train the model on that learning rate for many many iterations >100000 and during the last iteration train on a small learning rate to catch the local maximum.
        - create representation stoi, itos
        - creating context first assign context*[0], ie '...' in representation and moving the context by concatenating [1:]+currentcontext eg [1:]+ 'a' converts to '..a'
        - build the datasets using the aforementioned logic
        - create embeddings by putting the input context and next letter
        - initialize weights and biases randomly
        - have all the parameters require gradient
        - implement the decaying loss rate, in exponentials because that's more telling instead of linear
        - create minibatch
        - train on the minibatch, indexing only the minibatch index vectrs
        - forward, backward, logging
        - check the losses
        - check the logging graph 
        - change hyperparameters and repeat until low loss

fourth lecture:
    not much neural networks created in this lecture but optimizationof the network and initial weights, batch normalization is introduced
    when we initialize the network weights randomly, the network guesses incorrectly and confident at that so we need better starting neurons.
    when we draw the plot of the loss over time, the figure looks like a hockey stick because of the random initialization of the weights, to solve that problem and have a roughly equal probability among the weights, which in turn reduces training time, we divide the initial weights by 1/fan_in**0.5    basically scaling down the initial weigths, the scale down factor is done through math and the activation functions have their own best init scale down. we are reducing the variance to 1.
    this is applicable for small network like this but on  a large scale network, it doesnt work and a new method is proposed and taht is batch normalization. it's basically normalizing the layer before it goes through to activation. this in turn affects the results of the network. it's all math here.
    during batch normalization, we need to multiply gain and add bias during inference because of our extra math, and instead of doing it after training, it's done during training, but on the side. optimizing the running mean and running bias. beacause the data was normalized during batch normalization.


fifth lecture:
    this lecture was more about backpropagation, didn't struggle too much with the calculus but struggled with the tensor shapes, broadcasting, need to look further and get more practice on tensor manipulation

sixth lecture:
    increased context length for better accuracy
    basically, instead of creating one whole list of embeddings, we embed in groups of two, therefore we have to change the shape of everything, from 2 dimensions to three.
    created embedding class to create embeddings
    flatten/flattenConsecutive afterwards, this is equivalent to torch.flatten. i think this is to create a single long list of all the elements? and in this case to create appropriate dimensions after embedding
    created sequential for a list of layers, reducing the lines of repeated code and get closer to torch.nn.sequential
    we can now assign model as sequential layers, embedding, flattening(dimensions) three times because halving three times
    flattening the layer is important because if we were to use without flattening, we would just get another linear layer instead of desired complexity of the layers.
    some other cleanup of the code and using this architecture.
    keep in mind dilated convolutions mimicking.
    instead of using the whole context we use pairs of inputs three times, 
    
