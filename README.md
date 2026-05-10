
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

