import mnist 
import random
import numpy as np
import csv
from utils.regularization import Zero_Reg

class FFNetwork(object):
    def __init__(self, layers, activations, loss_func, momentum=0.0,init_weight_avg_var = (0,1), regularization=None, reg_const = 0.0, lr=0.5, correct_score=1.0, incorrect_score=0.0):

        assert len(layers)==(len(activations)+1), 'layers and activations must have the same size'
        self.activations = activations
        self.layers = layers

        self.init_weight_avg, self.init_weight_var = init_weight_avg_var

        self.biases = [  np.random.randn(num_neuron,1) for num_neuron in layers[1:]]
        self.weights = [ self.init_weight_var * np.random.randn(l_prev,l) + self.init_weight_avg for l_prev,l in zip(layers[:-1],layers[1:])]

        self.nets = [ np.zeros((num_neuron,1)) for index,num_neuron in enumerate(layers)]
        self.outputs = [ np.zeros((num_neuron,1))  for index,num_neuron in enumerate(layers)]
        self.loss_func = loss_func
        self.lr = lr
        self.momentum = momentum
        if regularization is None:
            regularization = Zero_Reg
        self.regularization = regularization
        self.reg_const = reg_const
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        self.num_layers = len(layers)

        self.forward( self.nets[0] )
        self.dw = [np.zeros(weight.shape) for weight in self.weights]
        self.db = [np.zeros(bias.shape) for bias in self.biases]
        self.dw_prev = [np.zeros(weight.shape) for weight in self.weights]
        self.db_prev = [np.zeros(bias.shape) for bias in self.biases]

    def push_hidden_layer(self,num_neuron,activation):

        self.nets.insert(-1, np.zeros((num_neuron,1)) )
        self.outputs.insert(-1, np.zeros((num_neuron,1)) )        

        self.biases[-1] = np.random.randn(num_neuron,1)
        self.db[-1] = np.zeros( self.biases[-1].shape )
        self.db_prev[-1] = np.zeros( self.biases[-1].shape )

        self.weights[-1] =  self.init_weight_var * np.random.randn(self.layers[-2],num_neuron) + self.init_weight_avg
        self.dw[-1] = np.zeros( self.weights[-1].shape )
        self.dw_prev[-1] = np.zeros( self.weights[-1].shape )

        self.biases.append( np.random.randn(self.layers[-1],1) )
        self.db.append( np.zeros( (self.layers[-1],1) ) )
        self.db_prev.append( np.zeros( (self.layers[-1],1) ) )

        self.weights.append( self.init_weight_var * np.random.randn(num_neuron,self.layers[-1]) + self.init_weight_avg )
        self.dw.append( np.zeros( (num_neuron,self.layers[-1]) ) )
        self.dw_prev.append( np.zeros( (num_neuron,self.layers[-1]) ) )


        self.layers.insert(-1,num_neuron)
        self.activations.insert(-1,activation)
        self.num_layers += 1
        

    def forward(self,input_arr_1d):
        self.nets[0] = input_arr_1d
        self.outputs[0] = input_arr_1d
        
        for index,(weight,activation ) in enumerate(zip(self.weights,self.activations)):
            sigma = np.dot(weight.T,self.outputs[index])
            self.nets[index+1] = np.add(sigma,self.biases[index])
            self.outputs[index+1] = activation.f(self.nets[index+1])
        return self.outputs[-1]

    def test_net(self,val_data):
        acc = 0
        for sample in val_data:
            img, label = sample
            out = self.forward(img)
            acc += (out.argmax() == label)
        return acc
    
    def train(self, data_loader, num_epochs, batch_size):
        train_data = data_loader.train_data
        val_data = data_loader.val_data
        num_samples = len(train_data)
        num_classes = data_loader.num_classes()
        num_validation = len(val_data)

        losses = []
        accuracies = []

        for epoch in range(num_epochs):
            random.shuffle( train_data )
            mini_batches = [ train_data[ix:ix+batch_size] for ix in range(0,num_samples,batch_size) ]
            epoch_losses = []
            for ix, batch in enumerate(mini_batches):
                print('{}/{}({}) : {}/{}({})'.format(
                    epoch,num_epochs,f"{(epoch/num_epochs)*100:.2f}%",
                    ix,len(mini_batches),f"{(ix/len(mini_batches))*100:.2f}%"
                    ),end='\r')
                epoch_losses.append( self.update_mini_batch(batch,batch_size,num_classes, num_samples) )
            
            acc = self.test_net(val_data)
            
            losses.append(epoch_losses)
            accuracies.append( acc )
            
            print("Accuracy after training epoch {}: {} / {})".format(epoch, acc, num_validation))
        return losses,accuracies

    def update_mini_batch(self, batch, batch_size,num_classes, num_samples):
        dloss_reg = self.regularization.df(self.weights)
        loss_reg = self.regularization.f(self.weights)
        # print('loss_reg : {}'.format(loss_reg))
        # print('dloss_reg : \n{}\n'.format(dloss_reg))

        self.dw_prev = list(self.dw)
        self.db_prev = list(self.db)
        
        self.dw = [np.zeros(weight.shape) for weight in self.weights]
        self.db = [np.zeros(bias.shape) for bias in self.biases]
        dw_coeffient = self.lr / batch_size

        mini_batche_losses = []

        decay = (self.lr * self.reg_const) 



        for sample in batch:
            img, label = sample
            y_expected = np.ones((num_classes,1)) * self.incorrect_score
            y_expected[label,0] = self.correct_score
            deltas,loss = self.backpropagate(img,y_expected)
            # self.dw = [self.momentum * dw_prev + np.transpose(np.outer(delta, out)) for (dw_prev, delta, out) in zip(self.dw, deltas, self.outputs[:-1])]
            # self.db = [self.momentum * db_prev + delta for (db_prev, delta) in zip(self.db, deltas)]
            mini_batche_losses.append( loss + loss_reg )
            self.dw = [dw_prev + np.transpose(np.outer(delta, out)) for (dw_prev, delta, out) in zip(self.dw, deltas, self.outputs[:-1])]
            self.db = [db_prev + delta for (db_prev, delta) in zip(self.db, deltas)]
        self.weights = [ w_prev - decay * dw_reg - dw_coeffient * delta_w for (w_prev, delta_w,dw_reg) in zip(self.weights, self.dw, dloss_reg) ]
        self.biases = [ bias_prev - dw_coeffient * delta_b for (bias_prev, delta_b) in zip(self.biases, self.db) ]
        # self.weights = [ w_prev - dw_coeffient * delta_w + delta_w_prev for (w_prev, delta_w, delta_w_prev) in zip(self.weights, self.dw, self.dw_prev) ]
        # self.biases = [ bias_prev - dw_coeffient * delta_b + delta_b_prev for (bias_prev, delta_b, delta_b_prev) in zip(self.biases, self.db, self.db_prev) ]
    
        # self.weights = [ w_prev - dw_coeffient * delta_w + self.momentum * delta_w_prev  for (w_prev, delta_w, delta_w_prev) in zip(self.weights, self.dw, self.dw_prev) ]
        # self.biases = [ bias_prev - dw_coeffient * delta_b + self.momentum * delta_b_prev for (bias_prev, delta_b, delta_b_prev) in zip(self.biases, self.db, self.db_prev) ]
    
        return mini_batche_losses

    def backpropagate(self,x_input,y_expected):
        delta = [None] * (self.num_layers-1)
        self.forward(x_input)
        loss = self.loss_func.f(y_expected,self.outputs[-1])
        delta[-1] = self.loss_func.df(y_expected,self.outputs[-1])
        for ix in range(2,self.num_layers):
            delta[-ix] = np.dot( self.weights[-ix+1],delta[-ix+1] ) * self.activations[-ix+1].df(self.nets[-ix])
        return delta,np.sum(loss).item()

    def save_weights(self,filepath):
        for index,weight in enumerate(self.weights):
            filename = filepath + ('{}.csv'.format(index))
            np.savetxt(filename,weight)

    def load_weights(self,filepath,num_of_layers):
        for layer in range(num_of_layers):
            filename = filepath + ('{}.csv'.format(layer))
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                headers = next(reader)
                data = list(reader)
                data = np.array(data).astype(float)
            self.weights[layer] = data

    def __str__(self):
        print('type : Feed Forward')
        print('input layer : {} neurons'.format(self.layers[0]))
        for index,(layer,activation) in enumerate(zip(self.layers[1:],self.activations)):
            print('{} : {} neurons , activation : {} '.format(index+1,layer,activation))
        print('loss function : {}'.format(self.loss_func))
        print('learning rate : {}'.format(self.lr))
        print('momentum : {}'.format(self.momentum))
        print('weight initialization average : {}'.format(self.init_weight_avg))
        print('weight initialization variance : {}'.format(self.init_weight_var))
        return ''