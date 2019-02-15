import numpy as np
import time
import copy

#create the option to switch between different nonlins (first class functions)
def sigmoid(x, deriv = False):
    if deriv is True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def tanh(x, deriv = False):
    if deriv is True:
        return 1 - x*x
    return np.tanh(x)

def relu(x, deriv = False):
    if deriv is True:
        return int(x > 0)
    else:
        return max(0,x)

#TODO: Write docstrings so I don't have to read code more to know exactly what is happening 
class Model(object):
    def __init__(self, layers, learning_rate):
        self.layers = [np.zeros((l,1)) for l in layers] #[784,128,10] is 784 in, 128 middle, 10 out
        self.weights = [2*np.random.rand(layers[i+1], layers[i])-1 for i in range(len(layers)-1)]
        self.biases =  [2*np.random.rand(l,1)-1 for l in layers[1:]]
        self.learning_rate = learning_rate
        self.epochs_trained = 0
    
    #single image input
    #TODO: implement batches into first layer (staying consistent with rest of architecture)
    def load_image(self, img):
        inlayer = img / 255
        inlayer = inlayer.flatten()
        self.layers[0] = np.array([inlayer]).T
    
    def evaluate(self):
        for i in range(len(self.layers) - 1):
            self.layers[i+1] = sigmoid(np.dot(self.weights[i], self.layers[i]) + self.biases[i])
    
    # compares expected with last layer and returns a value
    def cost(self, y):
        if type(y) is list:
            expected = np.array(y)
        else: #is nparray in same format as output layer
            expected = y
        
        output = self.layers[-1]
        return np.sum(np.square(output - expected))/2
    
    # uses cost and weights and biases to compute gradient, returns gradient matrix
    # only for single training example (for now)
    def compute_gradient(self, expected):
        neuron_derivs = list(range(len(self.layers)))
        weight_derivs = list(range(len(self.weights)))
        bias_derivs = list(range(len(self.biases)))
        
        neuron_derivs[-1] = self.layers[-1] - expected
        for i in reversed(range(len(neuron_derivs)-1)):
            temparr = neuron_derivs[i+1] * sigmoid(self.layers[i+1], deriv=True)
            temparr = temparr.T * self.weights[i].T
            summed_terms = np.array([np.sum(temparr, axis = 1)]).T
            neuron_derivs[i] = summed_terms
            
        for i in range(len(weight_derivs)):
            temparr = neuron_derivs[i+1] * sigmoid(self.layers[i+1], deriv=True)
            weight_derivs[i] = np.dot(temparr, self.layers[i].T)
            
        for i in range(len(bias_derivs)):
            temparr = neuron_derivs[i+1] * sigmoid(self.layers[i+1], deriv=True)
            bias_derivs[i] = temparr
        
        return weight_derivs, bias_derivs
        
    def apply_gradient(self, gradient):
        weight_derivs = gradient[0]
        bias_derivs = gradient[1]
        weight_derivs = [self.learning_rate * w for w in weight_derivs]
        bias_derivs = [self.learning_rate * b for b in bias_derivs]
        self.weights = [wm-gm for wm,gm in zip(self.weights, weight_derivs)]
        self.biases = [bm-gm for bm,gm in zip(self.biases, bias_derivs)]        
    
    #TODO: add functionality to track test set fitness as well, save that too
    def train(self, _imgs, _labels, batch_size = 100, epochs = 5):
        set_length = len(_imgs)
        batches = set_length//batch_size
        for epoch in range(1,epochs+1):
            self.epochs_trained += 1
            imgs = copy.copy(_imgs)
            labels = copy.copy(_labels)
            avgcost = 0
            number_correct = 0
            print("Epoch %s:" % self.epochs_trained)
            epoch_start = time.time()
            for batch in range(batches):
                imgs_to_train = imgs[:batch_size]
                labels_to_train = labels[:batch_size]
                imgs = imgs[batch_size:]
                labels = labels[batch_size:]
                avgweights = []
                avgbiases = []
                for img,label in zip(imgs_to_train,labels_to_train):
                    expected = np.zeros((10,1), dtype='uint8')
                    expected[label,0] = 1
                    self.load_image(img)
                    self.evaluate()
                    avgcost += self.cost(expected)
                    weight_derivs, bias_derivs = self.compute_gradient(expected)
                    avgweights.append(weight_derivs)
                    avgbiases.append(bias_derivs)
                    if np.argmax(self.layers[-1]) == np.argmax(expected):
                        number_correct += 1
                avgweights = np.average(avgweights, axis=0)
                avgbiases = np.average(avgbiases, axis=0)
                self.apply_gradient((avgweights, avgbiases))
            avgcost /= (batch_size * batches)
            print("    Average Cost: %s" % avgcost)
            print("    Percentage Correct: %s%%" % round(100*number_correct/(batch_size * batches), 3))
            print("    Time Taken: %s" % (time.time()-epoch_start))
            with open('logfile.csv', 'a') as f:
                f.write('%s,%s,%s\n' % (epoch, avgcost, round(100*number_correct/(batch_size * batches), 3)))
        print("Training Complete!")
    
    def predict(self, img):
        self.load_image(img)
        self.evaluate()
        return np.argmax(self.layers[-1].flatten().tolist())
    
    #test large groups of test sets at once, will be called in training to prevent overfitting as well
    #returns average cost, percentage correct in tuple
    def test(self, _imgs, _labels, batch_size = 100):
        set_length = len(_imgs)
        batches = set_length//batch_size
        imgs = copy.copy(_imgs)
        labels = copy.copy(_labels)
        avgcosts = []
        number_correct = 0
        for batch in range(batches):
            avgcost = 0
            imgs_to_train = imgs[:batch_size]
            labels_to_train = labels[:batch_size]
            imgs = imgs[batch_size:]
            labels = labels[batch_size:]
            for img,label in zip(imgs_to_train,labels_to_train):
                expected = np.zeros((10,1), dtype='uint8')
                expected[label,0] = 1
                self.load_image(img)
                self.evaluate()
                avgcost += self.cost(expected)
                if np.argmax(self.layers[-1]) == np.argmax(expected):
                    number_correct += 1
            avgcost /= batch_size
            avgcosts.append(avgcost)
        avgcost = np.average(avgcosts)
        percentage_correct = number_correct/(batch_size * batches)
        return avgcost, percentage_correct
        
        
    def save_network(self, fp):
        d = {}
        for i in range(len(self.weights)):
            d["weights_%s" % i] = self.weights[i]
        for i in range(len(self.biases)):
            d["biases_%s" % i] = self.biases[i]
        np.savez(fp, **d)
    
    @classmethod
    def load_network(cls, fp):
        npzarchive = np.load(fp)
        weights, biases = [], []
        n_weights, n_biases = 0,0
        for i in npzarchive:
            if 'weights' in i:
                n_weights += 1
            elif 'biases' in i:
                n_biases += 1
        weights = [0]*n_weights
        biases = [0]*n_biases
        for i in npzarchive:
            if 'weights' in i:
                index = int(i.split('_')[-1])
                weights[index] = npzarchive[i]
            elif 'biases' in i:
                index = int(i.split('_')[-1])
                biases[index] = npzarchive[i]
        layers = [weights[0].shape[1]] + [arr.shape[0] for arr in biases]
        model = cls(layers, 1.0)
        model.weights = weights
        model.biases = biases
        return model
        