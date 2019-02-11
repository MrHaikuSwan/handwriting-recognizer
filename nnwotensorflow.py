import numpy as np
import copy
from PIL import Image

def sigmoid(x, deriv = False):
    if deriv is True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def relu(x, deriv = False):
    if deriv is True:
        if x > 0:
            return 1
        else:
            return 0
    else:
        return max(0,x)
    
class Model(object):
    def __init__(self, layers, learning_rate):
        self.layers = [np.zeros((l,1)) for l in layers] #[784,128,10] is 784 in, 128 middle, 10 out
        self.weights = [2*np.random.rand(layers[i+1], layers[i])-1 for i in range(len(layers)-1)]
        self.biases =  [2*np.random.rand(l,1)-1 for l in layers[1:]]
        self.learning_rate = learning_rate
    
    #single image input
    #TODO: implement batches into first layer (staying consistent with rest of architecture)
    def load_image(self, img):
        if type(img) is Image.Image:
            inlayer = np.array(img) / 255
        elif type(img) is np.ndarray and len(img.shape) == 2:
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
        
    def train(self, _imgs, _labels, batch_size = 100, epochs = 1):
        imgs = copy.copy(_imgs)
        labels = copy.copy(_labels)
        for epoch in range(1,epochs+1):
            print("Epoch %s: %s training examples" % (epoch, batch_size))
            if len(imgs) < batch_size:
                imgs += _imgs
                labels += _labels
            imgs_to_train = imgs[:batch_size]
            labels_to_train = labels[:batch_size]
            imgs = imgs[batch_size:]
            labels = labels[batch_size:]
            avgcost = 0
            number_correct = 0
            avgweights = []
            avgbiases = []
            number_done = 0
            hyphens = 0
            print('|', end='')
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
                number_done += 1
                if (10*number_done)//batch_size > hyphens:
                    print('-', end='')
                    hyphens += 1
            print('|')
            avgcost /= batch_size
            avgweights = np.average(avgweights, axis=0)
            avgbiases = np.average(avgbiases, axis=0)
            self.apply_gradient((avgweights, avgbiases))
            print("    Average Cost: %s" % avgcost)
            print("    Percentage Correct: %s%%" % round(100*number_correct/batch_size, 3))
            with open('logfile.csv', 'a') as f:
                f.write('%s,%s,%s\n' % (epoch, avgcost, round(100*number_correct/batch_size, 3)))
        print("Training Complete!")
    
    def test(self, img):
        self.load_image(img)
        self.evaluate()
        return np.argmax(self.layers[-1].flatten().tolist())
            
