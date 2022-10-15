import numpy as np
from scipy import signal
from math import sqrt

input_ = np.random.rand(28, 28)
kernel = np.random.rand(4, 4)
output_gradient = np.random.rand(28, 28)
parameters = np.random.rand(28, 28)
biases = 0.4
learning_rate = 0.001
depth = 1
iteration_number = 1
stepsize = 1

class ADAM():
    
    def __init__(self, output_gradient, parameters, stepsize, input_, kernel, biases, learning_rate, depth, iteration_number):
        # output_gradient is the gradient from the previous layer further in the network
        # hyperparameters are stepsize, learning rate, and iteration number
        self.output_gradient = output_gradient
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.input_ = input_
        self.biases = biases
        self.depth = depth
        self.parameters = parameters
        self.stepsize = stepsize
        self.iteration_number = iteration_number
    
    #def valid_cross_correlate(input_, output_gradient):
        #shape_input = np.shape(input_)
        #shape_output_gradient = np.shape(output_gradient)

                

        #This is to calculate the gradient of the inputs from the previous layer(pretty sure they are the input parameters so the image after convolution)
        #feed it as output_gradient to do the next layer etc. until all weights are upated and another forward pass can commence
        #Called internally by ADAM
    def backpropagated_gradient(self):
        kernel_gradient = np.zeros(np.shape(self.kernel))
        input_gradient = np.zeros(np.shape(self.output_gradient))
        
        # signal.correlate2d and signal.convolve2d need to be done manually as they dont't work with 1d rows or columns (needs 2d)
        # since we pass one image at a time we don't have a tensor with depth so 1D slices don't grant 2D images

        #for i in range(self.depth):
        #for i in range(np.shape(self.input_)[0]):
            #for j in range(np.shape(self.input)[1]):
        kernel_gradient = signal.correlate2d(self.input_, self.output_gradient, 'valid')
        input_gradient = signal.convolve2d(self.output_gradient, self.kernel, 'full')
                #kernel_gradient[i, j] = signal.correlate2d(self.input_[j], self.output_gradient[i], 'valid')
                #input_gradient[j] += signal.convolve2d(self.output_gradient[i], self.kernel[i, j], 'full')
        
        # below 2 lines may not be needed
        self.kernel -= self.learning_rate * kernel_gradient
        self.biases -= self.learning_rate * self.output_gradient

        return input_gradient    

    #call this to optimize using adam
    #outputs parameters (image after convolution on a backward pass) tweeked according to adaptive moment estimation and gradient descent
    def adam_optimization(self):
        Beta1 = 0.9
        Beta2 = 0.999
        epsilon = 10**-8
        
        m_moment = np.zeros(np.shape(self.backpropagated_gradient()))
        v_moment = np.zeros(np.shape(m_moment))
        
        for t in range(self.iteration_number):
            gt = self.backpropagated_gradient()

            print(gt) #TEST LINE
            print("gradient 1" + str(np.shape(gt)[0]))
            print("gradient 2" + str(np.shape(gt)[1]))

            for i in range(np.shape(self.parameters)[0]):
                for j in range(np.shape(self.parameters)[1]):
                    m_moment[i, j] = Beta1 * m_moment[i, j] + (1 - ((Beta1*10)*gt[i, j]/10))
                    v_moment[i, j] = Beta2 * v_moment[i, j] + (1 - (((Beta2*1000)**1)/1000))*(gt[i, j]**2)
                    m_hat =  m_moment[i,j]/(1 - (((Beta1*10)/10)**(t + 1)))
                    v_hat =  v_moment[i,j]/(1 - (((Beta2*1000)**1)/1000)**(t + 1))
                   #print("D6 " + str(v_moment[i,j]))
                   #print("D7 " + str(m_moment[i,j]))
                   #print("D8 " + str((pow(Beta1, t))))
                    #print("D9 " + str(((Beta2*1000)**1)/1000)) 
                   #print("D4 " + str(epsilon))
                   #print("D5 " + str(sqrt(v_hat)))
                   #print("D1 " + str(sqrt(v_hat) + epsilon))
                   #print("D2 " + str(self.stepsize*m_hat / (sqrt(v_hat) + epsilon)))
                   #print("D3 " + str(self.parameters[i, j] - self.stepsize*m_hat / (sqrt(v_hat) + epsilon)))
                    self.parameters[i, j] = self.parameters[i, j] - ((self.stepsize * m_hat) / (sqrt(v_hat) + epsilon))
                    #print(self.parameters[i,j])
        #final output are the parameters/weights that have been optimized
        return self.parameters

weights = ADAM(output_gradient, parameters, stepsize, input_, kernel, biases,learning_rate, depth, iteration_number).adam_optimization()

print(weights)
print(np.shape(weights)[0])
print(np.shape(weights)[1])

#a = [[1,2,3], [4,5,6], [7,8,9]]
#b = [[1,1,1], [1,1,1], [1,1,1]]

#c = signal.correlate2d(a, b, 'valid')

#print(c)

#d = signal.convolve2d(a, b, 'full')

#print(d)
