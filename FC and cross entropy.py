import numpy as np

class Fully_connected_layer():
    def __init__(self, previous_activation_layer, weights, iter_num, biases, label):
        self.previous_activation_layer = previous_activation_layer
        self.weights = weights
        self.iter_num = iter_num
        self.biases = biases
        self.label = label

        self.output_tuple = np.zeros(10)
        self.output_tuple[self.label] = 1
        if self.iter_num == 0:
            self.weights = np.random.rand(10, np.shape(previous_activation_layer))

    def FC(self):
        predictions = np.zeros(10)

        for i in range(self.weights[0]):
            predictions[i] = np.dot(self.previous_activation_layer, self.weights[i]) + self.biases[i]

        return predictions

    def cross_entropy_loss(self):
        predictions = Fully_connected_layer.FC()
        predicted_value = max(predictions)

        predicted_label_index = predictions.index(predicted_value)

        if self.label == predicted_label_index:
             return -np.log(predicted_value)
        else:
             return -np.log(1 - predicted_value)


