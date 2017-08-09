import numpy as np
import random

class Trainer:
    def __init__(self, model):
        self.model = model
        
    def train(self, epochs, inputs, outputs):
        inputs, outputs = self.shuffle(inputs, outputs)
        
        for e in range(epochs):
            rand_index = int(random.randrange(len(inputs)))
            
            training_input = np.empty([1, len(inputs[0])])
            training_input[0] = inputs[rand_index]
            
            training_output = np.empty([1, len(outputs[0])])
            training_output[0] = outputs[rand_index]
            self.model.sgd_step(training_input, training_output, .01)
            
    def shuffle(self, ins, outs):
        inholder = np.zeros_like(ins)
        outholder = np.zeros_like(outs)
        previous = {}
        for i in range(len(ins)):
            rand_index = int(random.randrange(len(ins)))
            if(rand_index in previous):
                while(rand_index in previous):
                    rand_index = int(random.randrange(len(ins)))
            inholder[i] = ins[rand_index]
            outholder[i] = outs[rand_index]
            previous[rand_index] = 1
        return inholder, outholder  
          
    def test(self, training_input, expected_output):
        # Test Data that was trained on        
        total_num = 0
        total_correct = 0
        
        correct_login = 0
        correct_registration = 0
        correct_payment = 0
        
        for i in range(len(training_input)):
            total_num += 1
            inputs = np.empty([1, len(training_input[i])])
            inputs[0] = training_input[i]

            test_out = self.model.forward_propagation(inputs)
                
            if np.argmax(test_out[0][0]) == np.argmax(expected_output[i]):
                total_correct += 1
                
                if np.argmax(expected_output[i]) == 0:
                    correct_login += 1
                    
                if np.argmax(expected_output[i]) == 1:
                    correct_registration += 1
                        
                if np.argmax(expected_output[i]) == 2:
                    correct_payment += 1
                
            print("Accuracy: " + str((total_correct/total_num) * 100) + "%")
        
        # print("Logins Correct out of 8: " + str(correct_login))
        # print("Registrations Correct out of 6: " + str(correct_registration))
        # print("Payments Correct out of 8: " + str(correct_payment))
        