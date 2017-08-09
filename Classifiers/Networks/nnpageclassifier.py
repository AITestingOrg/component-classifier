import numpy as np
import sys
import pandas as pd
import os
import random
from html.parser import HTMLParser

def trainWithInput(epochs, model, inputs, outputs):
    for e in range(epochs):
        rand_index = int(random.randrange(len(inputs)))
        
        training_input = np.empty([1, 142])
        training_input[0] = inputs[rand_index]
        
        training_output = np.empty([1, 3])
        training_output[0] = outputs[rand_index]
        model.sgd_step(training_input, training_output, .01)
            
def decodeArray(array):
    maxPercent = -1
    maxIndex = -1
    for i in range(len(array)):
        if array[i] > maxPercent:
            maxPercent =  array[i]
            maxIndex = i
    
    if maxIndex == 0:
        return "None"
        
    elif maxIndex == 1:
        return "Login Page"
        
    elif maxIndex == 2:
        return "Registration Page"
    
    elif maxIndex == 3:
        return "Payment Page"
        
    else:
        return "Not recognized" 

expected_output = pd.read_csv('expected_output.csv', sep =',', header = None)
expected_output = expected_output.as_matrix().T

#model
model = RNN(142, 100, len(expected_output[0]))

try:
    U = pd.read_csv('U.csv', sep =',', header = None)
    U = U.as_matrix()

    V = pd.read_csv('V.csv', sep =',', header = None)
    V = V.as_matrix()

    W = pd.read_csv('W.csv', sep =',', header = None)
    W = W.as_matrix()

    model.setUpWeights(U, V, W)

except Exception as e:
    print("No previous weights found")

np.random.seed(10)
                   
#Registration_Pages Data
reg_files = pd.read_csv('registration_pages.csv', sep =',', header = None)
reg_files = reg_files.as_matrix()

#Login_Pages Data
login_files = pd.read_csv('login_pages.csv', sep =',', header = None)
login_files = login_files.as_matrix()

#Payment_Pages Data
payment_files = pd.read_csv('payment_pages.csv', sep =',', header = None)
payment_files = payment_files.as_matrix()

handler = DataHandler()

cur_dir = os.getcwd()
login_dir = cur_dir + "/Login_Pages/"
reg_dir = cur_dir + "/Registration_Pages/"
payment_dir = cur_dir + "/Payment_Pages/"

#Create Training Data
training_input = list()

handler.load_data(login_dir, login_files, training_input)
handler.load_data(reg_dir, reg_files, training_input)
handler.load_data(payment_dir, payment_files, training_input)
    
# trainWithInput(10000, model, training_input, expected_output)
# model.saveWeights()

F = open('test.html', 'r')
html = F.readlines()
test_input = list()

for i in range(len(html)):
    parser.feed(html[i])

test_input.append(encoder.encode(parser.getTags()))
parser.clear()

print(test_input)

test_out = model.forward_propagation(test_input)
print(test_out[0][0])

print(decodeArray(test_out[0][0]))
