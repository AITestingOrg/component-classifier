import numpy as np
import random
from html.parser import HTMLParser
from sklearn.ensemble import RandomForestClassifier

class RFC:
    def __init__(self):
        np.random.seed(10)
        self.clf = RandomForestClassifier()
        
    def findbestestimator(self, training_input=None, training_output=None, testing_input=None, testing_output=None):  
        self.clear_page_count()  
        maxacc = 0
        num_est = 0
        classifer = None
        best_predictions = None
        e = 0
        num_iters = 0
        percentages = None
        
        print("Finding best parameters for data...")
        for n in range(1):  
            n = n + 1  
            clf = RandomForestClassifier(n_estimators= 4)
            for iters in range(100):
                clf.fit(training_input, training_output)
                predictions = clf.predict(testing_input)
                correct = 0
                total = 0
                
                for i in range(len(predictions)):
                    if i != 0 and predictions[i][np.argmax(predictions[i])] == 0:
                        predictions[i][np.argmax(predictions[i-1])] = 1
                        
                    if predictions[i][np.argmax(predictions[i])] == 0:
                        total += 1
                        continue
                        
                    elif np.argmax(predictions[i]) == np.argmax(testing_output[i]):
                        correct += 1
                        self.increase_page_count(predictions[i])
                            
                    else:
                        self.increase_page_count(predictions[i])
                            
                    total += 1

                if (correct/total) * 100 > maxacc:
                    maxacc = (correct/total) * 100
                    self.clf = clf
                    num_est = n
                    best_predictions = predictions
                    num_iters = i
                        
        self.clear_page_count()
        
        print("Finished finding data")   
        print("")     
        print("Max Accuracy: " + str(maxacc) + " with " + str(num_est) + " estimator(s) and " + str(i) + " iteration(s).")
        print(self.clf)
        return [best_predictions, self.clf]
    
    def clear_page_count(self):
        self.correct_login = 0
        self.correct_reg = 0
        self.correct_pay = 0
        
        self.total_login = 0
        self.total_reg = 0
        self.total_pay = 0
            
    def increase_page_count(self, prediction):
        if np.argmax(prediction) == 0:
            self.total_login += 1
            
        elif np.argmax(prediction) == 1:
            self.total_reg += 1
                
        elif np.argmax(prediction) == 2:
            self.total_pay += 1
            
    def test_single_page(self, filename, parser, encoder):
        testing_input = list();
        testfile = open(filename, 'r')
        html = testfile.readlines()
        for t in range(len(html)):
            parser.feed(html[t])
        testing_input.append(encoder.encode(parser.getTags()))  
        prediction = self.clf.predict(testing_input)
        
        if(np.argmax(prediction)) == 0:
            print("Login Page")
            return 'Login Page'
            
        elif(np.argmax(prediction)) == 1:
            print("Registration Page")
            return 'Registration Page'
            
        elif(np.argmax(prediction)) == 2:
            print("Payment Page")
            return 'Payment Page'
            
    def test_raw_html(self, html, parser, encoder):
        testing_input = list();
        for t in range(len(html.split())):
            parser.feed(html[t])
        testing_input.append(encoder.encode(parser.getTags()))  
        prediction = self.clf.predict(testing_input)
        
        if(np.argmax(prediction)) == 0:
            print("Login Page")
            return 'Login Page'
            
        elif(np.argmax(prediction)) == 1:
            print("Registration Page")
            return 'Registration Page'
            
        elif(np.argmax(prediction)) == 2:
            print("Payment Page")
            return 'Payment Page'
        