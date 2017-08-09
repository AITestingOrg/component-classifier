from html_encoder import *
import numpy as np
import sys
import pandas as pd

class DataHandler:
    def __init__(self):
        self.encoder = HTMLEncoder()
        self.parser = MyHTMLParser()
        self.parser.clear()
        self.split = .5
        self.training_input = list()
        self.training_output = list()
        
        self.testing_input = list()
        self.testing_output = list()
        
        self.testing_index = 0
    
    def convert_to_matrix(self, csv):
        data = pd.read_csv(csv, sep =',', header = None)
        data = data.as_matrix() 
        return data
        
    def load_data(self, directory, files, expected_output):  
        try:
            for i in range(len(files)):
                F = open(directory + files[i][0], 'r')
                html = F.readlines()
                
                #Feed all of the html into parser
                for h in range(len(html)):
                    self.parser.feed(html[h])
                
                if i < len(files)*self.split:
                    self.training_input.append(self.encoder.encode(self.parser.getTags()))
                    self.training_output.append(expected_output[self.testing_index])
                    
                else:
                    self.testing_input.append(self.encoder.encode(self.parser.getTags()))
                    self.testing_output.append(expected_output[self.testing_index])
                    
                self.parser.clear()
                self.encoder.clear()
                self.testing_index += 1
                
        except Exception as e:
            print("Error occured in load data: " + str(e))
    
    def set_split(self, percent):
        if percent > 1:
            self.split = percent / 100
        else:
            self.split = percent
                    
    def getinputs(self):
        return [self.training_input, self.testing_input]
        
    def getoutputs(self):
        return [self.training_output, self.testing_output]
        