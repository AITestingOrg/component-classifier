#semantic memory GloVe
import numpy as np

class GloveData():
    def __init__(self):
        self.embeddings = {}
    
    def loadGloveModel(self, gloveFile):
        print ("Loading Glove Model")
        f = open(gloveFile,'r')
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            self.embeddings[word] = embedding
        
    def generateModuleEmbeddings(self, string):
        stringEmbeddings=[[self.embeddings[w.lower()] for w in s.split(' ') if w.lower() in self.embeddings] for s in string.split('\n')]
        return stringEmbeddings
        
    def to_numpy_matrix(self, string):
        matrix = self.generateModuleEmbeddings(string)
        numpy_matrix = np.zeros((len(matrix),), dtype=object)
        for m in range(len(matrix)):
            numpy_matrix[m] = np.append(None, matrix[m])
            numpy_matrix[m] = np.delete(numpy_matrix[m], 0)
        return numpy_matrix