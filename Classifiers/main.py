from data_handler import DataHandler
from html_encoder import *
from randomforestclassifier import RFC

handler = DataHandler()
clf = RFC()

# The Data
expected_output = handler.convert_to_matrix('three_outputs.csv')
login_files = handler.convert_to_matrix('login_pages.csv')                 
reg_files = handler.convert_to_matrix('registration_pages.csv') 
payment_files = handler.convert_to_matrix('payment_pages.csv') 

# Directories
cur_dir = os.getcwd()
login_dir = cur_dir + "/Login_Pages/"
reg_dir = cur_dir + "/Registration_Pages/"
payment_dir = cur_dir + "/Payment_Pages/"

# Load Data
handler.load_data(login_dir, login_files, expected_output)
handler.load_data(reg_dir, reg_files, expected_output)
handler.load_data(payment_dir, payment_files, expected_output)

# Transfer Data
training_input, testing_input = handler.getinputs()
training_output, testing_output = handler.getoutputs()

predictions, classifer = clf.findbestestimator(training_input, training_output, testing_input, testing_output)
