
from rnn import GRU
from memory import Ensemble
from trainer import Trainer
from data_handler import *
from glove import *

handler = DataHandler()

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

story = """Matt is sitting right next to Bryan 
Bryan is in a cubical and he is holding a pen
The pen has blue ink, but writes poorly
Good thing Matt bought a new pen that is red
Bryan then went to the bathroom 
Since he forgot to close his computer, he got hacked"""

questions = """Where is Matt?
Where is Bryan?
Which pen writes poorly?
Which is the better pen?
What did Matt buy?
How many pens are there?"""

answers = """cubical
bathroom
blue pen
red pen
red pen
two"""

data = GloveData()

print("Started Generation...")
data.loadGloveModel("glovedata.txt")
print("Finished Generation")
print("")

print("Converting arrays to embedding matricies...")
print("")
s = data.to_numpy_matrix(story)
q = data.generateModuleEmbeddings(questions)
a = data.generateModuleEmbeddings(answers)
q = data.to_numpy_matrix(questions)

episodic = Ensemble(len(s), 150, len(a))
episodic.forward(training_input, training_output)

