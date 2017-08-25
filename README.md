Readme – Classifier

How to add an existing page type:

Currently the classifier only classifies three types of pages: Login, Registration/Sign-Up, Payment.  If you want to add one of these types of pages, follow these steps.

1.	Save the HTML of the page you want to classify as a unique .html inside of the directory of the type of page it is.  
a.	Example: if you have a page that you want to classify as a login page, save the html of the page as a .html inside the Login Pages folder.

2.	Once the html is saved inside the folder, you will need to add the name of the .html file inside of the .csv file of which ever type of page you are classifying.  
a.	Example: if you just saved the login page in the Login Pages folder as example.html, then inside the login_pages.csv, you will need to put example.html underneath the last value inside the .csv file.

3.	Each of the different page type has a one-hot vector that defines it: Login = [1 0 0], Registration = [0 1 0], Payment = [0 0 1].  Once the .csv of the page type as been appended, you will need to append to the expected_outputs.csv.  If you add a login page inside the expected_outputs.csv, you will add a row containing the values [1 0 0] underneath the last [1 0 0]. The same is to be done if it is a different type of page, but change the values respectively to what the page type should be.

How to add a new page type:

To add a new page, you will have to set up a few new files and a new directory.

1.	Make a new folder that will have similar htmls that represent the new page type and name it something on the lines of the new page type.

2.	You will have to make a new .csv that contains the new html names of the html files inside of the directory that was created.

3.	Once that has been done you can repeat steps 2 and 3 from above.

4.	This step, it is very similar, but now you will have to extend the column values.  If you only classify one new page type, then instead of [1 0 0], it will be [1 0 0 0]. So, inside of the expected_outputs.csv you will have to add an extra column value of 0 to every existing page type except for the new page type in which there will be a 1 at the end of the row.

5.	You will have to also alter app.py to account for the new inputs.  Inside app.py there are a couple of variables already that represent the directory of certain page types.  You will have to create a new variable that represents the new directory that you created.

6.	You will also have to create a new variable that represents the name of the .csv file that you have created.

7.	Once that is finished, you will need to call the load_data method from the data handler object that takes in the directory and the file name and appends the data to the class’ inputs and outputs.

8.	You are finished setting up the new data.


Running app.py

You will need to make an environment variable once per new terminal window:

export FLASK_APP=app.py

Then to run the application:

	python –m flask run
