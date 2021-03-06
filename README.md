# Predicting the probability of a mortgage default
Given a private dataset, transform the data so that it is usable for an XGBoost classifier and try to predict whether a client will or will not default in the next twelve months.

There is a logistic regression model written and a scorecard script, both written in R and located in the R folder.

There is also an XGBoost model written in Jupyter using Python.

# How to run the notebooks

1. Go to the folder<br>
  *cd C:\path\to\repo\mortgage-default-prediction*
2. Create a virtual environment (Windows)<br>
  *virtualenv --python C:\Path\To\Python\python.exe venv*<br>
(Linux)<br>
  *python -m venv [directory]*
3. Activate the virtual environment (Win)<br>
  *.\venv\Scripts\activate*<br>
(Linux)<br>
  *source myvenv/bin/activate*
4. Once you're in, you need to install the libraries in the requirements.txt file<br>
  *pip install -r requirements.txt*

You're all set to go! :)

# Different notebooks

Currently, the notebooks have current functions

0. Look at the structure of the data
1. Transforms the data into a more workable shape (using random rows from twelve month windows, to make sure that entries don't repeat)
2. Simple model using two parameters
3. A more advanced model using basic feature engineering

Better models with more advanced features will be added until the project deadline

# Disclaimer

The data for this analysis is private, so it is included in the .gitignore file. You can try to replicate these results by uploading it and running the scripts

# Credits

Credits to Arba, Polina and Arina 
