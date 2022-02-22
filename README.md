# Predicting the probability of a mortgage default
Given a private dataset, transform the data so that it is usable for an XGBoost classifier and try to predict whether a client will or will not default in the next twelve months.

# How to run notebooks located here

1. Go to the folder
> cd C:\path\to\repo\mortgage-default-prediction
2. Create a virtual environment (Windows)
> virtualenv --python C:\Path\To\Python\python.exe venv
(Linux)
> python -m venv [directory]
3. Activate the virtual environment (Win)
> .\venv\Scripts\activate
(Linux)
> source myvenv/bin/activate

You are all set to go! :)

# Different notebooks

Currently, the notebooks have current functions

0. Look at the structure of the data
1. Transforms the data into a more workable shape (using random rows from twelve month windows, to make sure that entries don't repeat)
2. Simple model using two parameters
3. A more advanced model using basic feature engineering

And more notebooks that have more advanced models will be added

# Disclaimer

The data for this analysis is private, so it is included in the .gitignore file. You can try to replicate these results by uploading it and running the scripts
