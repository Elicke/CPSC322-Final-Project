# Spotify Hit Prediction Project
### Contributors: Kyle Manning
### Class: CPSC 322, Spring 2022
## Project Description
This project revolves around the classification task of predicting whether Spotify tracks are 'Hits' or 'Flops' based on a number of different attributes.
## Project Organization
- input_data contains the .csv dataset files pertaining to each different decade
- mysklearn contains the source code for different classifiers, different evaluation functions, the MyPyTable class, as well as different utility and plotting functions
- hit_flop_app.py is the file for the predict API endpoint created via a Flask app. This file uses the nb.p file that was created by naive_bayes_pickler.py
- project_proposal.ipynb contains... well... the project proposal
- mid_project_demo.ipynb contains... well... the mid-project demo
- project_report.ipynb contains the full Jupyter Notebook technical report
## How to Use the Flask Web App
1. Run the 'hit_flop_app.py' file
2. Go to 'http://127.0.0.1:5001/predict?danceability=0.652&energy=0.698&valence=0.47&tempo=96.021' in your web browser, replacing the attribute values as appropriate
3. '1.0' signifies that 'Hit' is predicted, while '0.0' signifies that 'Flop' is predicted