### Contents

This folder contains the code necessary to run the server which returns requests for obtaining the image captions of the images on a webpage

- finalized_model.sav contains the weights for loading in the model
- pipeline.py contains the various pipeline options for obtaining image captions
- server.py is the main file for running the server and taking requests

### Running the server

First create a virtual environment using: `python3 -m venv env`

Then activate the environment with: `source env/bin/activate`

Install the requirements with: `pip install -r requirements.txt`

Finally run the server with: `python server.py`

