# -*- coding: utf-8 -*-
import os

"""
Download the artist dataset using the Kaggle API. In order to use the Kaggle’s public API, 
you must first authenticate using an API token. From the site header, click on your user profile picture, 
then on “My Account” from the dropdown menu. 
This will take you to your account settings at https://www.kaggle.com/account. 
Scroll down to the section of the page labelled API:

To create a new token, click on the “Create New API Token” button. 
This will download a fresh authentication token onto your machine. Upload `kaggle.json` into the main directory.
"""

os.environ["KAGGLE_CONFIG_DIR"] = "C:\\Users\\Paul\\Documents\\AIDA\\keras-flask-deploy-webapp"

"""or set username and API-Key as enviroment variables

```
os.environ['KAGGLE_USERNAME'] = 'paulbauriegel'
os.environ['KAGGLE_KEY'] = '...'
```

Authenticate and Download Dataset from Kaggle
"""

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('puneet6060/intel-image-classification', unzip=True, path='dataset')