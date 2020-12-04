wml_credentials = {
                      "apikey":"mKIhXojyhfky3PSK9p-puRPjs1-D7h81laZ3V8tobAGc",
                      "url": "https://eu-gb.ml.cloud.ibm.com"
}

from ibm_watson_machine_learning import APIClient, metanames

client = APIClient(wml_credentials)

# https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html
# https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-import-keras.html
# https://dataplatform.cloud.ibm.com/docs/content/wsj/wmls/wmls-deploy-python-types.html
#client.software_specifications.list()
wmltype = 'tensorflow_2.1-py3.7' 
#wmltype = 'tensorflow_1.15-py3.6' # for keras
software_spec_uid = client.software_specifications.get_id_by_name(wmltype)

"""
metadata = {
            client.spaces.ConfigurationMetaNames.NAME: 'AIDA-demo',
            client.spaces.ConfigurationMetaNames.DESCRIPTION: 'For my models'
            client.spaces.ConfigurationMetaNames.STORAGE: ''
            }
space_details = client.spaces.store(meta_props=metadata)
space_uid = client.spaces.get_uid(space_details)
"""
space_uid = '2931996b-8437-4f17-9126-5053ac8f9405'
# set the default space
client.set.default_space(space_uid)

# see available meta names for software specs
#https://github.com/IBM/watson-machine-learning-samples/blob/master/cloud/notebooks/python_sdk/deployments/tensorflow/Use%20Tensorflow%20to%20recognize%20hand-written%20digits.ipynb
print('Available software specs configuration:', client.software_specifications.ConfigurationMetaNames.get())
client.software_specifications.list()

sofware_spec_uid = client.software_specifications.get_id_by_name("default_py3.7")
metadata = {
            client.repository.ModelMetaNames.NAME: 'External Tensorflow model',
            client.repository.ModelMetaNames.TYPE: 'tensorflow_2.1',
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
}

published_model = client.repository.store_model(
    model='models/model_ftv2_RN50.tgz',
    meta_props=metadata)

import json

published_model_uid = client.repository.get_model_uid(published_model)
model_details = client.repository.get_details(published_model_uid)
print(json.dumps(model_details, indent=2))

models_details = client.repository.list_models()

metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Deployment of external Tensorflow model",
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

created_deployment = client.deployments.create(published_model_uid, meta_props=metadata)
deployment_uid = client.deployments.get_uid(created_deployment)

scoring_endpoint = client.deployments.get_scoring_href(created_deployment)
print(scoring_endpoint)
client.deployments.list()

client.deployments.get_details(deployment_uid)