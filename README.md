# FEDn implementation of cifar10 with keras 
This repository contains an example client (and compute package and seed file) for FEDn training of a Keras VGG16 model using CIFAR-10.
This example is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode.
## Setting up a client
###Local training and test data
To download and prepare the partitioned dataset (creates N partitions) in ./data/clients/0 etc:
```bash
python client/Load_data.py {NR_OF_CLIENTS}
```

[comment]: <> (### Initiate your federation)

[comment]: <> (This command loads and partitions the dataset, generates a )

[comment]: <> (docker-compose.yaml for your choice of numbers of clients and)

[comment]: <> (initiates a seed model to start from:)

[comment]: <> (&#40;Replace: {NR_OF_CLIENTS} with the number of clients you want to build you federation with&#41;)

[comment]: <> (```bash)

[comment]: <> (pip install -r init_requirements.txt)

[comment]: <> (python init_federation.py {NR_OF_CLIENTS})

[comment]: <> (```)




## Configuring the Reducer  
Navigate to 'https://localhost:8090' (or the url of your Reducer) and follow instructions to upload the compute package in 'package/package.tar.gz' and the initial model in 'initial_model/initial_model.npz'. 

## Creating a compute package
Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the compute package:

```bash
tar -czvf package.tar.gz client
```
To clear the system and set a new compute package, see: https://github.com/scaleoutsystems/fedn/blob/master/docs/FAQ.md

For an explaination of the compute package structure and content: https://github.com/scaleoutsystems/fedn/blob/develop/docs/tutorial.md
 
## Creating a new initial model
The baseline model (VGG16) is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serializes that to a file.  If you wish to alter the initial model, edit 'init_model.py' and 'models/imdb_model.py' then regenerate the initial model file (install dependencies as needed, see requirements.txt):

```bash
python init_model.py 
```
### Configuring the client
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are expsosed in the file 'settings.yaml': 

```yaml 
# Parameters for local training
test_size: 0.25
batch_size: 32
epochs: 1
#trained layers 0 means all layers in the model, otherwise just select the layers based on the identified number
trained_Layers: 6
```

## Attaching a client to the federation

1. First, download 'client.yaml' from the Reducer 'Network' page, and replace the content in your local 'client.yaml'. 
2. Start a client. Here there are different options (see below): 
    - Docker 
    - docker-compose
 
#### docker-compose
To start N clients: 

```bash
docker-compose -f docker-compose.dev.yaml -f extra-hosta.yaml up --build 
```
### Start training 
When clients are running, navigate to the 'Control' page of the Reducer to start the training. 




## License
Apache-2.0 (see LICENSE file for full information).
