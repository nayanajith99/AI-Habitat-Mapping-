# AI-Habitat-Mapping
This Model is medium accuracy AI Model for habitat mapping 

## supported datasets
this project natively supports followinf datasets in YOLOv8 format
1. [FLAIR Challenge](https://ignf.github.io/FLAIR/) | [original dataset in kaggle (unofficial)](https://www.kaggle.com/datasets/uom200399g/flair-dataset/) | [processed dataset](https://www.kaggle.com/datasets/uom200399g/flair-dataset/)

## file info
* you can find all notebooks used in notebooks folder
* and all config files in configs golder
* predict.py - can predict the areas of interest in a image with a trained model
* train.py - can be used to train a new model (literally train any YOLOv8 model with right config file)

## demonstrations

you can find a working example of the work [here](https://huggingface.co/spaces/mohotta/HabitatMapping)

you can run the pretained model with gradio app by cloning the hugging face repo and installing requirements using the requirements.txt file. 
1. clone the project
```
git clone https://huggingface.co/spaces/mohotta/HabitatMapping
```
2. cd into the project
```
cd HabitatMapping
```
3. install requirements
```
pip install -r requirements.txt
```
4. run the app
```
python app.py 
```
