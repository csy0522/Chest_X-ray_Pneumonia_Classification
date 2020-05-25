# Chest X-ray Pneumonia Classification

This project classifies Lung X-ray images to three different categoris: <br/>
1. Normal
2. Pneumonial-Bacteria
3. Pneumonial-Virus

## Getting Started

Use any python IDE to open the project. I personally use Jupyter Notebook from Anaconda, but a good alternative is Google Colab. You can download both Anaconda and Jupyter Notebook from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Jupyter Notebook](https://jupyter.org/) - An Open-source Web Application
For more about Google Colab, go to:
* [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) - Free Jupyter notebook environment that runs entirely in the cloud.

### Data

The data for this project is available on Kaggle which a huge community space for data scientists. Click the following link to download the dataset:
* [X-ray Image Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia?) - Kaggle X-ray Dataset<br/>
* The data is also available on the **data** repository.

### Installation

Before running the program, type the following command to install the libraries that the project depends on

```
pip install numpy, matplotlib, keras, opencv-contrib-python
```
Or simply type the following:

```
pip install -r requirements.txt
```

## Running the tests

- The description of each function is located on top of them. Please read them before running to understand the overall structure of the project. <br/>
- This project uses 2 different models to classify a bundle of lung x-ray images to 3 different categoris.<br/>
- The following shows the prediction from both models:

![Prediction](/data/prediction.png)

- The left side shows the prediction made by Artificial Neural Network (Model 1)<br/>
- The right side shows the prediction made by Convolutional Neural Network (Model 2)<br/>
- These two only show the first part of predictions. The entire prediction is at the bottom of **main.ipynb**.<br/>
- For more detail, please read the descriptions on top of each function, and go to **main.ipynb**. The Neural_Network class is designed to let the users customize their own model. Let me know if you can come up with a model that givse 100% accuracy!.<br/>
- I also added a **py** file for the main functionin in the **src** directory if you want to run it using IDE

## Deployment

Download other dataset from online (Ex: Kaggle) and insert the data to the model in order to test its accuracy.
* [Kaggle](https://www.kaggle.com/) - The Machine Learning and Data Science Community

## Built With

* [Python](https://www.python.org/) - The Programming Language

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
