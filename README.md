# Chest X-ray Pneumonia Classification

This project classifies Lung X-ray images to three different categoris: <br/>
1. Normal
2. Pneumonial-Bacteria
3. Pneumonial-Virus

## Getting Started

Use any python IDE to open the project. I personally use Jupyter Notebook from Anaconda.You can download both Anaconda or Jupyter Notebook from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Jupyter Notebook](https://jupyter.org/) - An Open-source Web Application

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

![ANN Model Prediction]() ![ANN Model Prediction]()

- The blue line is the actual stock price, and the red line is the prediction. <br/>
- For more detail, please read the descriptions on top of each function, and run **main.py**. Make sure to run it from an ide that's able to show graphs. The output will show more deails, including accuracy, loss, and more graphs.<br/>
- I also added a **ipynb** file for the main functionin in the **src** directory if you want to run it using [Jupyter Notebook](https://jupyter.org/)

## Deployment

Download other dataset from online (Ex: Kaggle) and insert the data to the model in order to test its accuracy.
* [Kaggle](https://www.kaggle.com/) - The Machine Learning and Data Science Community

## Built With

* [Python](https://www.python.org/) - The Programming Language
* [Keras](https://keras.io/) - The Python Deep Learning library

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
