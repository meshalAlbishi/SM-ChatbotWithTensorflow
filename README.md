# SM-ChatbotWithTensorflow
the chatbot is ready in english
wait for the arabic version :')
wait for the UI version :')
in progress :)


## Installing Dependencies

#### Python 3.7
Follow instructions to install the latest version of python for your platform in the [python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)


#### Virtual Environment
We recommend working within a virtual environment whenever using Python for projects. This keeps your dependencies for each project separate and organized. Instructions for setting up a virtual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


#### PIP Dependencies
Once you have your virtual environment setup and cloned the repository, install dependencies by running:
``` 
pip install -r requirements.txt
```

#### Key Dependencies

- [Flask](http://flask.pocoo.org/) is a lightweight backend microservices framework.
- [tensorflow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning.
- [tflearn](http://tflearn.org/) is a modular and transparent deep learning library built on top of Tensorflow.
- [nltk](https://www.nltk.org/) is a leading platform for building Python programs to work with human language data.
- [NumPy](https://numpy.org/) is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

## Running the server locally in development environment

in the first clone the project 
```
git clone https://github.com/meshalAlbishi/SM-ChatbotWithTensorflow
```

navigate to the `root` of the project directory.

To run the server:

```
pip install -r requirements.txt
python main.py
```
you can now open the website suing:
- http://localhost:5000/
or 
- http://127.0.0.1:5000/


## Error Handling

The API have clear and defined errors that will make the debug process easier for developers.

### Error Types:

- 404 - Not Found
- 500 - Server Error

### Error Response Example:

```
{
    "success": False,
    "error": 404,
    "message": "Resource Not Found"
}
```
