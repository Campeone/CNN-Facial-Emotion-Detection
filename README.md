# Facial Emotion Detection with Flask Web App
==============================

## Overview
Facial Emotion Detection is a deep learning project aimed at developing a Convolutional Neural Network (CNN) for facial emotion recognition. The Computer Vision project was trained to accurately detect and classify facial expressions such as "Anger", "Happiness", and "Sadness", leveraging datasets with labeled facial images.The repository contains the Train and Test datasets, implementation code, the trained model, and the Flask deployment code.

Users can upload images containing faces, and the application will detect and classify the emotions present in each face.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## Features
- Upload Image: Users can upload images containing faces.
- Emotion Detection: The application detects facial expressions and classifies them into different emotions such as happy, sad, angry, etc.
- Display Results: Detected emotions are displayed alongside the uploaded image for user visualization.
- User-Friendly Interface: The Flask web app provides a simple and intuitive interface for users to interact with.

## Technologies Used
- Python
- TensorFlow, Keras: Deep learning frameworks for training and deploying the emotion detection model.
- OpenCV: Image processing library for face detection and manipulation. 
- Pandas: For data preprocessing.
- Matplotlib: Image visualization and plotting.
- Flask: Web framework for developing the user interface.
- HTML/CSS: Frontend design and layout.

## Files in Repository
- `FaceNet.py`: Flask application script containing the backend logic for image processing and emotion detection.
- `templates/`: Directory containing HTML templates for the web application.
- `static/`: Directory containing static files such as CSS stylesheets and JavaScript scripts.
- `model/’: Trained CNN deep learning model for facial emotion detection.
- `notebooks/’: Directory containing jupyter notebook file for training the deep learning model for facial emotion detection.
- `README.md`: Readme file providing an overview of the project and instructions for replication.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python FaceNet.py`.
4. Access the web application through your browser at `http://localhost:5000`.
5. Upload an image containing faces and observe the detected emotions.

## Example
![Facial Emotion Detection Web App](example.png)

## Future Enhancements
- Improve model accuracy and robustness through further training and fine-tuning.
- Implement real-time video emotion detection for live webcam feeds.
- Enhance the user interface with additional features such as image cropping and resizing.

Feel free to reach out with any questions or feedback!

**Author:** Ojo Timilehin 
**Contact:** Ojotimilehin01@gmail.com

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
