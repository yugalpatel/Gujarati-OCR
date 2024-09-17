# Gujarati OCR Project

## Overview
This project is an Optical Character Recognition (OCR) system specifically designed to recognize characters from the Gujarati script. The goal was to create a robust deep learning model that can accurately identify and classify Gujarati characters from typed text using a custom convolutional neural network architecture. Achieving an impressive 99% accuracy, this project demonstrates the capabilities of machine learning in tackling non-Latin scripts, particularly regional languages like Gujarati, which are often underrepresented in the OCR domain.

The system is designed to handle various font styles and modifiers in the Gujarati language, making it useful for text digitization, archival, and translation services in regional scripts.

<p align="center">
  <img width="400" alt="Screenshot 2024-09-16 at 3 29 11 PM" src="https://github.com/user-attachments/assets/c1039577-c663-4ff6-a1ed-a876932e76ec">
  <img width="400" alt="Screenshot 2024-09-16 at 3 29 31 PM" src="https://github.com/user-attachments/assets/11c4792e-7375-4b05-aa15-bc507d0bc940">
</p>


## Motivation
With the growing importance of preserving and digitizing regional languages, there is a need for advanced OCR systems capable of recognizing non-Latin scripts. Gujarati, a widely spoken language in India, presents challenges such as complex modifiers (Maatra), joint characters, and varied fonts. This project was motivated by the desire to bridge this gap by developing a system that can accurately interpret and digitize Gujarati script, thus contributing to the digitization of regional languages and ensuring their preservation in the digital age.

## Project Features
- Recognition of 385 unique Gujarati characters including modifiers, vowels, and joint characters.
- Handles different fonts such as "Shruti" and "TERAFONT-VARUN", with variations like bold and italic.
- Achieves 99% accuracy in character recognition using the EfficientNetB3 architecture.
- Supports image preprocessing and augmentation techniques like rotation, zooming, and brightness shifts to enhance model robustness.
- Capable of classifying both basic and joint Gujarati characters with impressive precision.

## Model Architecture
The model is based on **EfficientNetB3**, a state-of-the-art convolutional neural network architecture optimized for both performance and computational efficiency. EfficientNetB3 was chosen for its ability to scale depth, width, and resolution uniformly, making it an excellent fit for image-based tasks like OCR. The model was fine-tuned on a highly diverse dataset of Gujarati characters with data augmentation to improve generalization. Custom layers were added to account for the specific nuances of Gujarati script, such as its modifiers and joint characters. Through this approach, the model achieves an outstanding 99% accuracy.

## Dataset
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/ananddd/gujarati-ocr-typed-gujarati-characters/data). It contains images of Gujarati characters in "Shruti" and "TERAFONT-VARUN" fonts, each in various style variations. Specifically, the dataset includes:
- 374 different characters (34 base characters with 11 modifier variations)
- 11 separate vowels
- 35 joint characters, each formed by combining two basic characters
- Image augmentation (horizontal-vertical shift, zooming, rotation, brightness) to introduce variability

All images are 32x32 in size, and the characters are typed, not handwritten, making the dataset ideal for recognizing printed Gujarati script.

## Results
The model achieved a **99% accuracy** on the test set, demonstrating its strong ability to recognize and classify Gujarati characters across different fonts, styles, and modifiers. This high accuracy was made possible by the use of EfficientNetB3, extensive data augmentation, and the richness of the dataset.

Key Performance Metrics:
- **Accuracy**: 99%
- **Loss**: Minimal validation loss after hyperparameter tuning
- **Generalization**: The model was tested on unseen characters and modifiers, showcasing its robustness in handling variations in the Gujarati script.

## Instructions to Run Locally
To run the Gujarati OCR project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yugalpatel/Gujarati-OCR.git
   cd Gujarati-OCR
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** (optional):
   If you want to train the model from scratch, download the dataset from Kaggle:
   [Gujarati OCR Dataset](https://www.kaggle.com/datasets/ananddd/gujarati-ocr-typed-gujarati-characters/data)

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000` to start using the Gujarati OCR web interface.

## Technologies Used
- **Python**: The core language used for development.
- **Flask**: A micro web framework used for creating the web interface.
- **TensorFlow**: For deep learning model development and training.
- **EfficientNetB3**: Convolutional Neural Network used for image classification.
- **Pandas**: For managing and manipulating data.
- **OpenCV**: For image processing and preprocessing.
- **PIL**: Python Imaging Library for handling image file formats.
- **Kaggle Notebooks**: For model training and experimentation.
  
## Future Work

- **Recognizing Strings of Characters**: Extending the model to recognize entire strings of Gujarati characters rather than individual characters, which would enable full word and sentence recognition, making the system suitable for document digitization and translation.
- **Real-time OCR**: Integrating the model into a mobile or desktop application for real-time character recognition from scanned documents or photos.

## Acknowledgments
I extend my sincere thanks to [Ananddd](https://www.kaggle.com/ananddd) for providing the Gujarati OCR dataset used in this project. The dataset's comprehensiveness and diversity were essential to the success of this OCR model. You can access the dataset here: [Gujarati OCR Dataset](https://www.kaggle.com/datasets/ananddd/gujarati-ocr-typed-gujarati-characters/data).
