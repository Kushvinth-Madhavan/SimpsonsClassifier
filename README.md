
# SimpsonsClassifier

A PyTorch-based project for classifying characters from "The Simpsons" using a ResNet-18 model. This repository allows you to train a custom vision model on a dataset of images and predict the class of any Simpsons character in a test image.

## Features
- Dataset loader with custom transformations.
- Pretrained ResNet-18 fine-tuned for multi-class classification.
- Training and validation with detailed metrics (loss, accuracy).
- Single-image prediction for testing trained models.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- PIL (Pillow)
- CUDA or MPS support for GPU acceleration (optional)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Kushvinth-Madhavan/SimpsonsClassifier.git
   cd SimpsonsClassifier
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Go to the [Simpsons Characters Dataset on Kaggle](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset).
   - Download the dataset as a ZIP file.
   - Extract the dataset and place the `kaggle_simpson_testset` folder in the following directory:
     ```
     Simpsons_Characters_Data/kaggle_simpson_testset/
     ```

4. Verify the dataset structure:
   ```
   Simpsons_Characters_Data/
     kaggle_simpson_testset/
       character1_image1.jpg
       character2_image2.jpg
       ...
   ```

## Usage

### 1. Training
Run the script to train the model:
```bash
python model.py
```
This will load your dataset, split it into training and validation sets, and start training the ResNet-18 model. Training progress, accuracy, and loss are displayed during the process.

### 2. Predicting
After training, you can predict a single image's class:
```bash
Enter the path of the image to predict: /path/to/image.jpg
```
The model will output the predicted class.

## Customization
- **Dataset Directory**: Modify the `dataset_path` variable in `main()` to point to your dataset.
- **Hyperparameters**: Adjust batch size, learning rate, and other parameters in `train_model()` and `main()`.
- **Transforms**: Update the `train_transform` and `val_transform` for preprocessing customization.

## Example Output
### Training:
```
Epoch 1/10
----------
Training Loss: 0.5678, Accuracy: 85.23%
Validation Loss: 0.4532, Accuracy: 88.15%
...
```

### Prediction:
```
Predicted Class: Homer_Simpson
```

## Future Enhancements
- Add support for additional datasets.
- Integrate a web interface for easier predictions.
- Explore hyperparameter optimization.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.
