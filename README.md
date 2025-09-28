# cnn_animals_image_classification

A self-defined CNN model to classify 10 species of animals. This project provides a user-friendly web interface to classify animal images using a Convolutional Neural Network (CNN) model.

## Features and Functionality

*   **Image Upload:** Users can upload animal images from their local machine.
*   **Sample Gallery:** Users can select images from a pre-defined gallery of 10 animal species.  The sample images are located within the `cnn-image-classifier/public` directory and are named like `butterfly-orange-monarch.jpg`, `cat-tabby-sitting.jpg`, etc.
*   **AI-Powered Classification:** Employs a CNN model to predict the species of the animal in the image.
*   **Classification Results:** Displays the predicted animal class, confidence score, and processing time.
*   **Top Predictions:** Shows the top 3 predicted classes with confidence scores.
*   **User Interface:** Provides a clean and intuitive user interface built with Next.js and Tailwind CSS.
*   **Progress Indicator:** Shows a progress bar during image analysis.
*   **Error Handling:** Displays informative error messages when prediction fails (e.g., when the FastAPI server is not running).
*   **Dark/Light Theme Support:** The application uses Next Themes to offer dark and light theme support.

## Technology Stack

*   **Frontend:**
    *   [Next.js](https://nextjs.org/) - React framework for building the user interface.
    *   [React](https://reactjs.org/) - JavaScript library for building user interfaces.
    *   [Tailwind CSS](https://tailwindcss.com/) - CSS framework for styling the application.
    *   [Radix UI](https://www.radix-ui.com/) - Unstyled, accessible components for React.
    *   [Lucide React](https://lucide.dev/) - Icon library.
    *   [Vercel Analytics](https://vercel.com/analytics) - Web analytics.
    *   [Next Themes](https://github.com/pacocoursey/next-themes) - Theme switching for Next.js.
*   **Backend:**
    *   [FastAPI](https://fastapi.tiangolo.com/) - Python framework for building the API.
    *   [PyTorch](https://pytorch.org/) - Deep learning framework for training and running the CNN model.
    *   [OpenCV (cv2)](https://opencv.org/) - Library for image processing.
    *   [Torchvision](https://pytorch.org/vision/stable/index.html) - Library for computer vision tasks.
*   **Other:**
    *   [NumPy](https://numpy.org/) - Library for numerical computing.
    *   [Matplotlib](https://matplotlib.org/) - Library for visualize data.

## Prerequisites

Before running the application, ensure you have the following installed:

*   [Node.js](https://nodejs.org/) (version 18 or higher)
*   [Python](https://www.python.org/) (version 3.7 or higher)
*   [pip](https://pypi.org/project/pip/) - Python package installer

## Installation Instructions

Follow these steps to set up the project:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vanhdev-web/cnn_animals_image_classification
    cd cnn_animals_image_classification
    ```

2.  **Install frontend dependencies:**

    ```bash
    cd cnn-image-classifier
    npm install
    ```

3.  **Install backend dependencies:**

    ```bash
    pip install fastapi uvicorn torch torchvision opencv-python scikit-learn pillow
    ```

4.  **Download the Animal Dataset:**

    Download the dataset to the root directory. The expected directory structure is `Animals/data/train` and `Animals/data/test` with each containing subdirectories for each animal category (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel). You may need to create these directories if they don't exist.  Example: `Animals/data/train/cat/image1.jpg`

5.  **Download the pre-trained model (optional):**

    Download the `best_cnn.pt` file into the `Animals/trained_model` directory.  If you skip this step, the application will use a randomly initialized model, and the predictions will be inaccurate until you train the model yourself.

## Usage Guide

1.  **Start the FastAPI backend:**

    ```bash
    cd Animals
    python main.py
    ```

    This will start the server at `http://localhost:8000`.

2.  **Start the Next.js frontend:**

    ```bash
    cd cnn-image-classifier
    npm run dev
    ```

    This will start the development server at `http://localhost:3000`.

3.  **Open the application in your browser:**

    Navigate to `http://localhost:3000` to use the application.

4.  **Classify an image:**
    *   **Upload Image:** Click the "Upload" tab and either drag and drop an image or click to select a file from your computer.
    *   **Select Sample:** Click the "Sample Animals" tab and choose an image from the gallery.
    *   Click the "Classify Animal" button to see the prediction.

## API Documentation

The FastAPI backend provides a single endpoint:

*   **`POST /predict`**:  Accepts an image file as input and returns a JSON response containing the predicted animal class, confidence score, and processing time.

    **Request:**

    ```
    POST /predict
    Content-Type: multipart/form-data

    file: (image file)
    ```

    **Response:**

    ```json
    {
      "class": "cat",
      "confidence": 0.98,
      "processing_time": 0.123
    }
    ```

## Contributing Guidelines

Contributions are welcome! To contribute to this project:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes and write tests.
4.  Submit a pull request with a clear description of your changes.

## Training the CNN Model

1.  **Navigate to the `Animals` directory:**

    ```bash
    cd Animals
    ```

2.  **Run the training script:**

    ```bash
    python train_cnn.py --root . --epoch 10
    ```

    *   `--root`: Specifies the root directory of the animal dataset. Defaults to `.`.
    *   `--epoch`:  Specifies the number of training epochs. Defaults to 10.
    *   `--batch_size`: Specifies the batch size for training. Defaults to 8.
    *   `--image_size`: Specifies the image size to which all images will be resized during training. Defaults to 224.
    *   `--logging`: Specifies the directory to save tensorboard logs. Defaults to `tensorboard`.
    *   `--trained_model`: Specifies the directory to save the trained models. Defaults to `trained_model`.
    *   `--checkpoint`: Specifies the path to a checkpoint file to resume training from. Defaults to `None`.

    **Note:** Training the model requires a significant amount of computational resources and may take a long time depending on your hardware and dataset size.

## License Information

No license specified. All rights reserved.

## Contact/Support Information

For questions or support, please contact the repository owner through GitHub.