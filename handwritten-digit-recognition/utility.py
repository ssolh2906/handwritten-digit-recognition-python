from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def visualize_4_digits(images, labels, title="Title"):
    """Visualizes MNIST handwritten digits using subplots.

    Args:
   images (np.ndarray): A NumPy array containing the digit images.
            - Shape should be (n_images, 8, 8) where n_images is the number of images.
            - If the data is flattened (shape of (n_images,)), it will be reshaped to (n_images, 8, 8) internally.
        labels (np.ndarray): A NumPy array containing the digit labels (0-9). Should have the same length as `images`.
        title (str, optional): The title for the entire plot. Defaults to "Title".

    Returns:
        None
    """

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, images, labels):
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title}: {label}")

    plt.show()


def plot_confusion_matrix(y_test, predicted):
    """
    Plots a confusion matrix using scikit-learn's ConfusionMatrixDisplay.

    Args:
        y_test (np.ndarray): Array of true labels.
        predicted (np.ndarray): Array of predicted labels.

    Returns:
        Confusion Matrix Display.
    """

    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")

    plt.show()  # Display the plot
    return disp
