# Pyplot
import matplotlib.pyplot as plt

# datasets, classifiers and performance metrics.
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

import utility

# Dataset
digits = datasets.load_digits()


def predict_digits_sl(digits):
    # Classification
    # Flatten the images, image: (8,8) -> (64,)
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split 50% train / 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    # Visualize 4 test samples and results.
    utility.visualize_4_digits(images=X_test, labels=predicted, title="Prediction")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    utility.plot_confusion_matrix(y_test, predicted)


predict_digits_sl(digits)
