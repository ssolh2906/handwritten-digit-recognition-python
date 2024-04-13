# Pyplot
import matplotlib.pyplot as plt

# datasets, classifiers and performance metrics.
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Dataset
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

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

_, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# Confusion Matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Cofusion Matrix:\n{disp.confusion_matrix}")

plt.show()
