from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load digits dataset
digits = load_digits()

# Features and labels
X = digits.images.reshape((len(digits.images), -1))  # Flatten 8x8 images
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple SVM model
model = SVC(gamma=0.001)
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")