from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from audio_processing import process_sound_files


if __name__ == "__main__":
    input_gun_dir = "./dataset/train/guns/"
    input_nongun_dir = "./dataset/train/non guns/"
    test_gun_dir = "./dataset/test/guns/"
    test_nongun_dir = "./dataset/test/non guns/"

    # Process training data
    gun_features, gun_labels = process_sound_files(input_gun_dir, label=1)
    nongun_features, nongun_labels = process_sound_files(input_nongun_dir, label=0)

    # Combine features and labels
    features_array = np.vstack((gun_features, nongun_features))
    labels = np.hstack((gun_labels, nongun_labels))

    assert len(features_array) == len(
        labels
    ), "Mismatch between features and labels length"

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel="rbf")  # You can also try 'rbf', 'poly', etc.
    svm_model.fit(X_train_scaled, y_train)

    # Evaluate on the training set (to get training accuracy)
    y_train_pred = svm_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on the test set (to get test accuracy)
    y_test_pred = svm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Test on new data (test set for gun and non-gun sounds)
    test_features_gun, test_labels_gun = process_sound_files(test_gun_dir, label=1)
    test_features_nongun, test_labels_nongun = process_sound_files(
        test_nongun_dir, label=0
    )

    test_features = np.vstack((test_features_gun, test_features_nongun))
    test_labels = np.hstack((test_labels_gun, test_labels_nongun))

    # Standardize the test data
    test_features_scaled = scaler.transform(test_features)

    # Evaluate on new test data
    y_test_pred_new = svm_model.predict(test_features_scaled)
    test_accuracy_new = accuracy_score(test_labels, y_test_pred_new)
    print(f"Test accuracy on new data: {test_accuracy_new:.4f}")  # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel="rbf")  # You can also try 'rbf', 'poly', etc.
    svm_model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Test on new data (test set for gun and non-gun sounds)
    test_features_gun, test_labels_gun = process_sound_files(test_gun_dir, label=1)
    test_features_nongun, test_labels_nongun = process_sound_files(
        test_nongun_dir, label=0
    )

    test_features = np.vstack((test_features_gun, test_features_nongun))
    test_labels = np.hstack((test_labels_gun, test_labels_nongun))

    test_features_scaled = scaler.transform(test_features)

    y_test_pred = svm_model.predict(test_features_scaled)
    test_accuracy = accuracy_score(test_labels, y_test_pred)
    print(f"Test accuracy on new data: {test_accuracy:.4f}")
