import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import numpy as np
from sklearn.metrics import roc_curve, auc
from osgeo import gdal
import scipy.io as sio
from tensorflow.python.training import momentum

# ========== GPU Configuration ==========
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========== Paths ==========
BASE_PATH = "xx"
DATA_PATH = {
    "positive_tif": BASE_PATH + "deposit.tif",
    "evidence_tif": BASE_PATH + "evidence.tif",
    "train_data": BASE_PATH + "train_data.npy",
    "train_label": BASE_PATH + "train_label.npy",
    "val_data": BASE_PATH + "val_data.npy",
    "val_label": BASE_PATH + "val_label.npy",
    "prediction_data": BASE_PATH + "prediction_data.npy",
    "label": BASE_PATH + "label.npy"
}
OUTPUT_training_PATH = BASE_PATH + "loss_acc.csv"
output_prediction_path = BASE_PATH + "prediction_mean.csv"
out_tif_path = BASE_PATH + "model uncertainty_mean.tif"
out_mat_path = BASE_PATH + "model uncertainty.mat"

# Model parameters
WINDOW_SIZE = 9
ALL_CHANNEL = 44
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.00006
MC_SAMPLES = 100


# ========== Model Building and Training ==========
def build_bayesian_cnn():
    """Build a Bayesian CNN model with Dropout"""
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', activation='relu',
               input_shape=(ALL_CHANNEL, WINDOW_SIZE, WINDOW_SIZE)),
        BatchNormalization(momentum=0.9, epsilon=1e-3),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.9, epsilon=1e-3),
        Dropout(0.1),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.9, epsilon=1e-3),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    return model


def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['acc'], label='Training')
    ax1.plot(history.history['val_acc'], label='Validation')
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Accuracy'), ax1.legend()
    ax1.set_title('Accuracy')

    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Loss'), ax2.legend()
    ax2.set_title('Loss')

    plt.tight_layout()
    plt.show()


def save_geotiff(data, ref_path, out_path):
    """Save GeoTIFF file"""
    ref_ds = gdal.Open(ref_path)
    if ref_ds:
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)

        out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
        out_ds.SetProjection(ref_ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.GetRasterBand(1).ComputeStatistics(False)
        out_ds = None
        ref_ds = None


def monte_carlo_predict(model, data, n_samples=MC_SAMPLES):
    """Perform Monte Carlo Dropout prediction"""
    print(f"Starting Monte Carlo Dropout sampling, {n_samples} times...")

    # List to store all prediction results
    predictions = []

    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            print(f"Performing sampling {i + 1}/{n_samples}...")

        # Use training=True to enable dropout during prediction
        pred = model(data, training=True)
        predictions.append(pred.numpy())

    # Stack all predictions into an array
    mc_predictions = np.stack(predictions, axis=-1)
    print(f"Monte Carlo prediction completed, result shape: {mc_predictions.shape}")

    return mc_predictions


def plot_mean_prediction(mean_pred, output_matrix_shape):
    """Plot the mean prediction result"""
    plt.figure(figsize=(8, 6))

    # Reshape to a 2D matrix
    mean_matrix = mean_pred.reshape(output_matrix_shape)

    # Plot mean prediction
    plt.imshow(mean_matrix, cmap='hot')
    plt.colorbar()
    plt.title('Mean Prediction')
    plt.show()

    return mean_matrix


def train_and_predict():
    """Train the model and perform Bayesian prediction"""
    print("Start training Bayesian CNN model...")

    # Load data
    train_data = np.load(DATA_PATH["train_data"])
    train_labels = np.load(DATA_PATH["train_label"])
    valid_data = np.load(DATA_PATH["val_data"])
    valid_labels = np.load(DATA_PATH["val_label"])

    # Build and train the Bayesian CNN model
    model = build_bayesian_cnn()
    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(valid_data, valid_labels),
        verbose=2
    )

    # Plot training process
    plot_training_history(history)

    # Save training records
    pd.DataFrame({
        'train_loss': history.history['loss'],
        'test_loss': history.history['val_loss'],
        'train_acc': history.history['acc'],
        'test_acc': history.history['val_acc']
    }).to_csv(OUTPUT_training_PATH, index=False)

    # Monte Carlo Dropout prediction
    print("Starting Monte Carlo Dropout prediction...")
    all_array = np.load(DATA_PATH["prediction_data"])

    # Perform MC sampling
    mc_predictions = monte_carlo_predict(model, all_array, n_samples=MC_SAMPLES)

    # Calculate mean prediction
    mean_prediction = np.mean(mc_predictions, axis=-1)

    # Reshape prediction results to spatial format
    deposit = gdal.Open(DATA_PATH["positive_tif"])
    output_matrix_shape = (deposit.RasterYSize, deposit.RasterXSize)

    # Plot only the mean prediction
    mean_matrix = plot_mean_prediction(mean_prediction, output_matrix_shape)

    # Reshape MC predictions to X×Y×n format
    mc_predictions_spatial = mc_predictions.reshape(output_matrix_shape[0], output_matrix_shape[1], MC_SAMPLES)
    print(f"Reshaped MC predictions shape: {mc_predictions_spatial.shape}")

    # Save MC predictions as a mat file
    mc_data = {
        'mc_predictions': mc_predictions_spatial  # X×Y×n
    }

    # Compute variance and entropy
    variance_matrix = np.var(mc_predictions_spatial, axis=2)

    # Compute entropy (information entropy)
    epsilon = 1e-8  # Avoid log(0)
    entropy_matrix = -np.mean(
        mc_predictions_spatial * np.log(mc_predictions_spatial + epsilon) +
        (1 - mc_predictions_spatial) * np.log(1 - mc_predictions_spatial + epsilon),
        axis=2
    )

    # Save mean, variance, and entropy as GeoTIFF
    save_geotiff(mean_matrix, DATA_PATH["positive_tif"], out_tif_path)
    print(f"Mean prediction result saved to: {out_tif_path}")

    save_geotiff(variance_matrix, DATA_PATH["positive_tif"], BASE_PATH + "model uncertainty_variance.tif")
    print(f"Variance prediction result saved to: {BASE_PATH + 'model uncertainty_variance.tif'}")

    save_geotiff(entropy_matrix, DATA_PATH["positive_tif"], BASE_PATH + "model uncertainty_entropy.tif")
    print(f"Entropy prediction result saved to: {BASE_PATH + 'model uncertainty_entropy.tif'}")

    # Model evaluation (ROC curve) - using mean prediction
    roc_auc = np.nan  # Initialize as NaN
    if os.path.exists(DATA_PATH["label"]):
        label = np.load(DATA_PATH["label"]).flatten()
        mean_pred_flat = mean_prediction.flatten()
        fpr, tpr, _ = roc_curve(label, mean_pred_flat)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Mean Prediction)'), plt.legend(loc='best')
        plt.show()

        print(f"Model training completed! AUC: {roc_auc:.3f}")
    else:
        print("Label data not found, skipping ROC curve calculation")

    # Add AUC value to the mat file data and resave
    mc_data['auc'] = roc_auc
    sio.savemat(out_mat_path, mc_data)
    print(f"Monte Carlo predictions and AUC value saved to: {out_mat_path}")


# ========== Main Program ==========
if __name__ == "__main__":
    train_and_predict()