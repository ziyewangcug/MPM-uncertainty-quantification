import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import numpy as np
from sklearn.metrics import roc_curve, auc
import math
from osgeo import gdal
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========== INPUT ==========
# input data: deposit.tif [X Y]-known locations of deposits (label: 1) and negative samples (label: 0). evidence.tif [X Y n]-n geological, geochemical, and geophysical evidence layers
BASE_PATH = "xx"
DATA_PATH = {
    "positive_tif": BASE_PATH + "deposit.tif",
    "evidence_tif": BASE_PATH + "evidence_norm.tif",
    "train_data": BASE_PATH + "train_data.npy",
    "train_label": BASE_PATH + "train_label.npy",
    "val_data": BASE_PATH + "val_data.npy",
    "val_label": BASE_PATH + "val_label.npy",
    "prediction_data": BASE_PATH + "prediction_data.npy",
    "label": BASE_PATH + "label.npy"
}
OUTPUT_training_PATH = BASE_PATH + "loss_acc.csv"
output_prediction_path = BASE_PATH + "prediction_output.csv"
out_tif_path = BASE_PATH + "prediction_output.tif"

# parameters
WINDOW_SIZE = 9
ALL_CHANNEL = 44
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.00006
ARGUMENT = 3
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# ========== Data reading ==========
def ReadMutiTif(path):
    dataset = gdal.Open(path)
    nXSize, nYSize = dataset.RasterXSize, dataset.RasterYSize
    YG_geotrans, YG_proj = dataset.GetGeoTransform(), dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, nXSize, nYSize)
    Nodata = [dataset.GetRasterBand(i + 1).GetNoDataValue() for i in range(im_data.shape[0])]
    index = [YG_geotrans[3] + j * YG_geotrans[5] for j in range(nYSize)]
    columns = [YG_geotrans[0] + i * YG_geotrans[1] for i in range(nXSize)]
    return im_data, Nodata, index, columns

def read_tif(path):
    dataset = gdal.Open(path)
    Nodata = dataset.GetRasterBand(1).GetNoDataValue()
    geotransform = dataset.GetGeoTransform()
    nXSize, nYSize = dataset.RasterXSize, dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, nXSize, nYSize)
    index = [geotransform[3] + j * geotransform[5] for j in range(nYSize)]
    columns = [geotransform[0] + i * geotransform[1] for i in range(nXSize)]
    return im_data, Nodata, index, columns

# ========== Data Preprocessing Functions ==========
def MakeCNNTrainData(PointXY, Feature2d, WindowSize):
    """Generate sliding window data for CNN training"""
    # Calculate grid spacing
    XDistance, YDistance = abs(PointXY[0, :] - PointXY[0, 0]), abs(PointXY[1, :] - PointXY[1, 0])
    ColumGap, IndexGap = np.min(XDistance[XDistance != 0]), np.min(YDistance[YDistance != 0])
    # Create grid coordinates
    Column = np.arange(PointXY[0, :].min(), PointXY[0, :].max() + ColumGap, ColumGap)
    Index = np.arange(PointXY[1, :].max(), PointXY[1, :].min() - IndexGap, -IndexGap)
    # Normalize coordinates
    PointXYIndex = np.array([(PointXY[0, :] - Column[0]) / ColumGap,
                             (PointXY[1, :] - Index[0]) / (-IndexGap)])
    # Generate sliding window data
    half_win = math.floor(WindowSize / 2)
    CNNPaddingData = []
    NoDataPadding = np.full(Feature2d.shape[-1], -99)

    for i in range(PointXYIndex.shape[1]):
        WindowData = []
        for j in range(-half_win, half_win + 1):
            row_data = []
            for k in range(-half_win, half_win + 1):
                XIndex, YIndex = int(PointXYIndex[0, i] + k), int(PointXYIndex[1, i] + j)
                # Boundary check, fill with no data
                if (0 <= XIndex < Feature2d.shape[0]) and (0 <= YIndex < Feature2d.shape[1]):
                    row_data.append(Feature2d[XIndex, YIndex, :])
                else:
                    row_data.append(NoDataPadding)
            WindowData.append(row_data)
        CNNPaddingData.append(WindowData)

    # Fill no data values
    CNNTrainData = []
    for window in CNNPaddingData:
        chan_data = []
        for ch in range(len(window[0][0])):
            channel_window = np.array(window)[:, :, ch]
            mask = channel_window != -99
            if np.any(mask):
                channel_window[~mask] = np.mean(channel_window[mask])
            chan_data.append(channel_window)
        CNNTrainData.append(chan_data)

    return np.array(CNNTrainData)

def generate_label_array(positive_tif_path):
    """Generate positive and negative sample labels"""
    pos_data, _, _, _ = read_tif(positive_tif_path)
    label_auc = np.where(pos_data == 1, 1, 0)  # Binarize labels
    # Get positions of positive and negative samples
    pos_y, pos_x = np.where(pos_data == 1)
    non_pos_y, non_pos_x = np.where(pos_data != 1)
    # Balanced sampling of negative samples
    neg_idx = np.random.choice(len(non_pos_y), size=len(pos_y), replace=False)
    neg_y, neg_x = non_pos_y[neg_idx], non_pos_x[neg_idx]
    print(f"Positive samples: {len(pos_y)}, Negative samples: {len(neg_y)}")
    # Create label array
    label_array = np.full_like(pos_data, -1)
    label_array[pos_y, pos_x] = 1
    label_array[neg_y, neg_x] = 0
    return label_array, label_auc

def expand_label(label_array):
    """Expand labels by neighborhood"""
    label_array_agu = label_array.copy()
    half_win = math.floor(ARGUMENT / 2)
    rows, cols = label_array.shape
    # Expand positive samples
    pos_y, pos_x = np.where(label_array == 1)
    for y, x in zip(pos_y, pos_x):
        for dy in range(-half_win, half_win + 1):
            for dx in range(-half_win, half_win + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    label_array_agu[ny, nx] = 1

    # Expand negative samples (without overwriting positive samples)
    neg_y, neg_x = np.where(label_array == 0)
    for y, x in zip(neg_y, neg_x):
        for dy in range(-half_win, half_win + 1):
            for dx in range(-half_win, half_win + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and label_array_agu[ny, nx] != 1:
                    label_array_agu[ny, nx] = 0

    print(f"After expansion - Positive: {np.sum(label_array_agu == 1)}, Negative: {np.sum(label_array_agu == 0)}")
    return label_array_agu

def get_labeled_cnn_data(CNNTrainData, label_array_agu, HTPointXY, pos_column, pos_index):
    """Get CNN training data based on labels"""
    lon2col = {lon: idx for idx, lon in enumerate(pos_column)}
    lat2row = {lat: idx for idx, lat in enumerate(pos_index)}

    pos_cnn_data, neg_cnn_data = [], []

    for i in range(HTPointXY.shape[1]):
        col_idx = lon2col[HTPointXY[0, i]]
        row_idx = lat2row[HTPointXY[1, i]]

        label = label_array_agu[row_idx, col_idx]
        if label == 1:
            pos_cnn_data.append(CNNTrainData[i])
        elif label == 0:
            neg_cnn_data.append(CNNTrainData[i])

    return np.array(pos_cnn_data), np.array(neg_cnn_data)

def split_train_val(pos_data, neg_data, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE):
    """Split training and validation sets"""
    pos_train, pos_val = train_test_split(pos_data, train_size=train_ratio, random_state=random_state)
    neg_train, neg_val = train_test_split(neg_data, train_size=train_ratio, random_state=random_state)
    # Combine data and shuffle
    train_data = np.concatenate([pos_train, neg_train])
    train_label = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    val_data = np.concatenate([pos_val, neg_val])
    val_label = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))])

    shuffle_idx = np.random.permutation(len(train_data))
    train_data, train_label = train_data[shuffle_idx], train_label[shuffle_idx]

    print(f"Training set: {train_data.shape} (Pos:{np.sum(train_label == 1)}, Neg:{np.sum(train_label == 0)})")
    print(f"Validation set: {val_data.shape} (Pos:{np.sum(val_label == 1)}, Neg:{np.sum(val_label == 0)})")

    return train_data, train_label, val_data, val_label

def generate_samples():
    """Main function to generate training samples"""
    print("Start generating training samples...")
    # Read evidence data
    data, _, Index, Column = ReadMutiTif(DATA_PATH["evidence_tif"])

    # Flatten data and get coordinates
    HTPointFeature = data.reshape(data.shape[0], -1).T
    HTPointX = np.tile(Column, len(Index))
    HTPointY = np.repeat(Index, len(Column))
    HTPointXY = np.array([HTPointX, HTPointY])

    # Create background grid
    BackGround = np.full((len(Column), len(Index), HTPointFeature.shape[-1]), -99)
    for i, (x, y) in enumerate(zip(HTPointX, HTPointY)):
        BackGround[np.where(Column == x)[0], np.where(Index == y)[0]] = HTPointFeature[i]

    # Generate CNN training data
    CNNTrainData = MakeCNNTrainData(HTPointXY, BackGround, WINDOW_SIZE)

    # Generate labels and expand
    label_array, label_auc = generate_label_array(DATA_PATH["positive_tif"])
    label_array_agu = expand_label(label_array)

    # Get labeled CNN data and split
    pos_cnn, neg_cnn = get_labeled_cnn_data(CNNTrainData, label_array_agu, HTPointXY, Column, Index)
    train_data, train_label, val_data, val_label = split_train_val(pos_cnn, neg_cnn)

    # Save data
    np.save(DATA_PATH["train_data"], train_data)
    np.save(DATA_PATH["train_label"], train_label)
    np.save(DATA_PATH["val_data"], val_data)
    np.save(DATA_PATH["val_label"], val_label)
    np.save(DATA_PATH["prediction_data"], CNNTrainData)
    np.save(DATA_PATH["label"], label_auc.reshape(-1, 1))

    print("Sample generation completed!")

# ========== Model Building and Training ==========
def build_model():
    """Build CNN model"""
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', activation='relu',
               input_shape=(ALL_CHANNEL, WINDOW_SIZE, WINDOW_SIZE)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
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

def train_and_predict():
    """Train model and make predictions"""
    print("Start training model...")

    # Load data
    train_data = np.load(DATA_PATH["train_data"])
    train_labels = np.load(DATA_PATH["train_label"])
    valid_data = np.load(DATA_PATH["val_data"])
    valid_labels = np.load(DATA_PATH["val_label"])

    # Build and train model
    model = build_model()
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

    # Predict
    print("Start predicting...")
    all_array = np.load(DATA_PATH["prediction_data"])
    output = model.predict(all_array)

    # Reshape prediction results
    deposit = gdal.Open(DATA_PATH["positive_tif"])
    output_matrix = output.reshape(deposit.RasterYSize, deposit.RasterXSize)

    # Save prediction results
    pd.DataFrame(output_matrix).to_csv(output_prediction_path, index=False, header=False)
    save_geotiff(output_matrix, DATA_PATH["positive_tif"], out_tif_path)

    # Visualize prediction results
    plt.figure(figsize=(8, 6))
    plt.imshow(output_matrix, cmap='hot')
    plt.colorbar()
    plt.title('Prediction Results')
    plt.show()

    # Model evaluation (ROC curve)
    label = np.load(DATA_PATH["label"]).flatten()
    fpr, tpr, _ = roc_curve(label, output)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'), plt.legend(loc='best')
    plt.show()

    print(f"Model training completed! AUC: {roc_auc:.3f}")

# ========== Main Program ==========
if __name__ == "__main__":
    generate_samples()  # Generate sample data
    train_and_predict()  # Train model and make predictions