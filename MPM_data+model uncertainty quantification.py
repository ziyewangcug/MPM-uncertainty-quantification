import numpy as np
import scipy.io as sio
from osgeo import gdal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import math
from sklearn.model_selection import train_test_split
import os

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

# ========== Paths and Parameters ==========
BASE_PATH = "xx"
DATA_PATH = {
    "positive_tif": BASE_PATH + "deposit.tif",
    "combined_layers_mat": BASE_PATH + "evidence layers.mat",
}

# Model parameters
WINDOW_SIZE = 9
ALL_CHANNEL = 44
NUM_SIMULATIONS = 100
MC_SAMPLES = 100
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.00006

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
        print(f"Saved: {out_path}")


def read_tif(path):
    """Read a single-band TIFF file"""
    dataset = gdal.Open(path)
    Nodata = dataset.GetRasterBand(1).GetNoDataValue()
    geotransform = dataset.GetGeoTransform()
    nXSize, nYSize = dataset.RasterXSize, dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, nXSize, nYSize)
    # Generate coordinate indices
    index = [geotransform[3] + j * geotransform[5] for j in range(nYSize)]
    columns = [geotransform[0] + i * geotransform[1] for i in range(nXSize)]
    return im_data, Nodata, index, columns

def generate_label_array(positive_tif_path):
    """Generate positive and negative sample labels"""
    pos_data, _, _, _ = read_tif(positive_tif_path)
    label_auc = np.where(pos_data == 1, 1, 0)  # Binarize labels
    # Get positions of positive and negative samples
    pos_y, pos_x = np.where(pos_data == 1)
    non_pos_y, non_pos_x = np.where(pos_data != 1)
    # Balanced sampling of negative samples
    neg_idx = np.random.choice(len(non_pos_y), size=min(len(pos_y), len(non_pos_y)), replace=False)
    neg_y, neg_x = non_pos_y[neg_idx], non_pos_x[neg_idx]
    print(f"Positive samples: {len(pos_y)}, Negative samples: {len(neg_y)}")
    # Create label array
    label_array = np.full_like(pos_data, -1)
    label_array[pos_y, pos_x] = 1
    label_array[neg_y, neg_x] = 0
    return label_array, label_auc

def expand_label(label_array, argument=3):
    """Expand labels by neighborhood"""
    label_array_agu = label_array.copy()
    half_win = math.floor(argument / 2)
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

def MakeCNNTrainData(Feature2d, WindowSize):
    """Generate sliding window data for CNN training"""
    nYSize, nXSize, nBands = Feature2d.shape

    # Generate coordinate grid
    deposit_ds = gdal.Open(DATA_PATH["positive_tif"])
    geotransform = deposit_ds.GetGeoTransform()
    columns = [geotransform[0] + i * geotransform[1] for i in range(nXSize)]
    index = [geotransform[3] + j * geotransform[5] for j in range(nYSize)]

    # Create coordinate points
    HTPointX = np.tile(columns, len(index))
    HTPointY = np.repeat(index, len(columns))
    HTPointXY = np.array([HTPointX, HTPointY])

    # Normalize coordinates
    ColumGap, IndexGap = geotransform[1], abs(geotransform[5])
    PointXYIndex = np.array([(HTPointXY[0, :] - columns[0]) / ColumGap,
                             (HTPointXY[1, :] - index[0]) / (-IndexGap)])

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
                if (0 <= XIndex < Feature2d.shape[1]) and (0 <= YIndex < Feature2d.shape[0]):
                    row_data.append(Feature2d[YIndex, XIndex, :])
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

    return np.array(CNNTrainData), HTPointXY, columns, index

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


def generate_samples_for_simulation(evidence_data, simulation_idx):
    """Generate training samples for a specific simulation"""
    print(f"Start generating training samples for simulation {simulation_idx + 1}...")

    # Evidence data shape: (nYSize, nXSize, nBands)
    nYSize, nXSize, nBands = evidence_data.shape

    # Generate CNN training data
    CNNTrainData, HTPointXY, columns, index = MakeCNNTrainData(evidence_data, WINDOW_SIZE)

    # Generate labels and expand
    label_array, label_auc = generate_label_array(DATA_PATH["positive_tif"])
    label_array_agu = expand_label(label_array)

    # Get labeled CNN data and split
    pos_cnn, neg_cnn = get_labeled_cnn_data(CNNTrainData, label_array_agu, HTPointXY, columns, index)

    # If sample numbers are insufficient, use all available samples
    if len(pos_cnn) == 0 or len(neg_cnn) == 0:
        print("Warning: Positive or negative samples count is zero, using all available samples")
        all_data = CNNTrainData
        all_labels = np.array([1 if label_array_agu[lat2row[HTPointXY[1, i]], lon2col[HTPointXY[0, i]]] == 1 else 0
                               for i in range(len(CNNTrainData))])
        train_data, val_data, train_label, val_label = train_test_split(
            all_data, all_labels, train_size=0.8, random_state=42, stratify=all_labels)
    else:
        train_data, train_label, val_data, val_label = split_train_val(pos_cnn, neg_cnn)

    print(f"Sample generation for simulation {simulation_idx + 1} completed!")

    return train_data, train_label, val_data, val_label, CNNTrainData, label_auc


def split_train_val(pos_data, neg_data, train_ratio=0.8, random_state=42):
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
    print("Bayesian CNN model built")
    return model

def uncertainty_method_B():
    """
    Directly compute uncertainty from 10000 predictions (100 data simulations × 100 MC samples)
    """
    print("=" * 60)
    print("        Directly compute uncertainty from 10000 predictions")
    print("=" * 60)

    # 1. Check if input file exists
    combined_layers_file = DATA_PATH["combined_layers_mat"]
    if not os.path.exists(combined_layers_file):
        print(f"Error: Evidence layers file {combined_layers_file} does not exist")
        return None, None, None

    # 2. Load all simulation data
    print("Loading evidence layers data...")
    combined_layers_mat = sio.loadmat(combined_layers_file)
    combined_layers = combined_layers_mat['combined_layers'][0]  # 100 sets of simulated evidence layers

    print(f"Found {len(combined_layers)} simulated evidence layers")

    # 3. Build the Bayesian CNN model
    model = build_bayesian_cnn()

    # 4. Perform MC Dropout prediction for each data simulation
    print("Starting predictions...")
    all_predictions = []  # Store all predictions

    for sim_idx in range(min(NUM_SIMULATIONS, len(combined_layers))):
        print(f"\nProcessing data simulation {sim_idx + 1}/{min(NUM_SIMULATIONS, len(combined_layers))}")

        # Get evidence data for current simulation
        evidence_data = combined_layers[sim_idx]  # Shape: (H, W, C)

        # Generate prediction data for current simulation
        train_data, train_label, val_data, val_label, prediction_data, label_auc = generate_samples_for_simulation(
            evidence_data, sim_idx)

        # Train the model (for demonstration, we could use a pre-trained model or fast training)
        print(f"Training model {sim_idx + 1}...")
        model.fit(
            train_data, train_label,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val_data, val_label),
            verbose=0
        )

        # Perform MC Dropout predictions for this data simulation
        mc_predictions = []
        for mc_idx in range(MC_SAMPLES):
            if (mc_idx + 1) % 10 == 0:
                print(f"  Performing MC sampling {mc_idx + 1}/{MC_SAMPLES}...")
            pred = model(prediction_data, training=True)  # training=True enables Dropout
            mc_predictions.append(pred.numpy())

        # Reshape predictions to spatial format
        mc_predictions = np.stack(mc_predictions, axis=0)

        # Get spatial dimension information
        deposit_ds = gdal.Open(DATA_PATH["positive_tif"])
        nYSize, nXSize = deposit_ds.RasterYSize, deposit_ds.RasterXSize
        mc_predictions_spatial = mc_predictions.reshape(MC_SAMPLES, nYSize, nXSize)

        all_predictions.append(mc_predictions_spatial)

    # 5. Combine all predictions
    all_predictions = np.array(all_predictions)
    print(f"All predictions shape: {all_predictions.shape}")
    print(f"Total number of predictions: {all_predictions.size}")

    # 6. Directly compute total uncertainty
    print("Computing total uncertainty...")
    total_predictions = all_predictions.reshape(-1, all_predictions.shape[2], all_predictions.shape[3])
    total_variance = np.var(total_predictions, axis=0)

    # Compute mean prediction
    total_mean = np.mean(total_predictions, axis=0)

    # Compute entropy
    epsilon = 1e-9
    p_total = np.clip(total_mean, epsilon, 1 - epsilon)
    total_entropy = - (p_total * np.log(p_total) + (1 - p_total) * np.log(1 - p_total))

    # 7. Save results
    print("Saving result files...")
    ref_path = DATA_PATH["positive_tif"]

    save_geotiff(total_variance, ref_path, BASE_PATH + "total_variance.tif")
    save_geotiff(total_mean, ref_path, BASE_PATH + "total_mean.tif")
    save_geotiff(total_entropy, ref_path, BASE_PATH + "total_entropy.tif")

    # Save all predictions
    sio.savemat(BASE_PATH + "predictions.mat", {
        'all_predictions': all_predictions,
        'total_variance': total_variance,
        'total_mean': total_mean,
        'total_entropy': total_entropy,
    })

    return total_variance


if __name__ == "__main__":
    uncertainty_method_B()