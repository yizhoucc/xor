"""Tabular/text/audio/timeseries datasets for classification experiments."""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_adult(data_path, seed=42):
    """Load UCI Adult Income dataset. Binary classification: >50K vs <=50K.

    Returns (train_dataset, test_dataset, input_dim, num_classes).
    Requires: scikit-learn, pandas
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    train_path = os.path.join(data_path, 'adult', 'adult.data')
    test_path = os.path.join(data_path, 'adult', 'adult.test')

    # Download if not exists
    if not os.path.exists(train_path):
        os.makedirs(os.path.join(data_path, 'adult'), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            train_path)
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
            test_path)

    df_train = pd.read_csv(train_path, names=columns, sep=r',\s*', engine='python', na_values='?')
    df_test = pd.read_csv(test_path, names=columns, sep=r',\s*', engine='python',
                          na_values='?', skiprows=1)

    # Clean labels
    df_train['income'] = df_train['income'].str.strip().str.rstrip('.')
    df_test['income'] = df_test['income'].str.strip().str.rstrip('.')

    # Drop missing
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Encode
    cat_cols = df_train.select_dtypes(include='object').columns.drop('income')
    df_all = pd.concat([df_train, df_test])

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df_all[col])
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    X_train = df_train.drop('income', axis=1).values.astype(np.float32)
    y_train = (df_train['income'] == '>50K').values.astype(np.int64)
    X_test = df_test.drop('income', axis=1).values.astype(np.float32)
    y_test = (df_test['income'] == '>50K').values.astype(np.int64)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, X_train.shape[1], 2


def load_wine(data_path, seed=42):
    """Load UCI Wine Quality dataset. Multi-class classification (quality 3-9).

    Returns (train_dataset, test_dataset, input_dim, num_classes).
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    wine_path = os.path.join(data_path, 'wine', 'winequality-red.csv')

    if not os.path.exists(wine_path):
        os.makedirs(os.path.join(data_path, 'wine'), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            wine_path)

    df = pd.read_csv(wine_path, sep=';')
    X = df.drop('quality', axis=1).values.astype(np.float32)
    y = df['quality'].values

    # Remap quality to 0-indexed classes
    le = LabelEncoder()
    y = le.fit_transform(y).astype(np.int64)
    num_classes = len(le.classes_)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, X.shape[1], num_classes


def load_sst2(data_path, seed=42):
    """Load SST-2 sentiment classification. Binary: positive/negative.

    Uses bag-of-words representation for MLP compatibility.
    Returns (train_dataset, test_dataset, input_dim, num_classes).
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer

    cache_dir = os.path.join(data_path, 'sst2')
    ds = load_dataset('glue', 'sst2', cache_dir=cache_dir)

    texts_train = ds['train']['sentence']
    labels_train = np.array(ds['train']['label'], dtype=np.int64)
    texts_val = ds['validation']['sentence']
    labels_val = np.array(ds['validation']['label'], dtype=np.int64)

    # TF-IDF features (max 5000 dims for tractability)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(texts_train).toarray().astype(np.float32)
    X_val = vectorizer.transform(texts_val).toarray().astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(labels_train))
    test_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(labels_val))

    return train_dataset, test_dataset, X_train.shape[1], 2


def load_agnews(data_path, seed=42):
    """Load AG News topic classification. 4 classes: World, Sports, Business, Sci/Tech.

    Uses bag-of-words representation for MLP compatibility.
    Returns (train_dataset, test_dataset, input_dim, num_classes).
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer

    cache_dir = os.path.join(data_path, 'agnews')
    ds = load_dataset('ag_news', cache_dir=cache_dir)

    texts_train = ds['train']['text']
    labels_train = np.array(ds['train']['label'], dtype=np.int64)
    texts_test = ds['test']['text']
    labels_test = np.array(ds['test']['label'], dtype=np.int64)

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(texts_train).toarray().astype(np.float32)
    X_test = vectorizer.transform(texts_test).toarray().astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(labels_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(labels_test))

    return train_dataset, test_dataset, X_train.shape[1], 4


def load_speech_commands(data_path, seed=42, max_features=1000):
    """Load Speech Commands v2 (35 keywords). Mel-spectrogram → flattened for MLP.

    Returns (train_dataset, test_dataset, input_dim, num_classes).
    """
    import torchaudio

    root = os.path.join(data_path, 'speech_commands')
    os.makedirs(root, exist_ok=True)
    train_ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset='training')
    test_ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset='testing')

    # Build label map
    labels_set = sorted(set(s[2] for s in train_ds))
    label2idx = {l: i for i, l in enumerate(labels_set)}

    def process_dataset(ds, n_mels=40, target_len=32):
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels, n_fft=400, hop_length=160)
        features, labels = [], []
        for waveform, sr, label, *_ in ds:
            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            mel = mel_transform(waveform)  # (1, n_mels, time)
            # Pad/truncate to fixed length
            if mel.size(-1) < target_len:
                mel = torch.nn.functional.pad(mel, (0, target_len - mel.size(-1)))
            else:
                mel = mel[..., :target_len]
            features.append(mel.squeeze(0).flatten())
            labels.append(label2idx[label])
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = process_dataset(train_ds)
    X_test, y_test = process_dataset(test_ds)

    # Standardize
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset, X_train.shape[1], len(labels_set)


def load_ecg(data_path, seed=42):
    """Load ECG200 from UCR Time Series Archive. Binary classification.

    Returns (train_dataset, test_dataset, input_dim, num_classes).
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    ucr_dir = os.path.join(data_path, 'ucr')

    # Try multiple possible file locations and extensions
    for subdir in ['ECG200', '']:
        for ext in ['.tsv', '.txt']:
            tp = os.path.join(ucr_dir, subdir, f'ECG200_TRAIN{ext}')
            if os.path.exists(tp):
                train_path = tp
                test_path = os.path.join(ucr_dir, subdir, f'ECG200_TEST{ext}')
                break
        else:
            continue
        break
    else:
        # Download
        os.makedirs(ucr_dir, exist_ok=True)
        import urllib.request, zipfile, io
        url = 'https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip'
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(ucr_dir)
        # Find the extracted files
        for ext in ['.tsv', '.txt']:
            tp = os.path.join(ucr_dir, f'ECG200_TRAIN{ext}')
            if os.path.exists(tp):
                train_path = tp
                test_path = os.path.join(ucr_dir, f'ECG200_TEST{ext}')
                break

    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    X_train = train_data[:, 1:].astype(np.float32)
    y_train = train_data[:, 0].astype(np.int64)
    X_test = test_data[:, 1:].astype(np.float32)
    y_test = test_data[:, 0].astype(np.int64)

    # Remap labels to 0-indexed
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, X_train.shape[1], len(le.classes_)
