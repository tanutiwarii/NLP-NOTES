# NLP Notes

## Overview

This repository serves as a comprehensive collection of Jupyter notebooks designed to teach and demonstrate fundamental concepts in Natural Language Processing (NLP) and Deep Learning. Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. The goal is to read, decipher, understand, and make sense of human language in a manner that is valuable.

These notebooks are structured to provide hands-on experience with NLP techniques, starting from basic concepts and progressing to advanced topics including deep learning with recurrent neural networks, gated recurrent units, and sequence-to-sequence models. Each notebook includes theoretical explanations, code implementations, and practical examples.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [Prerequisites](#prerequisites)
- [Dependencies and Libraries](#dependencies-and-libraries)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.8 or higher
- **RAM**: 4 GB minimum (8 GB recommended for deep learning)
- **Disk Space**: 1 GB for installation + dependencies
- **Internet Connection**: Required for NLTK data and model downloads

### Recommended Setup
- **Operating System**: macOS or Linux (better package compatibility)
- **Python Version**: 3.10 or higher
- **RAM**: 16 GB or more (for training neural networks)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **Python Environment Manager**: Conda or venv
- **Terminal/Command Prompt**: bash, zsh, or PowerShell

### Version Compatibility

| Package | Version | Python Compatibility | Last Updated |
|---------|---------|---------------------|--------------|
| jupyter | 1.0.0 | 3.8+ | Active |
| jupyterlab | 4.0.5 | 3.8+ | Active |
| nltk | 3.8.1 | 3.6+ | Stable |
| gensim | 4.3.2 | 3.8+ | Active |
| numpy | 1.24.3 | 3.9+ | Stable |
| pandas | 2.0.3 | 3.9+ | Active |
| matplotlib | 3.7.2 | 3.9+ | Active |
| seaborn | 0.12.2 | 3.8+ | Stable |
| scikit-learn | 1.3.0 | 3.9+ | Active |
| tensorflow | 2.13.0 | 3.9+ | Active |
| tensorflow-hub | 0.13.0 | 3.9+ | Active |
| torch | 1.10.0+ | 3.8+ | Active |

## Notebooks

### 1. Basic NLP Concepts (`1-Basic_NLP.ipynb`)

This introductory notebook covers the foundational concepts of Natural Language Processing. Topics include:

- Understanding text data and its challenges
- Tokenization: breaking text into words, sentences, or other units
- Basic text preprocessing techniques
- Introduction to NLP libraries and tools
- Simple text analysis and statistics

### 2. Text Normalization (`2-Text_Normalization.ipynb`)

Text normalization is crucial for preparing raw text data for analysis. This notebook explores various preprocessing steps including:

- Case conversion (lowercasing)
- Removing numbers and punctuation
- Handling whitespace and special characters
- Stop word removal using NLTK
- Stemming and lemmatization concepts
- Regular expressions for text cleaning

### 3. Embedding Techniques (`3-Embedding_Techniques.ipynb`)

Word embeddings are dense vector representations of words that capture semantic meaning. This notebook delves into:

- One-hot encoding for categorical data
- Bag-of-Words (BoW) model
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Word2Vec and GloVe embeddings
- Implementation of embedding layers in neural networks
- Visualization of word embeddings
- Transfer learning with pre-trained embeddings

### 4. Recurrent Neural Networks (`4-RNN.ipynb`)

This notebook introduces Recurrent Neural Networks (RNNs) for sequential data processing. Key topics include:

- Understanding RNN architecture and memory retention
- Character-level text generation using RNNs
- Sequence prediction and temporal dependencies
- Building RNN models with TensorFlow/Keras
- Training RNNs on sequential data
- Applications in NLP and time series forecasting
- Vanishing/exploding gradient problems

### 5. Gated Recurrent Units (`5-GRU.ipynb`)

This advanced notebook explores two major deep learning architectures:

#### Part A: GRU (Gated Recurrent Units)
- GRU architecture: Update and Reset gates
- Advantages of GRUs over basic RNNs (gradient flow, memory)
- Time series forecasting with GRUs
- Temperature prediction using historical data
- Building GRU models with TensorFlow/Keras
- Data preprocessing for sequential models
- Model training and evaluation metrics
- Data scaling with MinMaxScaler

#### Part B: Seq2Seq Models (Sequence-to-Sequence)
- Sequence-to-Sequence architecture overview
- Encoder-Decoder framework
- Teacher forcing mechanism for training
- Implementation using PyTorch
- GRU-based Encoder for sequence encoding
- GRU-based Decoder for sequence generation
- Applications: Machine translation, text summarization, dialogue systems
- Dynamic batching and variable-length sequences

**Key Features:**
- Time-series forecasting with GRUs (100-step input, daily temperature prediction)
- Seq2Seq implementation with teacher forcing (configurable teacher_forcing_ratio=0.5-0.7)
- Detailed encoder-decoder architecture with embedding layers
- Complete training pipeline and inference examples

## Prerequisites

Before running these notebooks, ensure you have the following:

- **Python 3.8 or higher**: The notebooks are written in Python and require a compatible version.
- **Jupyter Notebook or JupyterLab**: For interactive execution of the notebooks.
- **Basic understanding of Python programming**: Familiarity with Python syntax, data structures, and libraries like NumPy and Pandas.
- **Basic machine learning knowledge**: Understanding of neural networks concepts helps with notebooks 4-5.
- **pip**: Python package manager for installing dependencies.

## Quick Start

For those who want to get started immediately:

```bash
# 1. Clone the repository
git clone https://github.com/tanutiwarii/NLP-NOTES.git
cd NLP-NOTES

# 2. Create and activate virtual environment
python -m venv nlp_env
source nlp_env/bin/activate  # macOS/Linux
# nlp_env\Scripts\activate   # Windows

# 3. Install everything in one command
pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Start learning
jupyter lab
```

Then open any `.ipynb` file and start exploring!

## Dependencies and Libraries

This project uses the following Python libraries. Each is essential for different NLP and deep learning tasks:

### Core Jupyter Environment

**Jupyter (v1.0.0)** & **JupyterLab (v4.0.5)**
- Interactive computing environment for executing notebook cells
- Supports inline visualizations, markdown documentation, and code output
- Used in all notebooks for interactive learning

### Natural Language Processing

**NLTK - Natural Language Toolkit (v3.8.1)**
- Purpose: Core NLP library for text processing and analysis
- Key Features:
  - **Tokenization**: Breaking text into words and sentences (`word_tokenize()`, `sent_tokenize()`)
  - **Stemming**: Reducing words to their root form (`PorterStemmer`)
  - **Lemmatization**: Converting words to their base form (`WordNetLemmatizer`)
  - **POS Tagging**: Identifying parts of speech in sentences (`pos_tag()`)
  - **Named Entity Recognition (NER)**: Extracting entities like people, places, organizations (`ne_chunk()`)
  - **Stop Words**: Common words to filter out ("the", "a", "is", etc.)
- Used in: 1-Basic_NLP, 2-Text_Normalization, 3-Embedding_Techniques
- Installation: `pip install nltk==3.8.1`
- First-time data download:
  ```python
  import nltk
  nltk.download('punkt')          # Tokenization models
  nltk.download('stopwords')      # Stop words corpus
  nltk.download('wordnet')        # Lemmatization database
  nltk.download('averaged_perceptron_tagger')  # POS tagging
  nltk.download('maxent_ne_chunker')  # NER models
  ```

**Gensim (v4.3.2)**
- Purpose: Library for topic modeling and word embeddings
- Key Features:
  - **Word2Vec**: Training word embeddings from text corpora
  - **FastText**: Character-level embeddings for handling out-of-vocabulary words
  - **Topic Modeling**: Latent Dirichlet Allocation (LDA) for document clustering
  - **Document Similarity**: Finding similar documents using vector representations
  - **Corpus Processing**: Efficient handling of large text collections
- Used in: 3-Embedding_Techniques for word embeddings and topic modeling
- Installation: `pip install gensim==4.3.2`

### Scientific Computing & Data Analysis

**NumPy (v1.24.3)**
- Purpose: Numerical computing and array operations
- Key Features:
  - Fast mathematical operations on arrays and matrices
  - Foundation for pandas and scikit-learn
  - Used for numerical text encoding and vector operations
- Used in: 3-Embedding_Techniques, 4-RNN, 5-GRU for numerical computations and data processing
- Installation: `pip install numpy==1.24.3`

**Pandas (v2.0.3)**
- Purpose: Data manipulation, analysis, and cleaning
- Key Features:
  - DataFrames for organizing text data in tabular format
  - Easy data filtering, grouping, and transformation
  - CSV and Excel file handling
  - Time-series data support
- Used in: 1-Basic_NLP, 2-Text_Normalization, 3-Embedding_Techniques, 5-GRU for organizing text and time-series datasets
- Installation: `pip install pandas==2.0.3`

### Data Visualization

**Matplotlib (v3.7.2)**
- Purpose: 2D plotting and data visualization
- Key Features:
  - Line plots, histograms, scatter plots, bar charts
  - Customizable colors, labels, and formatting
  - Foundation for other visualization libraries
  - Time-series visualization
- Used in: 3-Embedding_Techniques, 4-RNN, 5-GRU for visualizing embeddings, distributions, and training results
- Installation: `pip install matplotlib==3.7.2`

**Seaborn (v0.12.2)**
- Purpose: Statistical data visualization (built on top of matplotlib)
- Key Features:
  - Enhanced aesthetics for statistical plots
  - Heatmaps, distribution plots, correlation matrices
  - Better default styling than matplotlib
- Used in: 3-Embedding_Techniques for advanced visualizations
- Installation: `pip install seaborn==0.12.2`

### Machine Learning

**Scikit-learn (v1.3.0)**
- Purpose: Machine learning algorithms and text processing tools
- Key Features:
  - **TF-IDF Vectorization**: Converting text to numerical features (`TfidfVectorizer`)
  - **Classification models**: Support Vector Machines (SVM), Naive Bayes, etc.
  - **Text preprocessing**: CountVectorizer for bag-of-words models
  - **Data Scaling**: MinMaxScaler for normalizing features (0-1 range)
  - **Model evaluation**: Cross-validation, metrics, performance evaluation
- Used in: 3-Embedding_Techniques (TF-IDF), 5-GRU (MinMaxScaler for time-series data)
- Installation: `pip install scikit-learn==1.3.0`

### Deep Learning - TensorFlow/Keras

**TensorFlow (v2.13.0)**
- Purpose: Open-source deep learning framework for building neural networks
- Key Features:
  - **Keras API**: High-level neural network API (included in TensorFlow 2.x)
  - **RNNs and LSTMs**: For sequential data processing and text generation
  - **GRUs**: Gated Recurrent Units for better gradient flow and memory retention
  - **Embedding Layers**: Converting categorical data to dense vectors
  - **Sequential Models**: Layer-by-layer neural network construction
  - **Optimizers**: Adam, SGD, and other optimization algorithms
  - **Loss Functions**: mean_squared_error, categorical_crossentropy, etc.
  - **Training Pipeline**: fit(), predict(), evaluate() methods
- Used in: 3-Embedding_Techniques (embeddings), 4-RNN (character generation), 5-GRU (time series forecasting)
- Installation: `pip install tensorflow==2.13.0`
- For GPU support: `pip install tensorflow-gpu==2.13.0` (requires CUDA)

**TensorFlow Hub (v0.13.0)**
- Purpose: Repository of pre-trained machine learning models
- Key Features:
  - **Pre-trained Embeddings**: Universal Sentence Encoder, BERT models
  - **Transfer Learning**: Fine-tuning pre-trained models on custom data
  - **Model Loading**: Easy import of trained models for inference
  - **Embedding Extraction**: Getting vector representations from text
- Used in: 3-Embedding_Techniques for loading pre-trained word embeddings
- Installation: `pip install tensorflow-hub==0.13.0`

### Deep Learning - PyTorch

**PyTorch (v1.10.0+)**
- Purpose: Deep learning framework with dynamic computation graphs
- Key Features:
  - **Dynamic Graphs**: Computational graphs change at runtime (flexible)
  - **RNN/GRU Modules**: `nn.RNN()`, `nn.GRU()` for recurrent networks
  - **Embedding Layer**: `nn.Embedding()` for word/token embeddings
  - **Sequential Models**: `nn.Sequential()` for layer composition
  - **Optimizers**: Adam, SGD for training neural networks
  - **GPU Support**: CUDA and cuDNN integration for GPU acceleration
  - **Autograd**: Automatic differentiation for backpropagation
- Used in: 5-GRU (Seq2Seq models with Encoder-Decoder architecture)
- Installation: `pip install torch` or `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (for CUDA 11.8)
- Supported CUDA versions: CUDA 10.2, 11.7, 11.8, 12.1

**torchvision (v0.11.0+)** & **torchaudio (v0.10.0+)**
- Supplementary PyTorch libraries for computer vision and audio processing
- Installation: Bundled with PyTorch installation

## Installation

### Option 1: Quick Installation (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tanutiwarii/NLP-NOTES.git
   cd NLP-NOTES
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On macOS/Linux
   # nlp_env\Scripts\activate   # On Windows
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK datasets:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker')"
   ```

5. **Verify installation:**
   ```bash
   python verify_installation.py
   ```

### Option 2: Manual Installation

If you prefer to install packages individually:

```bash
# Jupyter and interactive environment
pip install jupyter==1.0.0
pip install jupyterlab==4.0.5

# NLP core library
pip install nltk==3.8.1
pip install gensim==4.3.2

# Scientific computing
pip install numpy==1.24.3
pip install pandas==2.0.3

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Machine learning
pip install scikit-learn==1.3.0

# Deep learning - TensorFlow
pip install tensorflow==2.13.0
pip install tensorflow-hub==0.13.0

# Deep learning - PyTorch
pip install torch torchvision torchaudio
```

Then download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
```

### Option 3: Using Conda (Alternative)

If you prefer Anaconda/Miniconda:

```bash
conda create -n nlp_env python=3.10
conda activate nlp_env
conda install -c conda-forge -c pytorch jupyter jupyterlab nltk gensim numpy pandas matplotlib seaborn scikit-learn tensorflow tensorflow-hub pytorch torchvision torchaudio
```

### Option 4: GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# TensorFlow with GPU support
pip install tensorflow-gpu==2.13.0

# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Starting the Jupyter Environment

**Using Jupyter Notebook:**
```bash
jupyter notebook
```
This opens a browser window at `http://localhost:8888` showing the file browser.

**Using JupyterLab (Recommended):**
```bash
jupyter lab
```
JupyterLab offers a more modern interface with better file management and extensions.

### Running the Notebooks

1. **Navigate to the notebook file** and click to open it
2. **Read the markdown cells** for explanations and theory
3. **Execute code cells** using `Shift + Enter` or the Run button
4. **Observe the output** (results, visualizations, errors)
5. **Modify the code** to experiment and learn

### Notebook Execution Order

**Recommended Learning Path:**
1. `1-Basic_NLP.ipynb` - Start here for fundamentals
2. `2-Text_Normalization.ipynb` - Learn data preprocessing
3. `3-Embedding_Techniques.ipynb` - Understand word vectors
4. `4-RNN.ipynb` - Learn sequence models
5. `5-GRU.ipynb` - Master advanced architectures (GRU and Seq2Seq)

### Notebook Execution Guide

- **Sequential Execution**: Run cells in order (top to bottom) as variables and imports depend on previous cells
- **Cell Restart**: Use `Kernel > Restart Kernel` to clear all variables and start fresh
- **Kernel Management**: Select `Kernel` menu to interrupt, restart, or reconnect to the kernel
- **Variable Explorer**: Use `%whos` command to see all defined variables
- **Cell Magic**: Commands like `%timeit`, `%%time` are included for performance measurement

### Dataset Requirements

- **Dataset File**: `data.csv` (required for notebook 5-GRU.ipynb)
  - Format: CSV with 'Date' column and temperature values
  - Size: ~8,000 rows (8,000 days of data)
  - Location: Place in project root directory

## Examples

### Example 1: Text Normalization Pipeline

This example demonstrates using multiple packages to clean and normalize text:

```python
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Raw text
text = "Python 3.0, released in 2008, was a major revision!!!"

# 1. Convert to lowercase (built-in string method)
text = text.lower()

# 2. Remove numbers using regex (re module)
text = re.sub(r'\d+', '', text)

# 3. Remove punctuation (re module)
text = re.sub(r'[^\w\s]', '', text)

# 4. Remove extra whitespace (built-in string method)
text = text.strip()

# 5. Remove stopwords (nltk)
stop_words = set(stopwords.words('english'))
words = text.split()
filtered_words = [word for word in words if word not in stop_words]

# 6. Apply stemming (nltk)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# 7. Store results in DataFrame (pandas)
results_df = pd.DataFrame({
    'Original': words,
    'Filtered': filtered_words,
    'Stemmed': stemmed_words
})

print(results_df)
```

**Packages Used:**
- `re`: Regular expressions for pattern matching
- `nltk`: Stopwords and stemming
- `pandas`: Data organization in DataFrame format

### Example 2: TF-IDF Vectorization and Visualization

This example shows text-to-vector conversion and visualization:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample documents
documents = [
    "Python is a great programming language",
    "Machine learning uses Python extensively",
    "Data science tools and Python go together"
]

# 1. Convert text to TF-IDF vectors (scikit-learn)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 2. Convert to numpy array for analysis (numpy)
tfidf_array = tfidf_matrix.toarray()

# 3. Create DataFrame for better visualization (pandas)
df = pd.DataFrame(
    tfidf_array,
    columns=vectorizer.get_feature_names_out()
)

# 4. Visualize TF-IDF scores (matplotlib & seaborn)
plt.figure(figsize=(12, 6))
sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('TF-IDF Scores Heatmap')
plt.xlabel('Features (Words)')
plt.ylabel('Documents')
plt.tight_layout()
plt.show()
```

**Packages Used:**
- `scikit-learn`: TF-IDF vectorization
- `numpy`: Numerical operations on matrices
- `pandas`: DataFrame organization
- `matplotlib` & `seaborn`: Visualization and heatmaps

### Example 3: GRU Time-Series Forecasting

This example shows temperature prediction using GRU:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Prepare data
def create_dataset(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step=100)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build and train model
model = Sequential([
    GRU(units=50, return_sequences=True, input_shape=(100, 1)),
    GRU(units=50),
    Dense(units=1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions
input_sequence = scaled_data[-100:].reshape(1, 100, 1)
predicted = model.predict(input_sequence)
predicted_temp = scaler.inverse_transform(predicted)
print(f"Predicted temperature: {predicted_temp[0][0]:.2f}°C")
```

**Packages Used:**
- `pandas`: Data loading and manipulation
- `numpy`: Array operations
- `scikit-learn`: Data scaling
- `tensorflow.keras`: GRU model construction and training

### Example 4: Seq2Seq Encoder-Decoder

This example demonstrates sequence-to-sequence learning with PyTorch:

```python
import torch
import torch.nn as nn

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden

# Usage
encoder = Encoder(input_dim=10, emb_dim=8, hidden_dim=16)
decoder = Decoder(output_dim=10, emb_dim=8, hidden_dim=16)
```

**Packages Used:**
- `torch`: PyTorch framework
- `torch.nn`: Neural network modules

## Troubleshooting

### Common Installation Issues

**Issue: Module not found (ImportError)**
```bash
# Solution: Ensure all packages are installed
pip install -r requirements.txt
pip install --upgrade pip setuptools wheel
jupyter lab --version  # Verify installation
```

**Issue: NLTK data not found (LookupError)**
```python
# Solution: Download NLTK data
import nltk
nltk.download('all')  # Download all NLTK resources
# Or download specific datasets:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
```

**Issue: Package version conflicts**
```bash
# Solution: Use a fresh virtual environment
python -m venv nlp_env_fresh
source nlp_env_fresh/bin/activate  # macOS/Linux
pip install --upgrade -r requirements.txt
```

**Issue: Pip install fails with permission error**
```bash
# Solution: Use --user flag or upgrade pip
pip install --user -r requirements.txt
# Or use virtual environment (recommended)
python -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
```

**Issue: PyTorch installation fails**
```bash
# Solution: Use official PyTorch installation command
# Visit https://pytorch.org/get-started/locally/ for your system
# Example for CPU:
pip install torch torchvision torchaudio
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Common Runtime Issues

**Issue: Kernel not responding**
```
Solution: Restart the kernel
- Click Kernel menu > Restart Kernel
- Or close and reopen the notebook
- Or restart Jupyter: Ctrl+C in terminal and restart
```

**Issue: Port already in use (Jupyter won't start)**
```bash
# Solution: Use a different port
jupyter notebook --port 8889
jupyter lab --port 8890
# Or find and kill the process using port 8888
lsof -ti:8888 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8888   # Windows (find PID, then taskkill /PID <pid>)
```

**Issue: Memory error or slow performance**
```python
# Solution: Clear notebook variables
%reset  # Clear all variables
# Or restart kernel and run specific cells only
# For deep learning: reduce batch size or model complexity
```

**Issue: NLTK downloads fail (network error)**
```python
# Solution: Specify download path or use alternative
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')  # Should work now
```

**Issue: TensorFlow/PyTorch not using GPU**
```python
# TensorFlow GPU check
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# PyTorch GPU check
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Solution: Install GPU-enabled versions
pip install tensorflow-gpu  # TensorFlow GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118  # PyTorch with CUDA
```

**Issue: Data file (data.csv) not found**
```
Solution: Ensure data.csv is in the project root directory
Path: /Users/tannutiwari/Downloads/Projects/Jupyter/data.csv
Format: CSV with 'Date' column and temperature values
```

### Verification Steps

To verify your installation is complete and working:

```python
# Run this in a Jupyter cell
import sys
print(f"Python: {sys.version}")

# Check all required packages
packages = ['nltk', 'gensim', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow', 'torch']
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")

# Check TensorFlow GPU
import tensorflow as tf
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Check PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check NLTK data
import nltk
print(f"\nNLTK data path: {nltk.data.path}")
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/new-nlp-concept`
3. **Add your notebook or improvements**
4. **Test your changes thoroughly**
5. **Commit your changes:** `git commit -m "Add detailed explanation of [concept]"`
6. **Push to the branch:** `git push origin feature/new-nlp-concept`
7. **Open a Pull Request**

Please ensure:
- Code follows PEP 8 style guidelines
- Notebooks include clear explanations and comments
- New concepts are well-documented
- Examples are practical and educational
- All dependencies are listed in requirements.txt
