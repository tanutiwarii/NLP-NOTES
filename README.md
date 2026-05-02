# NLP Notes

## Overview

This repository serves as a comprehensive collection of Jupyter notebooks designed to teach and demonstrate fundamental concepts in Natural Language Processing (NLP). Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. The goal is to read, decipher, understand, and make sense of human language in a manner that is valuable.

These notebooks are structured to provide hands-on experience with NLP techniques, starting from basic concepts and progressing to advanced topics including deep learning with recurrent neural networks. Each notebook includes theoretical explanations, code implementations, and practical examples.

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
- **RAM**: 2 GB minimum (4 GB recommended)
- **Disk Space**: 500 MB for installation + dependencies
- **Internet Connection**: Required for NLTK data download

### Recommended Setup
- **Operating System**: macOS or Linux (better package compatibility)
- **Python Version**: 3.10 or higher
- **RAM**: 8 GB or more
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

### 4. Recurrent Neural Networks (`4-RNN.ipynb`)

This notebook introduces Recurrent Neural Networks (RNNs) for sequential data processing. Key topics include:

- Understanding RNN architecture and memory retention
- Character-level text generation using RNNs
- Sequence prediction and temporal dependencies
- Building RNN models with TensorFlow/Keras
- Training RNNs on sequential data
- Applications in NLP and time series forecasting

### 5. Gated Recurrent Units (`5-GRU.ipynb`)

This advanced notebook explores Gated Recurrent Units (GRUs), a more sophisticated RNN variant. Topics covered:

- GRU architecture: Update and Reset gates
- Advantages of GRUs over basic RNNs (gradient flow, memory)
- Time series forecasting with GRUs
- Temperature prediction using historical data
- Building GRU models with TensorFlow/Keras
- Data preprocessing for sequential models
- Model training and evaluation metrics

## Prerequisites

Before running these notebooks, ensure you have the following:

- **Python 3.8 or higher**: The notebooks are written in Python and require a compatible version.
- **Jupyter Notebook or JupyterLab**: For interactive execution of the notebooks.
- **Basic understanding of Python programming**: Familiarity with Python syntax, data structures, and libraries like NumPy and Pandas.
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

This project uses the following Python libraries. Each is essential for different NLP tasks:

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
- Used in: 3-Embedding_Techniques, 5-GRU for organizing text datasets and time series data
- Installation: `pip install pandas==2.0.3`

### Data Visualization

**Matplotlib (v3.7.2)**
- Purpose: 2D plotting and data visualization
- Key Features:
  - Line plots, histograms, scatter plots, bar charts
  - Customizable colors, labels, and formatting
  - Foundation for other visualization libraries
- Used in: 3-Embedding_Techniques for visualizing embeddings and distributions
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
  - **Model evaluation**: Cross-validation, metrics, performance evaluation
- Used in: 3-Embedding_Techniques (TF-IDF), 5-GRU (data scaling)
- Installation: `pip install scikit-learn==1.3.0`

### Deep Learning & Neural Networks

**TensorFlow (v2.13.0)**
- Purpose: Open-source deep learning framework for building neural networks
- Key Features:
  - **Keras API**: High-level neural network API (included in TensorFlow 2.x)
  - **RNNs and LSTMs**: For sequential data processing and text generation
  - **GRUs**: Gated Recurrent Units for better gradient flow
  - **Embedding Layers**: Converting categorical data to dense vectors
  - **Optimizers**: Adam, SGD, and other optimization algorithms
  - **Model Building**: Sequential and Functional API for model construction
- Used in: 3-Embedding_Techniques (embeddings), 4-RNN (character generation), 5-GRU (time series forecasting)
- Installation: `pip install tensorflow==2.13.0`

**TensorFlow Hub (v0.13.0)**
- Purpose: Repository of pre-trained machine learning models
- Key Features:
  - **Pre-trained Embeddings**: Universal Sentence Encoder, BERT models
  - **Transfer Learning**: Fine-tuning pre-trained models on custom data
  - **Model Loading**: Easy import of trained models for inference
  - **Embedding Extraction**: Getting vector representations from text
- Used in: 3-Embedding_Techniques for loading pre-trained word embeddings
- Installation: `pip install tensorflow-hub==0.13.0`

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
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK datasets:**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker')"
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

# Deep learning
pip install tensorflow==2.13.0
pip install tensorflow-hub==0.13.0
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
conda install jupyter jupyterlab nltk gensim numpy pandas matplotlib seaborn scikit-learn tensorflow tensorflow-hub
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

### Notebook Execution Guide

- **Sequential Execution**: Run cells in order (top to bottom) as variables and imports depend on previous cells
- **Cell Restart**: Use `Kernel > Restart Kernel` to clear all variables and start fresh
- **Kernel Management**: Select `Kernel` menu to interrupt, restart, or reconnect to the kernel
- **Variable Explorer**: Use `%whos` command to see all defined variables
- **Cell Magic**: Commands like `%timeit`, `%%time` are included for performance measurement

### Troubleshooting Common Issues

**Issue: Module not found (ImportError)**
```
Solution: Ensure all packages are installed
pip install -r requirements.txt
jupyter lab --version  # Verify installation
```

**Issue: NLTK data not found**
```
Solution: Download NLTK data
import nltk
nltk.download('all')  # Download all NLTK resources
```

**Issue: Kernel not responding**
```
Solution: Restart the kernel
Kernel > Restart Kernel (in menu)
or
jupyter notebook --kernel=python3 (restart Jupyter)
```

**Issue: Port already in use (Jupyter won't start)**
```
Solution: Use a different port
jupyter notebook --port 8889
jupyter lab --port 8890
```

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

### Example 3: Named Entity Recognition (NER)

This example extracts named entities from text:

```python
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk

# Ensure NER models are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

# Sample text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

# 1. Tokenize the text (nltk)
tokens = word_tokenize(text)

# 2. Apply POS tagging (nltk)
pos_tags = pos_tag(tokens)

# 3. Extract named entities (nltk)
named_entities = ne_chunk(pos_tags)

print(named_entities)
# Output shows: PERSON (Steve Jobs), GPE (Cupertino, California), ORG (Apple Inc.)
```

**Packages Used:**
- `nltk`: Tokenization, POS tagging, and NER extraction

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

### Verification Steps

To verify your installation is complete and working:

```python
# Run this in a Jupyter cell
import sys
print(f"Python: {sys.version}")

# Check all required packages
packages = ['nltk', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Tannu Tiwari**
- GitHub: [@tanutiwarii](https://github.com/tanutiwarii)


## Acknowledgments

- NLTK library for natural language processing tools
- Jupyter community for the interactive computing environment
- Open-source NLP community for inspiration and resources

---

*Happy Learning! 🚀*