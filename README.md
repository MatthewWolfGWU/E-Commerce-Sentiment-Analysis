# E-Commerce Product Classification using Deep Learning

A comprehensive text classification project comparing traditional machine learning, deep learning (RNN/LSTM), and transformer-based models for automated e-commerce product categorization using Python, TensorFlow/Keras, and Hugging Face.

## Overview

This project tackles the challenge of automated product categorization for e-commerce platforms by building and evaluating multiple text classification models. With e-commerce platforms receiving thousands to millions of products daily, manual categorization is slow and costly. This project develops high-accuracy classifiers to automatically assign products to categories based on their text descriptions.

## Business Problem

**Challenge**: E-commerce platforms need to categorize products efficiently and accurately
**Impact**: Manual categorization is time-consuming, expensive, and doesn't scale
**Solution**: Automated text classification using machine learning and deep learning models

## Dataset

**Source**: Kaggle - E-Commerce Text Dataset (scraped from Indian e-commerce platform)
- **Size**: ~50,000 product descriptions
- **Features**: Product description text
- **Target Variable**: Product category (4 classes)

**Class Distribution**:
- **Household**: 38% (~19,000 products)
- **Books**: 22% (~11,000 products)
- **Clothing & Accessories**: 20% (~10,000 products)
- **Electronics**: 19% (~9,500 products)

**Text Characteristics**:
- **Average tokens per entry**: 115
- **95th percentile**: 312 tokens
- **Padding length**: 325 tokens (capturing majority of information)

## Model Architecture & Results

### 1. Traditional Machine Learning (Baseline Models)

#### Logistic Regression
**Architecture**:
- TF-IDF vectorization (max_features=1000, n-gram range=(1,2))
- Multinomial logistic regression classifier

**Performance**:
- **Test Accuracy**: 92.6%
- Strong baseline performance with simple, interpretable model

#### Support Vector Machine (SVM)
**Architecture**:
- TF-IDF vectorization (max_features=1000, n-gram range=(1,2))
- Linear SVM classifier

**Performance**:
- **Test Accuracy**: 92.7%
- Comparable to logistic regression
- Note: Accuracy increases with higher max_features (up to ~98% with 15,000 features)

### 2. Deep Learning Models

#### Simple RNN (3-Layer)
**Architecture**:
- Embedding layer: input_dim=20,000, output_dim=64
- 3 RNN layers: 64 neurons each, dropout=0.3
- Dense layer: 64 neurons
- Output: Softmax activation (4 classes)

**Training Configuration**:
- Batch size: 64
- Epochs: 7 (with early stopping)
- Optimizer: Adam

**Performance**:
- **Training Accuracy**: 97.84%
- **Test Accuracy**: 91.1%
- **Challenge**: Overfitting; struggled distinguishing household items from other categories

#### LSTM (2-Layer) ⭐ **BEST MODEL**
**Architecture**:
- Embedding layer: input_dim=20,000, output_dim=64
- 2 LSTM layers: 64 neurons each, dropout=0.3
- Output: Softmax activation (4 classes)

**Training Configuration**:
- Batch size: 64
- Epochs: 3
- No early stopping required

**Performance**:
- **Training Accuracy**: 95.3%
- **Test Accuracy**: 94.68% 🏆
- **Improvement**: 5% better household classification than RNN
- **Advantage**: Less overfitting, better generalization

#### Pre-trained Transformer (BART-large-MNLI)
**Model Details**:
- Facebook BART-large-MNLI (400M parameters)
- Pre-trained on MultiNLI dataset
- Inference time: ~3 hours for 5,500 test samples

**Performance**:

**Without Task-Specific Prompt**:
- Test Accuracy: 69.7%
- Issue: Lack of domain context

**With Custom Prompt** ("This amazon product description is about {}."):
- **Test Accuracy**: 79.1%
- **Improvement**: +9.4% with simple prompt engineering
- **Key Insight**: Household vs. Electronics classification significantly improved

### Model Comparison Summary

| Model | Test Accuracy | Training Time | Parameters | Notes |
|-------|---------------|---------------|------------|-------|
| **LSTM (2 layers)** | **94.68%** 🏆 | Fast | ~1.3M | Best overall |
| Logistic Regression | 92.6% | Very Fast | Minimal | Strong baseline |
| SVM (TF-IDF) | 92.7% | Fast | Minimal | Scalable with features |
| RNN (3 layers) | 91.1% | Moderate | ~1.5M | Overfitting issues |
| BART (with prompt) | 79.1% | Very Slow | 400M | Potential with fine-tuning |
| BART (no prompt) | 69.7% | Very Slow | 400M | Needs task context |

## Technologies Used

- **Python 3**: Primary programming language
- **Deep Learning Frameworks**:
  - TensorFlow/Keras: Neural network construction
  - PyTorch (via Hugging Face): Transformer models
- **Machine Learning**:
  - scikit-learn: Traditional ML models, TF-IDF, evaluation metrics
- **NLP & Text Processing**:
  - Keras Tokenizer: Text tokenization and padding
  - Hugging Face Transformers: Pre-trained models (BART)
  - TF-IDF Vectorization: Feature extraction
- **Data Analysis**:
  - pandas: Data manipulation
  - numpy: Numerical operations
  - matplotlib/seaborn: Visualization

## Methodology

### 1. Data Preprocessing
- Loaded 50,000 product descriptions
- Analyzed class distribution (handled slight imbalance)
- Split data: training set and test set (~11% holdout)

### 2. Text Tokenization
**Traditional ML**:
- TF-IDF vectorization with unigrams and bigrams
- Maximum features: 1,000-15,000 (varied by experiment)

**Deep Learning**:
- Keras Tokenizer with vocabulary size of 20,000
- Padding/truncating to 325 tokens (95th percentile)
- One-hot encoding of labels for neural networks

### 3. Model Development
**Baseline Models**:
- Implemented Logistic Regression and SVM with TF-IDF

**Neural Networks**:
- Built RNN and LSTM architectures from scratch
- Implemented dropout regularization (0.3 rate)
- Used early stopping to prevent overfitting

**Transfer Learning**:
- Leveraged BART-large-MNLI from Hugging Face
- Experimented with zero-shot classification
- Applied prompt engineering for task adaptation

### 4. Model Evaluation
- Primary metric: Test accuracy
- Confusion matrices for error analysis
- Class-wise performance breakdown

## Key Findings

### Classification Challenges
1. **Household Category Ambiguity**: All models struggled most with household items
   - Books, clothing, and electronics could reasonably fall under "household"
   - Required clearer semantic boundaries

2. **Model Complexity Trade-offs**:
   - LSTM found optimal balance between complexity and performance
   - More complex RNN (3 layers) overfit and underperformed
   - Simple models (Logistic/SVM) achieved strong 92%+ accuracy

3. **Prompt Engineering Impact**:
   - Simple prompt boosted BART accuracy by 9.4%
   - Demonstrates importance of task-specific context for LLMs
   - Suggests potential for further improvement with fine-tuning

### Confusion Matrix Insights

**RNN Issues**:
- Misclassified household items at higher rates
- Struggled with subtle category distinctions

**LSTM Improvements**:
- 5% better household classification accuracy
- More robust to category ambiguity
- Better generalization despite fewer epochs

**BART with Prompt**:
- Significantly improved household vs. electronics distinction
- Still underperformed task-specific trained models

## Project Structure

```
.
├── text.ipynb                  # Main Jupyter notebook with all models
├── Project Report.pdf          # Comprehensive project report
├── ecommerceDataset.csv        # Product descriptions dataset
└── README.md                   # Project documentation
```

## Future Work & Limitations

### Limitations
1. **Dataset Documentation**: Limited information on how categories were originally assigned
2. **Computational Resources**: BART inference took ~3 hours on test set
3. **Class Ambiguity**: Overlapping semantic boundaries between categories

### Proposed Improvements

1. **Parameter-Efficient Fine-Tuning (LoRA)**:
   - Fine-tune small subset of BART weights for task
   - Reduce computational cost while improving accuracy
   - Potential to exceed LSTM performance

2. **Alternative Architectures**:
   - Experiment with other transformers (RoBERTa, DistilBERT)
   - Test bidirectional LSTMs
   - Ensemble methods combining multiple models

3. **Enhanced Prompt Engineering**:
   - Few-shot learning with example descriptions
   - Chain-of-thought prompting
   - Task-specific instruction tuning

4. **Data Augmentation**:
   - Back-translation for augmentation
   - Synthetic product descriptions
   - Better handling of class imbalance

5. **Explainability**:
   - SHAP values for feature importance
   - Attention weight visualization
   - Error analysis for targeted improvements

## Business Impact

### Cost Savings
- Automated categorization reduces manual labor costs
- Scales to millions of products without linear cost increase

### Operational Efficiency
- Real-time product categorization at upload
- Consistent labeling across platform
- Reduced time-to-market for new products

### User Experience
- Improved search and filtering accuracy
- Better product recommendations
- Enhanced catalog organization

### Model Deployment Considerations
- **LSTM recommended** for production: 94.68% accuracy with fast inference
- SVM viable for resource-constrained environments: 92.7% accuracy with minimal resources
- BART potential with LoRA fine-tuning for highest accuracy requirements

## Academic Context

**Course**: DNSC 4280 - Machine Learning
**Institution**: George Washington University
**Date**: December 2025

## Contributors

- Jesse Mutamba
- Henrique Cassol
- Maya Serna
- Matthew Wolf

## References

1. Gautam. "E Commerce Text Dataset." Kaggle, 2019. [Link](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
2. "Facebook/Bart-Large-Mnli." Hugging Face. [Link](https://huggingface.co/facebook/bart-large-mnli)

## License

This project was completed as part of academic coursework at George Washington University. Dataset sourced from Kaggle under their terms of use.
