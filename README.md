# Product Title Conciseness Classifier
**CS5143 - Natural Language Processing (Spring 2026) | Programming Assignment 2**

A deep learning system that predicts whether an e-commerce product title is **concise (1)** or **not concise (0)**. Trained and evaluated on 36,283 product listings from Malaysia, Philippines, and Singapore.

> **Evaluation metric:** Root-Mean-Square Error (RMSE) on predicted probabilities - lower is better.

---

## Results Summary

| Model | Val RMSE | Improvement | Accuracy |
|---|---|---|---|
| Naive Baseline (always guess 0.685) | 0.4644 | - | - |
| E1 Logistic Regression | 0.3486 | −24.9% | - |
| E2 TextCNN + GloVe | 0.3407 | −26.6% | - |
| E3 BiLSTM + GloVe | 0.3288 | −29.2% | - |
| **E4 DistilBERT (best)** | **0.3245** | **−30.1%** | **86.3%** |

---

## 1. Data Preparation

**Dataset:** 36,283 product listings with fields: `title`, `category_lvl_1/2/3`, `short_description`, `price`, `product_type`, `concise_label`.

### What the Data Looks Like

![EDA Plots](<model weights and tokenizer/eda_plots.png>)

*Left: label distribution (68.5% concise vs 31.5% not). Centre: concise titles are shorter (peak ~8 words) vs non-concise (peak ~12 words). Right: concise rate per category - Fashion is the hardest, Home Appliances the easiest.*

- **68.5%** concise (24,866 samples) vs **31.5%** not concise (11,417 samples)
- Concise titles average **10.6 words** vs **14.7 words** for non-concise
- Category is a strong signal: Fashion only 44.8% concise vs Computers & Laptops at 83.3%

### Preprocessing
- HTML tags removed from `short_description`
- Missing/broken prices replaced with the country-specific median
- Stratified 85/15 split → **30,840 train** / **5,443 validation** (same label ratio in both)
- Non-concise class given extra training weight (`pos_weight = 0.4591`) to handle imbalance

### 10 Engineered Features

| Feature | What it measures | Corr. with label |
|---|---|---|
| `title_word_count` | Number of words in the title | −0.38 |
| `title_char_count` | Number of characters in the title | −0.43 |
| `title_type_token_ratio` | Fraction of unique words (low = lots of repetition) | +0.31 |
| `title_has_parentheses` | Contains brackets `()` or `[]` | - |
| `title_has_model_num` | Contains a model number like `ABC123` | - |
| `title_desc_overlap` | How many title words also appear in the description | - |
| `price_log_norm` | Price scaled within its country | +0.09 |
| `title_all_upper` | Entire title written in CAPITAL LETTERS | −0.07 |
| `title_frac_upper_words` | Fraction of ALL_CAPS words (brand/model name cues) | +0.14 |
| `price_cat_norm` | Price scaled within its product category | +0.09 |

All features are standardized with `StandardScaler` fit on training data only.

---

## 2. Models

Each model outputs a probability between 0 and 1 (how likely the title is concise). RMSE measures quality - it penalises confident wrong answers more than uncertain ones.

### E1 - Logistic Regression (Baseline)
Converts titles into word-frequency numbers (TF-IDF, 5,000 features, unigram+bigram) and learns which words predict conciseness. No deep learning - used as a simple starting point.
- Parameters: ~5,300

### E2 - TextCNN + GloVe
Uses sliding windows (sizes 3, 4, 5 words) to scan the title for important short phrases. Starts with pre-trained GloVe-100d word vectors (71.4% vocabulary coverage).
- Parameters: 1,151,273

### E3 - Bidirectional LSTM + GloVe *(Chapter 9)*
Reads the title word by word - forwards and backwards - so it understands context on both sides of every word. Good at spotting repeated or redundant information anywhere in the title.
- 2 layers, hidden size = 128, dropout = 0.3
- Parameters: 665,089

![BiLSTM Training Curve](<model weights and tokenizer/bilstm_curve.png>)

*BiLSTM training curve: training loss (blue) decreases steadily while validation RMSE (red) reaches its best at epoch 5 then levels off. Early stopping triggers at epoch 10.*

### E4 - DistilBERT *(Primary Model)*
A pre-trained transformer that understands language deeply. The product category is added before the title so the model can check whether title words actually match the product type - which is exactly what "concise" means in this task.

Input format:
```
[CLS] category_lvl_1 | category_lvl_2 | category_lvl_3 [SEP] title [SEP]
```
The `[CLS]` embedding (768-dim) is combined with the 10 handcrafted features before the final classifier.
- Learning rates: BERT body `2e-5`, classifier head `1e-4`
- Parameters: 66,562,561

---

## 3. Experiment Design

All four models are trained and tested on the **same data split** (seed=42) for a fair comparison.

| Exp | Optimizer | Max Epochs | Early Stopping |
|---|---|---|---|
| E1 - LR | LBFGS, balanced class weights | 1,000 iter | - |
| E2 - TextCNN | Adam lr=1e-3 | 15 | After 5 non-improving epochs |
| E3 - BiLSTM | Adam lr=1e-3 + LR decay | 20 | After 5 non-improving epochs |
| E4 - DistilBERT | AdamW, 10% warmup | 5 | After 2 non-improving epochs |

---

## 4. Results

![Model Comparison](<model weights and tokenizer/model_comparison.png>)

*All deep learning models clearly beat the naive baseline. DistilBERT (E4) achieves the lowest RMSE, followed closely by BiLSTM (E3).*

### RMSE by Product Category (DistilBERT)

![Per-Category RMSE](<model weights and tokenizer/per_category_rmse.png>)

*Electronics categories (Home Appliances, Computers) are easiest because product specs are objective. Style categories (Watches, Jewellery, Fashion) are hardest because "informative enough" is more subjective.*

| Category | RMSE | Category | RMSE |
|---|---|---|---|
| Watches, Sunglasses & Jewellery | 0.371 | Health & Beauty | 0.315 |
| Home & Living | 0.363 | Mobiles & Tablets | 0.287 |
| Cameras | 0.357 | Computers & Laptops | 0.257 |
| TV, Audio / Video, Gaming | 0.345 | Home Appliances | 0.238 |
| Fashion | 0.325 | | |

DistilBERT achieves the best RMSE of **0.3245** and **86.3% accuracy** - a 30.1% improvement over the naive baseline. Every deep learning model outperforms logistic regression, confirming that neural architectures capture richer language patterns.

---

## 5. Error Analysis (DistilBERT - 5,443 validation samples)

**746 total errors:** 424 False Positives + 322 False Negatives | **Accuracy: 86.3%**

### Word Count of Errors vs Correct Predictions

![Error Word Count](<model weights and tokenizer/error_word_count.png>)

*Correct predictions (blue) cluster tightly around short titles. False Positives (yellow) occur at short word counts - the model wrongly calls short titles concise. False Negatives (red) are spread across longer titles - the model over-penalises length.*

### False Positives (424) - Predicted concise, actually not
Short titles with brand names or model numbers that *look* concise, but annotators felt they were missing enough product detail.

```
p=0.990 | Mobiles & Tablets | "Luxury Perfume Bottle Case For Samsung Galaxy S5 I9600 (Black)"
p=0.989 | Mobiles & Tablets | "SAMSUNG GALAXY J5 PRIME 16GB (White Gold)"
p=0.980 | Home Appliances   | "LG 11.0KG WASHING MACHINE T2311"
```

**Why it happens:** The model sees a short title with a model number and assumes it is concise - but the annotation standard expected more product detail than just a name and a code.

### False Negatives (322) - Predicted not concise, actually concise
Longer titles where every word adds useful information, but the model wrongly flags them as too long.

```
p=0.009 | Mobiles & Tablets | "Samsung 3 Metres USB Charging Cable for Galaxy s6/s7 edge (White)"
p=0.011 | Fashion           | "Bifold Wallet Men's Genuine Leather Brown Credit/ID Card Holder Slim Purse"
p=0.015 | Watches           | "RIS Aliceband Hat Fascinator Feather Headband Wedding Lady Royal Ascot Pink"
```

**Why it happens:** Fashion and accessories titles list all important attributes (material, colour, compatibility) - which looks long but is not redundant. The model is too sensitive to title length in these cases.

### Model Calibration

![Calibration](<model weights and tokenizer/calibration.png>)

*Reliability diagram: the model is well-calibrated at high confidence (top-right closely follows the perfect diagonal). At lower probabilities (0.2-0.5 range) the model is slightly underconfident. Overall calibration is good and acceptable for RMSE purposes.*

### Key Takeaways
- More false positives than false negatives (424 vs 322) - the model leans toward predicting "concise", matching the 68.5% majority class
- Hardest categories: Watches & Jewellery and Home & Living - subjective annotation
- Easiest categories: Computers & Laptops and Home Appliances - objective spec patterns
- The model is slightly overconfident at very high probabilities - correctable with temperature scaling

---

## Project Structure

```
nlpa2/
├── product-title-classifier-24k7606.ipynb          # Main notebook - all code, models, and results
├── CS5143-NLP Spring 2026 A2-1.pdf      # Submission report
├── README.md                        # This file
├── CS5143-NLP PA2 data_train.csv    # Training data
├── glove.6B.100d.txt                # GloVe embeddings (downloaded automatically)
├── model weights and tokenizer/
│   ├── distilbert_finetuned.pt      # Saved DistilBERT weights
│   ├── bilstm_model.pt              # Saved BiLSTM weights
│   ├── textcnn_model.pt             # Saved TextCNN weights
│   ├── distilbert_tokenizer/        # Saved tokenizer files
│   ├── word2idx.json                # Vocabulary index
│   ├── feature_scaler.pkl           # Fitted StandardScaler
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│   ├── eda_plots.png                # Label distribution & EDA charts
│   ├── bilstm_curve.png             # BiLSTM training curve
│   ├── model_comparison.png         # All models RMSE comparison
│   ├── per_category_rmse.png        # DistilBERT per-category RMSE
│   ├── error_word_count.png         # Word count distribution by error type
│   └── calibration.png             # DistilBERT reliability diagram
```

## How to Run

Open `product-title-classifier-24k7606.ipynb` in Google Colab or Jupyter and run all cells top to bottom. GloVe embeddings (~822 MB) are downloaded automatically if not present. GPU recommended for DistilBERT training.
