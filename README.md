# IAB Taxonomy Classifier

A token-based taxonomy classifier that assigns [IAB Content Taxonomy](https://iabtechlab.com/standards/content-taxonomy/) categories to text documents using word co-occurrence statistics and similarity scoring, no neural networks or LLM API calls required.

## Inspiration

This project takes inspiration from the **GenAITechLab Taxonomy LLM** project by Vincent Granville ([GitHub](https://github.com/VincentGranville/Large-Language-Models/tree/main/xllm6)), which builds a multi-LLM knowledge retrieval system using taxonomy tables derived from crawled web data. The original project uses a Wolfram "Probability & Statistics" ontology. This project adapts the core algorithms to work with the IAB Content Taxonomy for advertiser vertical classification.

## How It Works

1. **Data Preparation** (`prepare_iab_data.py`): Downloads the [20 Newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups) dataset from HuggingFace and maps its 20 categories to IAB Tier 1 and Tier 2 verticals. Builds a word dictionary, category mappings, and related-word tables from the corpus.

2. **Taxonomy Classification** (`taxonomy_iab.py`): Builds taxonomy tables (topWords, wordGroups, connectedTopWords) from the dictionary, then assigns IAB categories to words and documents using a token-overlap similarity metric:

   ```
   similarity = sum(weight^0.50) / max(normA, normB)
   ```

   where weights come from dictionary word frequencies. The algorithm also incorporates:
   - Inverted index for fast connectedTopWords computation
   - Stem-aware matching (simple plural stripping)
   - Depth-weighted scoring (deeper IAB categories get a bonus)
   - Top-K candidate evaluation

## IAB Taxonomy Coverage

- **26 Tier 1 categories** (e.g., Technology & Computing, Sports, News & Politics)
- **~400 Tier 2 subcategories** (e.g., Artificial Intelligence, Basketball, Elections)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare data (downloads 20 Newsgroups, builds tables)
python prepare_iab_data.py

# Step 2: Run taxonomy classifier
python taxonomy_iab.py
```

### Requirements

- Python 3.8+
- pandas, numpy, requests, datasets (see `requirements.txt`)

## Output

The classifier produces:
- **Word-level assignments**: Each dictionary word gets an IAB category with a similarity score
- **Document-level classification**: Each document is classified by aggregating word-category scores
- **Evaluation metrics**: Tier 1 accuracy, Tier 2 accuracy, and per-category breakdowns
- Results saved to `iab_data/` directory

## Sample Results (20 Newsgroups)

| Metric | Score |
|--------|-------|
| Tier 1 Accuracy | ~28% |
| Tier 2 Accuracy | ~8% |
| Best Category (Technology & Computing) | ~51% |

These results use a simple token-overlap heuristic with no ML training. The approach is meant as a lightweight baseline and taxonomy exploration tool, not a production classifier.

## Project Structure

```
iab-taxonomy-classifier/
  prepare_iab_data.py      # Data download and preprocessing
  taxonomy_iab.py           # Main classifier (standalone, no external deps beyond stdlib)
  requirements.txt          # Python dependencies
  iab_data/
    iab_taxonomy.txt        # Full IAB Content Taxonomy (Tier 1 + Tier 2)
    iab_stopwords.txt       # Stopwords list
    (generated files)       # Created by prepare_iab_data.py and taxonomy_iab.py
```
