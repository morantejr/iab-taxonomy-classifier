"""
taxonomy_iab.py — Advertiser vertical classification using IAB Content Taxonomy.

Adapted from taxonomy_improved.py (Section 8.2 of the Granville textbook).
Uses data prepared by prepare_iab_data.py:
  - 11,314 documents from 20 Newsgroups (mapped to IAB verticals)
  - 40,524 dictionary words
  - IAB Content Taxonomy (26 Tier 1 verticals, ~400 Tier 2 subcategories)

The same algorithm that classifies Wolfram "Probability & Statistics" pages
into math subcategories is applied here to classify advertiser content
into IAB ad-industry verticals.

Steps:
  1. Load tables (dictionary, taxonomy, categories)
  2. Build taxonomy from scratch (topWords, connectedTopWords)
  3. Assign IAB categories to dictionary words using similarity
  4. Evaluate: compare assigned vs actual categories
  5. Classify documents into IAB verticals
"""
import os
import time
from collections import Counter, defaultdict

DATA_DIR = "iab_data"


#--- [1] Load tables (standalone, no xllm6_util dependency)

print("=" * 60)
print("STEP 1: Loading IAB tables")
print("=" * 60)

# Load dictionary
dictionary = {}
with open(os.path.join(DATA_DIR, "iab_dictionary.txt")) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            dictionary[parts[0]] = int(parts[1])

# Load arr_url
arr_url = []
with open(os.path.join(DATA_DIR, "iab_arr_url.txt")) as f:
    for line in f:
        arr_url.append(line.strip())

# Load hash_category
hash_category = defaultdict(dict)
with open(os.path.join(DATA_DIR, "iab_hash_category.txt")) as f:
    for line in f:
        parts = line.strip().split('\t')
        word = parts[0]
        # Pairs of (category_string, count)
        i = 1
        while i + 1 < len(parts):
            cat_str = parts[i]
            count = int(parts[i + 1])
            hash_category[word][cat_str] = count
            i += 2

# Load url_map
url_map = defaultdict(dict)
with open(os.path.join(DATA_DIR, "iab_url_map.txt")) as f:
    for line in f:
        parts = line.strip().split('\t')
        word = parts[0]
        i = 1
        while i + 1 < len(parts):
            doc_id = parts[i]
            count = int(parts[i + 1])
            url_map[word][doc_id] = count
            i += 2

# Load IAB taxonomy
categories = {}       # category_name -> depth
parent_categories = {}  # category_name -> parent_name
with open(os.path.join(DATA_DIR, "iab_taxonomy.txt")) as f:
    for line in f:
        parts = line.strip().split('\t|\t')
        if len(parts) == 3:
            cat_name = parts[0].strip().lower().replace(' ', '~')
            parent_name = parts[1].strip().lower().replace(' ', '~')
            depth = int(parts[2].strip())
            categories[cat_name] = depth
            if depth > 1:
                parent_categories[cat_name] = parent_name

# Load doc ground truth
doc_categories = {}
with open(os.path.join(DATA_DIR, "iab_doc_categories.txt")) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            doc_id = int(parts[0])
            tier2 = parts[1].lower().replace(' ', '~')
            tier1 = parts[2].lower().replace(' ', '~')
            doc_categories[doc_id] = (tier2, tier1)

# Load stopwords
stopwords = set()
with open(os.path.join(DATA_DIR, "iab_stopwords.txt")) as f:
    for line in f:
        stopwords.add(line.strip())

print(f"  dictionary:       {len(dictionary):,} words")
print(f"  documents:        {len(arr_url):,}")
print(f"  IAB categories:   {len(categories)}")
print(f"  hash_category:    {len(hash_category):,} words with category labels")
print(f"  url_map:          {len(url_map):,} words with doc mappings")

print(f"\nTop 20 words by count:")
for w, c in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {c:6d}  {w}")

print(f"\nIAB Tier 1 categories:")
for cat, depth in sorted(categories.items()):
    if depth == 1:
        subcats = sum(1 for c, d in categories.items() if d == 2 and parent_categories.get(c) == cat)
        print(f"  {cat:35s} ({subcats} subcategories)")


#--- [2] Build taxonomy from scratch (same algorithm as taxonomy_improved.py)

print("\n" + "=" * 60)
print("STEP 2: Building taxonomy from scratch (topWords + connections)")
print("=" * 60)

ignoreWords = stopwords | {"also", "would", "could", "much", "many", "well"}

def create_taxonomy_tables_optimized(threshold, thresh2, ignoreWords, dictionary):
    """
    Same optimized algorithm from taxonomy_improved.py:
    Uses inverted index for connectedTopWords instead of O(T^2 * D) loop.
    """
    topWords = {}
    wordGroups = {}
    connectedTopWords = {}
    smallDictionary = {}
    connectedByTopWord = {}
    missingConnections = {}

    for word in dictionary:
        n = dictionary[word]
        if n > threshold and word not in ignoreWords:
            topWords[word] = n

    for topWord in topWords:
        hash = {}
        for word in dictionary:
            n2 = dictionary[word]
            if topWord in word and n2 > thresh2 and word != topWord:
                hash[word] = n2
        if hash:
            hash = dict(sorted(hash.items(), key=lambda item: item[1], reverse=True))
        else:
            missingConnections[topWord] = 1
        wordGroups[topWord] = hash

    for topWord in topWords:
        for word in dictionary:
            if topWord in word:
                smallDictionary[word] = dictionary[word]

    # Inverted index approach (optimized)
    word_to_topwords = {}
    for word in smallDictionary:
        contained = [tw for tw in topWords if tw in word]
        if contained:
            word_to_topwords[word] = contained

    for word, tws in word_to_topwords.items():
        for twA in tws:
            for twB in tws:
                if twA != twB:
                    key = (twA, twB)
                    connectedTopWords[key] = connectedTopWords.get(key, 0) + 1

    for (twA, twB), count in connectedTopWords.items():
        if twA not in connectedByTopWord:
            connectedByTopWord[twA] = {}
        connectedByTopWord[twA][twB] = count

    for tw in connectedByTopWord:
        connectedByTopWord[tw] = dict(sorted(
            connectedByTopWord[tw].items(), key=lambda item: item[1], reverse=True))

    return [topWords, wordGroups, connectedTopWords,
            smallDictionary, connectedByTopWord, missingConnections]


threshold = 30
thresh2 = 2

t0 = time.time()
tables = create_taxonomy_tables_optimized(threshold, thresh2, ignoreWords, dictionary)
elapsed = time.time() - t0

topWords = tables[0]
wordGroups = tables[1]
connectedTopWords = tables[2]
smallDictionary = tables[3]
connectedByTopWord = tables[4]
missingConnections = tables[5]

print(f"  Built in {elapsed:.1f}s")
print(f"  topWords:           {len(topWords)}")
print(f"  smallDictionary:    {len(smallDictionary)}")
print(f"  connectedTopWords:  {len(connectedTopWords)}")
print(f"  missingConnections: {len(missingConnections)}")

print(f"\nTop 30 content-derived categories (topWords):")
for w, c in sorted(topWords.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"  {c:5d}  {w}")

print(f"\nSample word groups:")
for topWord in list(topWords.keys())[:5]:
    group = wordGroups[topWord]
    if group:
        print(f"\n  '{topWord}' (count={topWords[topWord]}):")
        for word, count in list(group.items())[:5]:
            print(f"    {count:5d}  {word}")


#--- [3] Assign IAB categories to dictionary words (improved similarity)

print("\n" + "=" * 60)
print("STEP 3: Assigning IAB categories to dictionary words")
print("=" * 60)

def simple_stem(word):
    """Basic stemming for fuzzy matching."""
    if len(word) <= 3:
        return word
    for suffix in ['ation', 'tion', 'sion', 'ment', 'ness', 'ally', 'ical',
                   'ious', 'ous', 'ing', 'ity', 'ive', 'ful', 'ble',
                   'ies', 'ied', 'ly', 'ed', 'al', 'er', 'es']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 4:
        return word[:-1]
    return word


def compute_similarity(dictionary, word, category, mode='depth', depth=0):
    """
    Improved similarity from taxonomy_improved.py:
    - Exact token matching + stem-based partial matching
    - Depth weighting to favor specific subcategories
    """
    tokensA = word.split("~")
    tokensB = category.split("~")

    stemsA = {simple_stem(t): t for t in tokensA}
    stemsB = {simple_stem(t): t for t in tokensB}

    normA = sum(dictionary.get(t, 0) ** 0.50 for t in tokensA)
    normB = sum(dictionary.get(t, 0) ** 0.50 for t in tokensB)

    if max(normA, normB) == 0:
        return 0.0

    similarity = 0
    # Exact match
    for tA in tokensA:
        for tB in tokensB:
            if tA == tB and tA in dictionary:
                similarity += dictionary[tA] ** 0.50

    # Stem match bonus
    for stemA, origA in stemsA.items():
        for stemB, origB in stemsB.items():
            if stemA == stemB and origA != origB:
                wA = dictionary.get(origA, 0)
                wB = dictionary.get(origB, 0)
                similarity += 0.5 * (min(wA, wB) ** 0.50)

    similarity /= max(normA, normB)

    # Depth weighting
    if mode == 'depth' and depth > 0:
        similarity *= (1 + 0.15 * depth)

    return similarity


def assign_categories(dictionary, categories, mode='depth', top_k=3):
    """Assign IAB categories to each dictionary word."""
    assignedCategories = {}
    counter = 0

    for word in dictionary:
        candidates = []
        for category in categories:
            depth = categories[category]
            sim = compute_similarity(dictionary, word, category,
                                     mode=mode, depth=depth)
            if sim > 0:
                candidates.append((category, depth, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)

        if candidates:
            assignedCategories[word] = {
                'best': candidates[0],
                'top_k': candidates[:top_k]
            }
        else:
            assignedCategories[word] = {
                'best': ("", 0, 0.0),
                'top_k': []
            }

        if counter % 5000 == 0:
            best = assignedCategories[word]['best']
            print(f"  {counter:5d}/{len(dictionary)}: sim={best[2]:.2f} "
                  f"{word} -> {best[0]}")
        counter += 1

    return assignedCategories


t0 = time.time()
assignedCategories = assign_categories(dictionary, categories, mode='depth', top_k=3)
elapsed = time.time() - t0
print(f"\n  Assigned in {elapsed:.1f}s")

# Statistics
uncategorized = sum(1 for w in assignedCategories if assignedCategories[w]['best'][0] == "")
high_sim = sum(1 for w in assignedCategories if assignedCategories[w]['best'][2] >= 0.5)
perfect = sum(1 for w in assignedCategories if assignedCategories[w]['best'][2] >= 1.0)

print(f"  Total words:    {len(assignedCategories):,}")
print(f"  Uncategorized:  {uncategorized:,} ({100*uncategorized/len(assignedCategories):.1f}%)")
print(f"  High sim >=0.5: {high_sim:,} ({100*high_sim/len(assignedCategories):.1f}%)")
print(f"  Perfect >=1.0:  {perfect:,}")

# Show sample assignments by IAB vertical
print(f"\nSample assignments per IAB Tier 1:")
tier1_examples = defaultdict(list)
for word in assignedCategories:
    best = assignedCategories[word]['best']
    cat, depth, sim = best
    if sim >= 0.5 and cat:
        parent = parent_categories.get(cat, cat)
        tier1_examples[parent].append((word, cat, sim))

for tier1 in sorted(tier1_examples.keys()):
    examples = sorted(tier1_examples[tier1], key=lambda x: x[2], reverse=True)[:3]
    print(f"\n  {tier1}:")
    for word, cat, sim in examples:
        print(f"    sim={sim:.2f}  {word} -> {cat}")


#--- [4] Evaluate: word-level accuracy

print("\n" + "=" * 60)
print("STEP 4: Word-level evaluation (assigned vs actual)")
print("=" * 60)

# Get actual category per word (from hash_category, take highest count)
word_actual = {}
for word in hash_category:
    cats = hash_category[word]
    if cats:
        best_cat_str = max(cats, key=cats.get)
        parts = best_cat_str.split(' | ')
        word_actual[word] = parts[0].strip().lower().replace(' ', '~')

matches = 0
parent_matches = 0
total = 0
uncategorized_eval = 0

for word in assignedCategories:
    if word in word_actual:
        total += 1
        assigned_cat = assignedCategories[word]['best'][0]
        actual_cat = word_actual[word]

        if assigned_cat == "":
            uncategorized_eval += 1
        elif assigned_cat == actual_cat:
            matches += 1
        else:
            if parent_categories.get(actual_cat) == assigned_cat:
                parent_matches += 1
            elif parent_categories.get(assigned_cat) == actual_cat:
                parent_matches += 1

print(f"  Total compared:     {total:,}")
print(f"  Exact matches:      {matches:,} ({100*matches/max(total,1):.1f}%)")
print(f"  Parent matches:     {parent_matches:,}")
print(f"  Exact+Parent:       {matches+parent_matches:,} ({100*(matches+parent_matches)/max(total,1):.1f}%)")
print(f"  Uncategorized:      {uncategorized_eval:,}")

# Top-K evaluation
topk_matches = 0
topk_parent = 0
for word in assignedCategories:
    if word in word_actual:
        actual_cat = word_actual[word]
        top_k = assignedCategories[word].get('top_k', [])
        found_exact = False
        found_parent = False
        for (cat, depth, sim) in top_k:
            if cat == actual_cat:
                found_exact = True
                break
            if parent_categories.get(actual_cat) == cat or parent_categories.get(cat) == actual_cat:
                found_parent = True
        if found_exact:
            topk_matches += 1
        elif found_parent:
            topk_parent += 1

print(f"\n  Top-3 exact:        {topk_matches:,} ({100*topk_matches/max(total,1):.1f}%)")
print(f"  Top-3 parent:       {topk_parent:,}")
print(f"  Top-3 total:        {topk_matches+topk_parent:,} ({100*(topk_matches+topk_parent)/max(total,1):.1f}%)")


#--- [5] Document-level classification

print("\n" + "=" * 60)
print("STEP 5: Document-level vertical classification")
print("=" * 60)

# For each document, aggregate word-level category assignments
# weighted by word frequency in that document

doc_predicted = {}

for doc_id in range(len(arr_url)):
    # Get all words in this document
    doc_words = []
    for word in url_map:
        if str(doc_id) in url_map[word]:
            doc_words.append(word)

    if not doc_words:
        continue

    # Aggregate category scores across all words in the document
    cat_scores = Counter()
    for word in doc_words:
        if word in assignedCategories:
            best = assignedCategories[word]['best']
            cat, depth, sim = best
            if cat and sim > 0:
                # Weight by similarity and word count in dictionary
                weight = sim * (dictionary.get(word, 1) ** 0.25)
                cat_scores[cat] += weight
                # Also add parent category score
                if cat in parent_categories:
                    cat_scores[parent_categories[cat]] += weight * 0.5

    if cat_scores:
        best_cat = cat_scores.most_common(1)[0][0]
        # Get tier1 (if best_cat is tier2, get its parent)
        if best_cat in parent_categories:
            pred_tier1 = parent_categories[best_cat]
            pred_tier2 = best_cat
        else:
            pred_tier1 = best_cat
            pred_tier2 = best_cat
        doc_predicted[doc_id] = (pred_tier2, pred_tier1)

print(f"  Documents classified: {len(doc_predicted):,} / {len(arr_url):,}")

# Evaluate document-level accuracy
tier1_correct = 0
tier2_correct = 0
total_docs = 0

tier1_confusion = defaultdict(Counter)

for doc_id in doc_predicted:
    if doc_id in doc_categories:
        total_docs += 1
        actual_tier2, actual_tier1 = doc_categories[doc_id]
        pred_tier2, pred_tier1 = doc_predicted[doc_id]

        if pred_tier1 == actual_tier1:
            tier1_correct += 1
        if pred_tier2 == actual_tier2:
            tier2_correct += 1

        tier1_confusion[actual_tier1][pred_tier1] += 1

print(f"\n  Document-level accuracy:")
print(f"    Tier 1 (vertical): {tier1_correct:,} / {total_docs:,} ({100*tier1_correct/max(total_docs,1):.1f}%)")
print(f"    Tier 2 (sub-cat):  {tier2_correct:,} / {total_docs:,} ({100*tier2_correct/max(total_docs,1):.1f}%)")

# Per-vertical accuracy
print(f"\n  Per-vertical Tier 1 accuracy:")
for actual_tier1 in sorted(tier1_confusion.keys()):
    preds = tier1_confusion[actual_tier1]
    total_for_cat = sum(preds.values())
    correct = preds.get(actual_tier1, 0)
    pct = 100 * correct / max(total_for_cat, 1)
    top_mistake = ""
    for pred, cnt in preds.most_common(2):
        if pred != actual_tier1:
            top_mistake = f"  (top confusion: {pred} [{cnt}])"
            break
    print(f"    {actual_tier1:35s} {correct:4d}/{total_for_cat:4d} ({pct:5.1f}%){top_mistake}")


#--- [6] Save results

print("\n" + "=" * 60)
print("STEP 6: Saving results")
print("=" * 60)

with open(os.path.join(DATA_DIR, "iab_assignedCategories.txt"), "w") as f:
    for word in assignedCategories:
        best = assignedCategories[word]['best']
        f.write(f"{word}\t{best}\n")

with open(os.path.join(DATA_DIR, "iab_doc_predictions.txt"), "w") as f:
    f.write("doc_id\tpred_tier2\tpred_tier1\tactual_tier2\tactual_tier1\tcorrect_tier1\n")
    for doc_id in sorted(doc_predicted.keys()):
        if doc_id in doc_categories:
            pred_tier2, pred_tier1 = doc_predicted[doc_id]
            actual_tier2, actual_tier1 = doc_categories[doc_id]
            correct = "YES" if pred_tier1 == actual_tier1 else "NO"
            f.write(f"{doc_id}\t{pred_tier2}\t{pred_tier1}\t{actual_tier2}\t{actual_tier1}\t{correct}\n")

print("  Saved iab_assignedCategories.txt")
print("  Saved iab_doc_predictions.txt")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"""
Summary:
  Dataset:              20 Newsgroups -> IAB verticals
  Documents:            {len(arr_url):,}
  Dictionary:           {len(dictionary):,} words
  IAB Categories:       {len(categories)} (Tier 1 + Tier 2)
  Tier 1 accuracy:      {100*tier1_correct/max(total_docs,1):.1f}%
  Tier 2 accuracy:      {100*tier2_correct/max(total_docs,1):.1f}%

To improve results, try:
  - Adjust threshold (line ~170): lower = more topWords = finer taxonomy
  - Adjust depth weight (line ~230): higher = more specific categories
  - Add domain-specific stopwords relevant to your advertiser data
  - Replace 20 Newsgroups with your own advertiser content/URLs
""")
