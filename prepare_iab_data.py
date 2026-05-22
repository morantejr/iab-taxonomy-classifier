"""
prepare_iab_data.py
Builds xllm6-compatible data tables from:
  1. IAB Content Taxonomy (Tier 1 + Tier 2 categories, hardcoded from IAB standard)
  2. 20 Newsgroups dataset (11,314 labeled text documents from HuggingFace)

This maps the 20 Newsgroups categories to IAB verticals, creating a realistic
advertiser-vertical classification scenario.

Output files (all in iab_data/ folder):
  iab_dictionary.txt        - words with counts (like xllm6_dictionary.txt)
  iab_arr_url.txt           - maps document IDs to source identifiers
  iab_hash_category.txt     - actual category assignments per word
  iab_url_map.txt           - document IDs attached to words
  iab_taxonomy.txt          - IAB taxonomy: category | parent | depth
  iab_stopwords.txt         - stopwords list
"""
import os
import re
from collections import Counter, defaultdict
from datasets import load_dataset

OUTPUT_DIR = "iab_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#--- [1] IAB Content Taxonomy (Tier 1 + Tier 2)
# Source: IAB Tech Lab Content Taxonomy 2.2 (industry standard for ad categorization)
# Tier 1 = top-level verticals, Tier 2 = subcategories

IAB_TAXONOMY = {
    "Automotive": [
        "Auto Body Styles", "Auto Type", "Budget Cars", "Certified Pre-Owned",
        "Car Culture", "Dash Cam Videos", "Electric Vehicles",
        "Luxury Cars", "Motorcycles", "Off-Road Vehicles",
        "Performance Cars", "Scooters", "SUVs", "Trucks", "Vintage Cars"
    ],
    "Business": [
        "Advertising", "Agriculture", "Biotech", "Business Software",
        "Construction", "Forestry", "Government", "Green Solutions",
        "Human Resources", "Logistics", "Marketing", "Metals"
    ],
    "Careers": [
        "Career Advice", "Career Planning", "College", "Job Fairs",
        "Job Search", "Nursing", "Resume Writing", "Scholarships",
        "Telecommuting", "US Military", "Vocational Training"
    ],
    "Education": [
        "Art History", "College Administration", "Distance Learning",
        "English as 2nd Language", "Graduate School", "Homeschooling",
        "Homework Study Tips", "Language Learning", "Online Education",
        "Private School", "Special Education", "Studying Business"
    ],
    "Electronics": [
        "Cameras", "Cell Phones", "Computer Peripherals", "Computer Reviews",
        "Desktops", "Email", "Entertainment", "GPS", "Home Video",
        "Internet", "Laptops", "Monitors", "Netbooks", "Portable",
        "Satellites", "Scanners", "Smartphones", "Tablets"
    ],
    "Entertainment": [
        "Celebrity Gossip", "Comics", "Culture", "Fine Art",
        "Humor", "Movies", "Music", "Television", "Video Games"
    ],
    "Family & Parenting": [
        "Adoption", "Babies & Toddlers", "Children", "Daycare",
        "Eldercare", "Family Internet", "Parenting Teens",
        "Parenting K-6 Kids", "Pregnancy", "Special Needs Kids"
    ],
    "Finance": [
        "Banking", "Beginning Investing", "Credit Cards", "Financial News",
        "Financial Planning", "Hedge Fund", "Insurance", "Investing",
        "Mortgage", "Mutual Funds", "Options", "Retirement Planning",
        "Stocks", "Tax Planning"
    ],
    "Food & Drink": [
        "American Cuisine", "Barbecues", "Cajun", "Chinese Cuisine",
        "Cocktails", "Coffee", "Cooking", "Desserts",
        "Dining Out", "Food Allergies", "French Cuisine",
        "Health Cooking", "Italian Cuisine", "Japanese Cuisine",
        "Mexican Cuisine", "Vegan", "Vegetarian", "Wine"
    ],
    "Health & Fitness": [
        "ADD", "AIDS HIV", "Allergies", "Alternative Medicine",
        "Arthritis", "Asthma", "Autism", "Bipolar Disorder",
        "Brain Tumor", "Cancer", "Chronic Fatigue", "Chronic Pain",
        "Dental Care", "Depression", "Dermatology", "Diabetes",
        "Epilepsy", "Exercise", "Heart Disease", "Incontinence",
        "Infertility", "Mens Health", "Nutrition", "Orthopedics",
        "Panic Anxiety", "Pediatrics", "Physical Therapy",
        "Psychology", "Senior Health", "Sexuality", "Sleep Disorders",
        "Smoking Cessation", "Substance Abuse", "Thyroid Disease",
        "Weight Loss", "Womens Health"
    ],
    "Hobbies & Interests": [
        "Art Technology", "Arts Crafts", "Beadwork", "Bird Watching",
        "Board Games", "Candle Making", "Card Games", "Chess",
        "Cigars", "Collecting", "Comic Books", "Drawing Sketching",
        "Freelance Writing", "Genealogy", "Getting Published", "Guitar",
        "Home Recording", "Inventors", "Jewelry Making", "Magic",
        "Model Trains", "Needlework", "Painting", "Photography",
        "Radio", "Roleplaying Games", "Sci-Fi Fantasy", "Screenwriting",
        "Stamps Coins", "Video Computer Games", "Woodworking"
    ],
    "Home & Garden": [
        "Appliances", "Entertaining", "Environmental Safety",
        "Gardening", "Home Repair", "Home Theater",
        "Interior Decorating", "Landscaping", "Remodeling"
    ],
    "Law, Gov & Politics": [
        "Commentary", "Immigration", "Legal Issues", "Politics",
        "US Government Resources", "US Politics"
    ],
    "News": [
        "International News", "Local News", "National News",
        "Technology News", "Weather"
    ],
    "Personal Finance": [
        "Beginning Investing", "Credit Debt Management",
        "Financial News", "Financial Planning", "Insurance",
        "Investing", "Mutual Funds", "Retirement Planning",
        "Stocks", "Tax Planning"
    ],
    "Pets": [
        "Aquariums", "Birds", "Cats", "Dogs", "Large Animals",
        "Reptiles", "Veterinary Medicine"
    ],
    "Real Estate": [
        "Apartments", "Architects", "Buying Selling Homes"
    ],
    "Religion & Spirituality": [
        "Alternative Religions", "Atheism Agnosticism", "Buddhism",
        "Catholicism", "Christianity", "Hinduism", "Islam",
        "Judaism", "Latter Day Saints", "Pagan Wiccan"
    ],
    "Science": [
        "Astrology", "Biology", "Botany", "Chemistry", "Geography",
        "Geology", "Paranormal", "Physics", "Space Astronomy",
        "Weather"
    ],
    "Shopping": [
        "Comparison", "Coupons", "Engines", "Gifts"
    ],
    "Society": [
        "Dating", "Divorce Support", "Ethnic Specific", "Gay Life",
        "Marriage", "Senior Living", "Teens", "Weddings"
    ],
    "Sports": [
        "Auto Racing", "Baseball", "Bicycling", "Bodybuilding",
        "Boxing", "Canoeing Kayaking", "Cheerleading", "Climbing",
        "Cricket", "Figure Skating", "Fly Fishing", "Football",
        "Freshwater Fishing", "Game Fish", "Golf", "Horse Racing",
        "Horses", "Hunting Shooting", "Inline Skating", "Martial Arts",
        "Mountain Biking", "NASCAR Racing", "Olympics", "Paintball",
        "Power Motorcycles", "Pro Basketball", "Pro Ice Hockey",
        "Rodeo", "Rugby", "Running Jogging", "Sailing", "Saltwater Fishing",
        "Scuba Diving", "Skateboarding", "Skiing", "Snowboarding",
        "Surfing Bodyboarding", "Swimming", "Table Tennis", "Tennis",
        "Volleyball", "Walking", "Waterski Wakeboard", "World Soccer",
        "Wrestling"
    ],
    "Style & Fashion": [
        "Accessories", "Beauty", "Body Art", "Clothing",
        "Fashion", "Jewelry"
    ],
    "Technology & Computing": [
        "3-D Graphics", "Animation", "Antivirus Software", "C C++",
        "Cameras Camcorders", "Cell Phones", "Computer Certification",
        "Computer Networking", "Computer Peripherals", "Computer Reviews",
        "Data Centers", "Databases", "Desktop Publishing", "Desktop Video",
        "Email", "Graphics Software", "Home Video", "Internet Technology",
        "Java", "JavaScript", "Linux", "Mac OS", "Mac Support",
        "MP3 MIDI", "Net Conferencing", "Net for Beginners",
        "Network Security", "Palmtops PDAs", "PC Support",
        "Portable", "Shareware Freeware", "Unix", "Visual Basic",
        "Web Clip Art", "Web Design", "Web Search", "Windows"
    ],
    "Travel": [
        "Adventure Travel", "Africa", "Air Travel", "Australia New Zealand",
        "Bed Breakfasts", "Budget Travel", "Business Travel",
        "Camping", "Canada", "Caribbean", "Cruises",
        "Eastern Europe", "Europe", "France", "Greece",
        "Honeymoons Getaways", "Hotels", "Italy", "Japan",
        "Mexico Central America", "National Parks", "South America",
        "Spas", "Theme Parks", "Traveling with Kids",
        "United Kingdom", "United States"
    ],
}


#--- [2] Map 20 Newsgroups -> IAB verticals
# This simulates an advertiser's content being classified into IAB verticals

NEWSGROUPS_TO_IAB = {
    "comp.graphics":             ("Technology & Computing", "Graphics Software"),
    "comp.os.ms-windows.misc":   ("Technology & Computing", "Windows"),
    "comp.sys.ibm.pc.hardware":  ("Technology & Computing", "PC Support"),
    "comp.sys.mac.hardware":     ("Technology & Computing", "Mac Support"),
    "comp.windows.x":            ("Technology & Computing", "Unix"),
    "misc.forsale":              ("Shopping", "Comparison"),
    "rec.autos":                 ("Automotive", "Car Culture"),
    "rec.motorcycles":           ("Automotive", "Motorcycles"),
    "rec.sport.baseball":        ("Sports", "Baseball"),
    "rec.sport.hockey":          ("Sports", "Pro Ice Hockey"),
    "sci.crypt":                 ("Technology & Computing", "Network Security"),
    "sci.electronics":           ("Electronics", "Computer Peripherals"),
    "sci.med":                   ("Health & Fitness", "Mens Health"),
    "sci.space":                 ("Science", "Space Astronomy"),
    "soc.religion.christian":    ("Religion & Spirituality", "Christianity"),
    "talk.politics.guns":        ("Law, Gov & Politics", "Politics"),
    "talk.politics.mideast":     ("Law, Gov & Politics", "Politics"),
    "talk.politics.misc":        ("Law, Gov & Politics", "Commentary"),
    "talk.religion.misc":        ("Religion & Spirituality", "Alternative Religions"),
    "alt.atheism":               ("Religion & Spirituality", "Atheism Agnosticism"),
}


#--- [3] Build stopwords list

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "their",
    "his", "her", "him", "who", "which", "what", "where", "when", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "about", "above", "after", "again", "also", "any",
    "because", "before", "between", "into", "through", "during", "out",
    "up", "down", "over", "under", "then", "once", "here", "there",
    "if", "while", "re", "ve", "ll", "don", "didn", "doesn", "isn",
    "wasn", "weren", "won", "wouldn", "shouldn", "couldn", "hadn",
    "hasn", "haven", "aren", "ain", "gt", "lt", "le", "de", "la",
    "subject", "lines", "organization", "writes", "article", "wrote",
    "posting", "host", "nntp", "reply", "distribution", "keywords",
    "university", "com", "edu", "cs", "ca", "ac", "uk", "us",
    "one", "two", "three", "four", "five", "new", "get", "like",
    "know", "think", "make", "say", "go", "see", "come", "take",
    "use", "find", "give", "tell", "work", "call", "try", "ask",
    "seem", "feel", "leave", "keep", "let", "begin", "show", "hear",
    "play", "run", "move", "live", "believe", "bring", "happen",
    "must", "well", "back", "even", "still", "way", "much", "many",
    "since", "long", "great", "right", "old", "big", "high", "small",
    "large", "next", "early", "young", "important", "last", "public",
    "good", "own", "first", "second", "third", "without", "however",
    "people", "year", "time", "day", "thing", "man", "world", "life",
    "hand", "part", "place", "case", "point", "group", "number",
    "fact", "state", "area", "lot", "set", "end", "head",
}


#--- [4] Text cleaning

def clean_text(text):
    """Clean newsgroup posting: remove headers, emails, signatures."""
    # Remove email headers (lines before first blank line are often headers)
    lines = text.split('\n')
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '' and i > 0:
            body_start = i + 1
            break

    text = '\n'.join(lines[body_start:])
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove signature blocks
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def tokenize(text, stopwords):
    """Extract single tokens and bigrams, filtering stopwords."""
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in stopwords]

    tokens = []
    # Single tokens
    for w in words:
        tokens.append(w)
    # Bigrams (multi-token words, using ~ separator like xllm6)
    for i in range(len(words) - 1):
        bigram = words[i] + "~" + words[i+1]
        tokens.append(bigram)

    return tokens


#--- [5] Main: build all tables

print("Loading 20 Newsgroups dataset from HuggingFace...")
ds = load_dataset('SetFit/20_newsgroups', split='train')
print(f"  Loaded {len(ds)} documents")

# Build tables
dictionary = Counter()       # word -> count across all docs
url_map = defaultdict(dict)  # word -> {doc_id: count}
arr_url = []                 # doc_id -> identifier string
hash_category = defaultdict(dict)  # word -> {category_string: count}
doc_categories = {}          # doc_id -> (tier2, tier1)

print("Processing documents...")
for doc_id, row in enumerate(ds):
    newsgroup = row['label_text']
    text = clean_text(row['text'])

    if newsgroup not in NEWSGROUPS_TO_IAB:
        continue

    tier1, tier2 = NEWSGROUPS_TO_IAB[newsgroup]
    depth_tier1 = 1
    depth_tier2 = 2

    # Build document identifier
    doc_name = f"doc_{doc_id}_{newsgroup}"
    arr_url.append(doc_name)
    actual_doc_id = len(arr_url) - 1
    doc_categories[actual_doc_id] = (tier2, tier1)

    # Tokenize
    tokens = tokenize(text, STOPWORDS)

    # Update dictionary and url_map
    seen_tokens = set()
    for token in tokens:
        dictionary[token] += 1
        if token not in seen_tokens:
            if str(actual_doc_id) not in url_map[token]:
                url_map[token][str(actual_doc_id)] = 1
            else:
                url_map[token][str(actual_doc_id)] += 1
            seen_tokens.add(token)

    # Update hash_category: assign the IAB category to each token in this doc
    category_str = f"{tier2} | {tier1}  | {depth_tier2}"
    for token in set(tokens):
        if category_str in hash_category[token]:
            hash_category[token][category_str] += 1
        else:
            hash_category[token][category_str] = 1

    if doc_id % 2000 == 0:
        print(f"  Processed {doc_id}/{len(ds)} docs...")

print(f"\nDone processing.")
print(f"  Documents: {len(arr_url)}")
print(f"  Dictionary words: {len(dictionary)}")

# Filter dictionary: keep words with count >= 3
min_count = 3
dictionary = {w: c for w, c in dictionary.items() if c >= min_count}
print(f"  Dictionary after filtering (count >= {min_count}): {len(dictionary)}")

# Filter url_map and hash_category to match filtered dictionary
url_map = {w: v for w, v in url_map.items() if w in dictionary}
hash_category = {w: v for w, v in hash_category.items() if w in dictionary}


#--- [6] Save all tables

print("\nSaving tables...")

# dictionary
with open(os.path.join(OUTPUT_DIR, "iab_dictionary.txt"), "w") as f:
    for word, count in sorted(dictionary.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{word}\t{count}\n")

# arr_url
with open(os.path.join(OUTPUT_DIR, "iab_arr_url.txt"), "w") as f:
    for url in arr_url:
        f.write(f"{url}\n")

# hash_category
with open(os.path.join(OUTPUT_DIR, "iab_hash_category.txt"), "w") as f:
    for word in hash_category:
        cats = hash_category[word]
        cats_str = "\t".join([f"{k}\t{v}" for k, v in cats.items()])
        f.write(f"{word}\t{cats_str}\n")

# url_map
with open(os.path.join(OUTPUT_DIR, "iab_url_map.txt"), "w") as f:
    for word in url_map:
        ids = url_map[word]
        ids_str = "\t".join([f"{k}\t{v}" for k, v in ids.items()])
        f.write(f"{word}\t{ids_str}\n")

# taxonomy (IAB hierarchy)
with open(os.path.join(OUTPUT_DIR, "iab_taxonomy.txt"), "w") as f:
    for tier1, tier2_list in IAB_TAXONOMY.items():
        f.write(f"{tier1}\t|\t{tier1}\t|\t1\n")
        for tier2 in tier2_list:
            f.write(f"{tier2}\t|\t{tier1}\t|\t2\n")

# stopwords
with open(os.path.join(OUTPUT_DIR, "iab_stopwords.txt"), "w") as f:
    for word in sorted(STOPWORDS):
        f.write(f"{word}\n")

# doc_categories (ground truth for evaluation)
with open(os.path.join(OUTPUT_DIR, "iab_doc_categories.txt"), "w") as f:
    for doc_id in sorted(doc_categories.keys()):
        tier2, tier1 = doc_categories[doc_id]
        f.write(f"{doc_id}\t{tier2}\t{tier1}\n")


#--- [7] Summary statistics

print("\nSaved files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname:40s} {size:>10,} bytes")

# Show IAB category distribution
cat_counts = Counter()
for doc_id, (tier2, tier1) in doc_categories.items():
    cat_counts[tier1] += 1

print(f"\nIAB Tier 1 distribution across {len(arr_url)} documents:")
for cat, count in cat_counts.most_common():
    bar = "#" * (count // 20)
    print(f"  {cat:30s} {count:5d} {bar}")

# Show top dictionary words
print(f"\nTop 20 dictionary words:")
sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_dict[:20]:
    print(f"  {count:6d}  {word}")

print("\nData preparation complete! Run: python taxonomy_iab.py")
