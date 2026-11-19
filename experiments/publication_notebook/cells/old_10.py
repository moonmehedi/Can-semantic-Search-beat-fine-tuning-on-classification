class TextNormalizer:
    def __init__(self):
        # Initialize text normalization dictionary
        self.slang_dict = {
            'imo': 'in my opinion',
            'imho': 'in my humble opinion',
            'tbh': 'to be honest',
            'afaik': 'as far as i know',
            'idk': "i don't know",
            'smh': 'shaking my head',
            'btw': 'by the way',
            'asap': 'as soon as possible',
            'fyi': 'for your information',
            'np': 'no problem',
            'thx': 'thanks',
            'pls': 'please',
            'u': 'you',
            'r': 'are',
            'ur': 'your',
            'n': 'and',
            'b/c': 'because',
            'cz' : 'because',
            'bcz' : 'because',
            'ofc' : 'ofcourse',
            'govt': 'government',
            'dept': 'department',
            'est': 'established',
            'approx': 'approximately',
            'min': 'minimum',
            'max': 'maximum',
            "n't": 'not',
            "nt" : 'not',
            "gn" : 'good night',
            "f9" : 'fine',
            'bpd': 'borderline personality disorder',
            'dbt': 'dialectical behavior therapy',
            'fp': 'favorite person',
            'si': 'suicidal ideation',
            'sh': 'self harm',
            "its" : 'it is',
            "people's" : 'people',
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had / she would",
            "she'd've": "she would have",
            "she'll": "she shall / she will",
            "she'll've": "she shall have / she will have",
            "she's": "she has / she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as / so is",
            "that'd": "that would / that had",
            "that'd've": "that would have",
            "that's": "that has / that is",
            "there'd": "there had / there would",
            "there'd've": "there would have",
            "there's": "there has / there is",
            "they'd": "they had / they would",
            "they'd've": "they would have",
            "they'll": "they shall / they will",
            "they'll've": "they shall have / they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had / we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what shall / what will",
            "what'll've": "what shall have / what will have",
            "what're": "what are",
            "what's": "what has / what is",
            "what've": "what have",
            "when's": "when has / when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where has / where is",
            "where've": "where have",
            "who'll": "who shall / who will",
            "who'll've": "who shall have / who will have",
            "who's": "who has / who is",
            "who've": "who have",
            "why's": "why has / why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had / you would",
            "you'd've": "you would have",
            "you'll": "you shall / you will",
            "you'll've": "you shall have / you will have",
            "you're": "you are",
            "you've": "you have"
}

    def clean_text(self, text):
        """Basic text cleaning"""
        text = str(text).lower().strip()
        # Remove special characters (keep basic punctuation)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
        return text

    def expand_slang(self, text):
        """Replace slang terms using dictionary"""
        words = text.split()
        expanded_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(expanded_words)

    def normalize_text(self, text):
        """Full text normalization pipeline"""
        if not isinstance(text, str) or not text.strip():
            return text
            
        # Clean text first
        text = self.clean_text(text)
        # Fix contractions
        text = contractions.fix(text)
        # Expand slang
        text = self.expand_slang(text)
        return text


# ---- same text cleaning as the earlier baseline (lower-case, special characters, contractions, slang) ----
normalizer = TextNormalizer()
def _normalise_split(ds):
    df = ds.to_pandas()
    for col in ("question", "answer"):
        df[col] = df[col].apply(normalizer.normalize_text)
    return Dataset.from_pandas(df, preserve_index=False)

CLEAN = DatasetDict({s: _normalise_split(RAW[s]) for s in ("train", "validation", "test")})
TEST_Q_NORM, TEST_A_NORM = list(CLEAN["test"]["question"]), list(CLEAN["test"]["answer"])
print("Example (normalised):", CLEAN["train"][0]["answer"][:120])