import re
from nltk.stem import PorterStemmer
from pathlib import Path

stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"

def load_stopwords():
    stopwords_path = "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/stopwords.txt"
    try:
        with open(stopwords_path, "r") as f:
            data = f.read()
            words = data.splitlines()
            return words
    except Exception as e:
        print(f"Error loading stopwords: {str(e)}")



stemmer = PorterStemmer()

def tokenise_string(text: str) -> list[str]:
    """
    Tokenize and normalize text for BM25.
    Handles markdown, medical symbols, and punctuation.
    """
    text = text.lower()
    
    text = re.sub(r'\*+', ' ', text)
    tokens = re.findall(r'\b\w+\b', text)
    
    medical_terms = [
        'who',  # World Health Organisation
        'fda',  # Food and Drug Administration
        'ema',  # European Medicines Agency
        'hiv',  # Human Immunodeficiency Virus
        'aids', # Acquired Immune Deficiency Syndrome
        'copd', # Chronic Obstructive Pulmonary Disease
        'adhd', # Attention Deficit Hyperactivity Disorder
        'iv',   # Intravenous
        'im',   # Intramuscular
        'po',   # Per os (oral)
    ]
    
    stopwords = load_stopwords() - medical_terms
    
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

def normalise_score(score, min_scores, max_scores):
    if min_scores == max_scores:
        return 1.0
    return (score - min_scores) / (max_scores - min_scores)

def fix_encoding_errors(text: str) -> str:
    """
    Fix common encoding errors in pharmaceutical PDFs.
    Handles double-encoding, malformed UTF-8, and Unicode escapes.
    """
    
    # Step 1: Fix double-encoded UTF-8 (UTF-8 bytes decoded as Latin-1)
    try:
        # Try to re-encode as Latin-1 and decode as UTF-8
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Can't fix this way, continue with other methods
        pass
    
    # Step 2: Manual replacements for common double-encoded sequences
    # These are UTF-8 bytes incorrectly interpreted as Latin-1
    double_encoded_fixes = {
        # Quotation marks
        '\u00e2\u0080\u009c': '"',  # Left double quote "
        '\u00e2\u0080\u009d': '"',  # Right double quote "
        '\u00e2\u0080\u0098': ''',  # Left single quote '
        '\u00e2\u0080\u0099': ''',  # Right single quote '
        '\u00e2\u0080\u00b2': '′',  # Prime symbol
        '\u00e2\u0080\u00b3': '″',  # Double prime
        
        # Dashes and hyphens
        '\u00e2\u0080\u0093': '–',  # En dash
        '\u00e2\u0080\u0094': '—',  # Em dash
        '\u00e2\u0080\u0090': '-',  # Hyphen
        '\u00e2\u0080\u0091': '-',  # Non-breaking hyphen
        
        # Spaces and breaks
        '\u00e2\u0080\u00a6': '…',  # Ellipsis
        '\u00c2\u00a0': ' ',        # Non-breaking space
        '\u00e2\u0080\u00af': ' ',  # Narrow non-breaking space
        
        # Mathematical symbols
        '\u00c2\u00b1': '±',              # Plus-minus
        '\u00c3\u0097': '×',              # Multiplication
        '\u00c3\u00b7': '÷',              # Division
        '\u00e2\u0089\u00a5': '≥',        # Greater than or equal
        '\u00e2\u0089\u00a4': '≤',        # Less than or equal
        '\u00e2\u0089\u0088': '≈',        # Approximately equal
        '\u00e2\u0089\u00a0': '≠',        # Not equal
        '\u00e2\u0088\u009e': '∞',        # Infinity
        '\u00e2\u0088\u009a': '√',        # Square root
        '\u00e2\u0086\u0092': '→',        # Right arrow
        '\u00e2\u0086\u0090': '←',        # Left arrow
        
        # Degree and other symbols
        '\u00c2\u00b0': '°',              # Degree symbol
        '\u00c2\u00b5': 'μ',              # Micro (mu)
        '\u00c2\u00b2': '²',              # Superscript 2
        '\u00c2\u00b3': '³',              # Superscript 3
        '\u00c2\u00bc': '¼',              # One quarter
        '\u00c2\u00bd': '½',              # One half
        '\u00c2\u00be': '¾',              # Three quarters
        
        # Greek letters (common in medical docs)
        '\u00ce\u00b1': 'α',              # Alpha
        '\u00ce\u00b2': 'β',              # Beta
        '\u00ce\u00b3': 'γ',              # Gamma
        '\u00ce\u0094': 'Δ',              # Delta
        
        # Bullets and list markers
        '\u00e2\u0080\u00a2': '•',        # Bullet
        '\u00e2\u0097\u008f': '◦',        # White bullet
        '\u00e2\u0096\u00aa': '▪',        # Black small square
        '\u00c2\u00b7': '·',              # Middle dot
    }
    
    for wrong, correct in double_encoded_fixes.items():
        text = text.replace(wrong, correct)
    
    # Step 3: Decode Unicode escape sequences (\u03bc style)
    try:
        # Handle double-escaped sequences first
        if '\\\\u' in text:
            text = text.replace('\\\\u', '\\u')
        
        # Decode \uXXXX sequences
        text = text.encode().decode('unicode_escape')
        
    except (UnicodeDecodeError, AttributeError):
        # Already decoded or contains invalid sequences
        pass
    
    # Step 4: Fix Windows-1252 (CP-1252) common errors
    # These occur when Windows encoding is read as UTF-8
    cp1252_fixes = {
        '\x80': '€',  # Euro sign
        '\x82': '‚',  # Single low-9 quotation mark
        '\x83': 'ƒ',  # Latin small letter f with hook
        '\x84': '„',  # Double low-9 quotation mark
        '\x85': '…',  # Horizontal ellipsis
        '\x86': '†',  # Dagger
        '\x87': '‡',  # Double dagger
        '\x88': 'ˆ',  # Modifier letter circumflex accent
        '\x89': '‰',  # Per mille sign
        '\x8a': 'Š',  # Latin capital letter S with caron
        '\x8b': '‹',  # Single left-pointing angle quotation mark
        '\x8c': 'Œ',  # Latin capital ligature OE
        '\x91': ''',  # Left single quotation mark
        '\x92': ''',  # Right single quotation mark
        '\x93': '"',  # Left double quotation mark
        '\x94': '"',  # Right double quotation mark
        '\x95': '•',  # Bullet
        '\x96': '–',  # En dash
        '\x97': '—',  # Em dash
        '\x98': '˜',  # Small tilde
        '\x99': '™',  # Trade mark sign
        '\x9a': 'š',  # Latin small letter s with caron
        '\x9b': '›',  # Single right-pointing angle quotation mark
        '\x9c': 'œ',  # Latin small ligature oe
        '\x9f': 'Ÿ',  # Latin capital letter Y with diaeresis
    }
    
    for wrong, correct in cp1252_fixes.items():
        text = text.replace(wrong, correct)
    
    # Step 5: Remove null bytes and other control characters
    # Keep only: tab, newline, carriage return
    text = ''.join(char for char in text 
                   if char >= ' ' or char in '\t\n\r')
    
    # Step 6: Normalize whitespace (collapse multiple spaces/newlines)
    # But preserve single newlines for structure
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines → 2 newlines
    
    return text

