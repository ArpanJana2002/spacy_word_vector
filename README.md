#  Word Vectors with spaCy

A beginner-friendly guide to using word vectors and NLP similarity features with spaCy.

---

##  Requirements

- Python 3.8+
- spaCy 3.x
- A spaCy model with word vectors (`md` or `lg`)

---

##  Installation

```bash
pip install spacy
```

Download a model with word vectors:

```bash
# Medium model (~50k vectors) — recommended
python -m spacy download en_core_web_md

# Large model (~685k vectors) — more accurate, larger download
python -m spacy download en_core_web_lg
```

>  **Note:** The small model (`en_core_web_sm`) does **not** include word vectors.

---

##  Quick Start

```python
import spacy

nlp = spacy.load("en_core_web_md")

doc = nlp("apple banana orange")
for token in doc:
    print(token.text, token.vector.shape, token.has_vector)
```

---

##  Features & Usage

### 1. Word Similarity

```python
word1 = nlp("cat")[0]
word2 = nlp("dog")[0]
print(word1.similarity(word2))  # e.g. 0.80
```

### 2. Sentence Similarity

```python
doc1 = nlp("I love cats")
doc2 = nlp("I love dogs")
print(doc1.similarity(doc2))  # e.g. 0.93
```

### 3. Accessing Vectors Directly

```python
token = nlp("coffee")[0]
print(token.vector)        # numpy array (300,)
print(token.vector_norm)   # L2 norm
print(token.has_vector)    # True/False
```

### 4. Vector Arithmetic

```python
import numpy as np

king  = nlp.vocab["king"].vector
man   = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector

result = king - man + woman  # ≈ queen
```

### 5. Finding Most Similar Words

```python
import numpy as np

def most_similar(word, nlp, n=5):
    query_vec = nlp.vocab[word].vector
    scores = {}
    for w in nlp.vocab:
        if w.has_vector and w.is_lower and w.is_alpha:
            scores[w.text] = np.dot(query_vec, w.vector) / (
                w.vector_norm * nlp.vocab[word].vector_norm + 1e-8
            )
    return sorted(scores, key=scores.get, reverse=True)[:n]

print(most_similar("king", nlp))
```

---

##  Recommended Platforms

| Platform | Internet | Free | Notes |
|----------|----------|------|-------|
| [Google Colab](https://colab.research.google.com) | ✅ | ✅ | Best option, runs in browser |
| [Kaggle Notebooks](https://www.kaggle.com) | ✅ | ✅ | Free GPU available |
| Local Jupyter | Depends | ✅ | Needs internet to download models |

---

##  Troubleshooting

**SyntaxError on `python -m spacy download`**
> In Jupyter/Colab, prefix shell commands with `!`:
> ```python
> !python -m spacy download en_core_web_md
> ```

**OSError: Can't find model**
> The model isn't downloaded yet. Run:
> ```python
> !python -m spacy download en_core_web_md
> ```

**Network/Connection Error during download**
> Your environment may not have internet access. Options:
> - Switch to Google Colab
> - Manually download the `.whl` file from [spaCy model releases](https://github.com/explosion/spacy-models/releases) and install with `pip install model.whl`

---

## Model Comparison

| Model | Vectors | Size | Best For |
|-------|---------|------|----------|
| `en_core_web_sm` | None | ~12 MB | Basic NLP, no similarity |
| `en_core_web_md` | ~50k | ~40 MB | General use, word similarity |
| `en_core_web_lg` | ~685k | ~560 MB | High accuracy similarity tasks |

---

##  Resources

- [spaCy Official Docs](https://spacy.io/usage/linguistic-features#vectors-similarity)
- [spaCy Model Releases](https://github.com/explosion/spacy-models/releases)
- [spaCy API Reference](https://spacy.io/api)

---
