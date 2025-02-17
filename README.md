# PCS Annotator: A Python Library for Classification Tasks Using Multiple LLMs

## How to Install

```bash
pip install pcs-annotator
```

## How to Use

```python
from pcs_annotator import PCS
```

### Create a Prompt for Classification Tasks

```python
prompt = """Analyze the news article and determine whether it is 'Fake' or 'Real.'
Label the article using the tag format: <label>Fake</label> or <label>Real</label>.
Afterward, provide your reasoning after the “<reasoning>” tag and close it with "</reasoning>".
Respond only with the label and the reasoning.
"""
```

You will need a CSV file with two columns: `label` and `text` to ensure better annotation, as the data is used to train the LLMs.

### Initialize the PCS Class

```python
pcs = PCS(prompt, dataset_path="gossipcop.csv")
```

### Annotate a New Text

```python
print(pcs.annotate("America is a country"))

