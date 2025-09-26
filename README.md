# RaPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery

This is the official implementation of our ACL 2025 Findings paper "RaPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery". 

## Overview

RaPID is an efficient framework for generating knowledge-intensive and comprehensive long texts, such as wiki-style articles. RaPID consists of three main modules:

1. **Retrieval-augmented Preliminary Outline Generation**
   - Reduces hallucinations by grounding the generation in retrieved facts
   - Ensures factual accuracy in the generated outline

2. **Attribute-constrained Search**
   - Enables efficient information discovery
   - Optimizes the retrieval process for relevant information

3. **Plan-guided Article Generation**
   - Enhances thematic coherence throughout the article
   - Maintains structural consistency in long-form content

## Installation

``` bash
conda create -n rapid python=3.10
conda activate rapid
pip install -r requirements.txt
```

## Configuration

RaPID requires API keys for both LLM and search engine services. These should be configured in a `secrets.toml` file:

1. Create a `secrets.toml` file in the project root directory:
```bash
touch secrets.toml
```

2. Add your keys to `secrets.toml`:
```toml
# LLM API Configuration
OPENAI_API_KEY = "your-llm-api-key"

# Search Engine API Configuration
GOOGLE_API_KEY = "your-search-api-key"
GOOGLE_CX = "your-search-engine-id"  
```

3. Make sure to add `secrets.toml` to your `.gitignore` file to prevent accidentally committing sensitive information:
```bash
echo "secrets.toml" >> .gitignore
```

## Data

The data files can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1GNWE0ZEPijFpdjuPOfWLjWwEcQYq3Kz2). After downloading, please follow these steps to set up the data directory:

1. Create the data directory structure:
```bash
mkdir -p wiki_dump/encode wiki_dump/original
```

2. Download and extract the files from Google Drive to the appropriate directories. The final directory structure should look like this:

- wiki_dump/
  - encode/
    - merged_encoded_vectors.pkl
  - original/
    - combined.jsonl
  - titles.csv


## Usage

1. Console example:
```bash
python example.py --retriever google \
        --output-dir ./results \
        --max-thread-num 3 \
        --do-clarify \
        --do-research \
        --do-generate-outline \
        --do-generate-article \
        --do-topo-generation \
        --do-polish-article \
        --interface console
```

2. Batch example:
```bash
python example.py --retriever google \
        --output-dir ./results \
        --max-thread-num 3 \
        --do-clarify \
        --do-research \
        --do-generate-outline \
        --do-generate-article \
        --do-topo-generation \
        --do-polish-article \
        --interface file \
        --input-dir ./FreshWiki/final.csv
```



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{gu-etal-2025-rapid,
    title = "{RAPID}: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery",
    author = "Gu, Hongchao  and Li, Dexun  and Dong, Kuicai  and Zhang, Hao  and Lv, Hang  and Wang, Hao  and Lian, Defu  and Liu, Yong  and Chen, Enhong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.859/",
    doi = "10.18653/v1/2025.findings-acl.859",
    pages = "16742--16763",
    ISBN = "979-8-89176-256-5",
}
```


## Acknowledgments

This codebase is primarily based on the original [STORM](https://github.com/stanford-oval/storm) implementation. We would like to express our gratitude to the STORM authors for their valuable contributions.
