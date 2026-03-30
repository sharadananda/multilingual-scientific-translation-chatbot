# Multilingual Scientific Translation Chatbot with RAG

**Author:** Sharadananda Mondal  
**Date:** March 2026  
**Contact:**
---

## Overview

This project implements a **multilingual scientific translation chatbot** designed to address a core limitation of current machine translation systems: *fluency does not equal accuracy in specialised domains*. General-purpose MT models trained on broad corpora consistently fail on low-frequency scientific terminology, producing fluent but semantically incorrect translations.

The chatbot incorporates **Retrieval-Augmented Generation (RAG)** to inject domain-specific terminology context at inference time, improving translation accuracy for scientific content without requiring model retraining. It supports translation across **8 major European languages** using state-of-the-art open-source neural MT models.

This work is motivated by the research agenda of the **CHIST-ERA project TINE (Translation is Not Enough)**, which aims to improve science communication across Europe through better machine translation of complex scientific documents, alongside accessibility measures such as text simplification and terminology explanation.

---

## Languages Supported

| Language | ISO Code |
|---|---|
| English | en |
| German | de |
| French | fr |
| Spanish | es |
| Italian | it |
| Portuguese | pt |
| Dutch | nl |
| Polish | pl |

28 direct language pair models are configured, with automatic **English pivot routing** for pairs without a direct model.

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│   RAG Retrieval Component       │
│   TF-IDF similarity search      │
│   over scientific glossary      │
│   → top-k relevant terms        │
└──────────────┬──────────────────┘
               │ context injection
               ▼
┌─────────────────────────────────┐
│   Helsinki-NLP OPUS-MT Model    │
│   (Marian MT architecture)      │
│   Beam search decoding (k=4)    │
└──────────────┬──────────────────┘
               │
               ▼
        Translation Output
               │
               ▼
┌─────────────────────────────────┐
│   BLEU Evaluation               │
│   (sacrebleu)                   │
│   RAG vs. no-RAG comparison     │
└─────────────────────────────────┘
```

### Components

**Translation Models — Helsinki-NLP OPUS-MT**  
Open-source neural MT models trained on the OPUS parallel corpus. Each language pair uses a dedicated fine-tuned model via the Hugging Face `transformers` library. Models use the Marian MT architecture — a transformer-based sequence-to-sequence framework.

**RAG Component — TF-IDF Retrieval**  
A scientific glossary of domain terms (covering NLP, ML, and MT terminology) is indexed using TF-IDF vectors. At inference time, the input text is vectorised and cosine similarity is computed against the glossary index. The top-k most relevant entries are retrieved and prepended to the source text as context before translation. This directly mirrors the architecture described in Lewis et al. (2020).

**Evaluation — BLEU Score**  
Side-by-side BLEU evaluation (Papineni et al., 2002) of RAG-augmented vs. standard translation on a scientific test set. The evaluation section explicitly discusses the limitations of BLEU for scientific MT and motivates the use of neural metrics such as COMET and BERTScore.

---

## Notebook Structure

| Section | Description |
|---|---|
| 1. Introduction | Research context, architecture overview, key references |
| 2. Installation | One-cell setup |
| 3. Imports & Configuration | Libraries and settings |
| 4. Language Configuration | 8 languages, 28 model pairs, pivot routing |
| 5. RAG Component | Scientific glossary construction and TF-IDF indexing |
| 6. RAG Retrieval Function | Cosine similarity retrieval with threshold filtering |
| 7. Translation Engine | Helsinki-NLP models, RAG injection, beam search |
| 8. Chatbot Interface | Interactive ipywidgets UI with RAG toggle |
| 9. BLEU Evaluation | RAG vs. no-RAG comparison on scientific test sentences |
| 10. Discussion & Limitations | Research implications, limitations, future directions |

---

## Installation & Usage

### Requirements

```bash
pip install transformers sentencepiece sacrebleu scikit-learn torch ipywidgets
```

### Running the Notebook

1. Clone this repository:
```bash
git clone https://github.com/sharadananda/multilingual-scientific-translation-chatbot.git
cd multilingual-scientific-translation-chatbot
```

2. Install dependencies:
```bash
pip install transformers sentencepiece sacrebleu scikit-learn torch ipywidgets
```

3. Launch Jupyter:
```bash
jupyter notebook MultlingualScientificTranslationChatbot_SMondal.ipynb
```

4. Run all cells in order. Models are downloaded automatically from Hugging Face on first run and cached locally.

> **Note:** First run will download Helsinki-NLP models (~300MB per language pair). Subsequent runs use the local cache and are significantly faster.

### Interactive Chatbot

Once all cells are run, Section 8 displays an interactive translation widget. Select source and target languages, enter scientific text, toggle RAG on or off, and click **Translate** to see results with retrieved terminology context.

---

## Research Context & Motivation

### The Problem

Current MT systems are trained on general-purpose parallel corpora. When applied to scientific documents, they encounter:

- **Low-frequency terminology** — domain-specific terms rare or absent in training data
- **Compound technical terms** — multi-word expressions with non-compositional translations
- **Discourse-level coherence** — scientific arguments spanning multiple sentences requiring consistent terminology
- **Evaluation inadequacy** — BLEU score measures n-gram overlap but does not capture semantic accuracy on specialised terms

### The RAG Approach

Retrieval-Augmented Generation (Lewis et al., 2020) offers a principled route to address domain-specific translation without full model retraining. By retrieving relevant domain knowledge at inference time and injecting it as context, the model is guided toward correct terminology without modifying its weights. This is particularly valuable for scientific MT where:

- Domain glossaries can be curated by subject-matter experts
- New terminology can be added without retraining
- Context injection is computationally cheap compared to fine-tuning

### Limitations & Future Work

1. **Glossary coverage** — Current glossary is a proof of concept (12 entries). Production systems require thousands of domain-specific terms per scientific field.
2. **Context injection method** — Prepending plain text context is the simplest RAG approach. More sophisticated methods (constrained decoding, terminology-aware beam search) would yield more reliable term injection.
3. **Evaluation** — BLEU inadequately captures scientific translation quality. Future work should incorporate COMET, BERTScore, and human evaluation on domain-specific terminology.
4. **Pivot quality** — English-pivot routing for cross-lingual pairs introduces cascading errors. Dedicated multilingual models (mBART-50, NLLB-200) would address this.
5. **Document-level translation** — Current implementation is sentence-level. Scientific documents require discourse-level coherence across paragraphs.

These limitations directly motivate the research agenda of projects such as **CHIST-ERA TINE**, which investigates novel inference-time strategies for LLMs on complex scientific documents alongside multi-dimensional evaluation and accessibility measures.

---

## Key References

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*. [[Paper]](https://arxiv.org/abs/1706.03762)
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*. [[Paper]](https://arxiv.org/abs/1409.0473)
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*. [[Paper]](https://arxiv.org/abs/2005.14165)
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*. [[Paper]](https://arxiv.org/abs/2005.11401)
- Papineni, K., et al. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL*. [[Paper]](https://aclanthology.org/P02-1040/)
- Helsinki-NLP OPUS-MT models. [[Hugging Face]](https://huggingface.co/Helsinki-NLP)

---

## Related Projects

- [PGDDAProjects](https://github.com/sharadananda/PGDDAProjects) — Data analytics projects from IIIT Bangalore PGDDA programme (credit risk analysis, HR analytics, predictive modelling)

---

## Licence

MIT Licence — see [LICENSE](LICENSE) for details.
