# Intelligent Document Q&A

A Python RAG application with a Streamlit interface, LlamaIndex sentence chunking, local MiniLM embeddings, Chroma persistence, and LangChain generation through Ollama or OpenAI. This is an implementation starter, not a production service or a benchmarked system.

## 1. Understand the project

An ordinary chatbot answers from its training. This app first searches your documents and gives the relevant passages to a language model, which answers with numbered source citations. You can open each source passage to check the answer.

Pipeline: upload → extract text → split into overlapping chunks → embed → store vectors in Chroma → embed question → retrieve nearest chunks → filter by distance → prompt LLM → check citation IDs → display answer and evidence.

LlamaIndex handles chunking; LangChain handles embedding and vector-store integration and the prompt/model/output chain. Chroma is the vector database. These roles deliberately avoid building two redundant RAG pipelines.

## 2. Requirements

- Python 3.11 recommended; Windows, macOS or Linux.
- Internet for package installation and the first MiniLM embedding-model download.
- For free local generation: install Ollama and download a model. Roughly 8 GB RAM is a practical starting point for the selected small model; performance depends on hardware.
- Alternatively, an OpenAI API key with available API credit. API usage may cost money; a ChatGPT subscription does not provide API credit.
- Retrieval-only mode needs no generation model or API key. Once its embedding model is cached, retrieval can run locally.

## 3. Setup (Windows PowerShell)

Extract this folder and open a terminal inside it.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell activation is blocked, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py` directly.

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Option A: local Ollama

Install from https://ollama.com/ and run:

```bash
ollama pull llama3.2:3b
```

Keep Ollama running. The default `.env` selects it. Start the application:

```bash
streamlit run app.py
```

### Option B: OpenAI

Edit `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Choose an available chat model for your account if the default is unavailable. Do not commit `.env`. This mode sends the question and retrieved document excerpts to the API; embedding remains local.

## 4. First demonstration

1. Upload `sample_docs/handbook.txt` and `sample_docs/refunds.md`.
2. Click **Build index**. The first run downloads MiniLM.
3. Leave **Retrieval only** checked and ask: `How many annual leave days do employees receive?`
4. Inspect the handbook evidence: **18 days per year**.
5. Uncheck **Retrieval only**, with your chosen model running, and repeat. The answer should cite the passage with `[1]` or another valid retrieved-source ID.
6. Ask: `How long after delivery can I request a refund?` Expected fact: **14 days**.
7. Ask about the CEO's private phone number. The intended response is abstention. If the model answers anyway, record that as a grounding failure; a citation alone is not proof.
8. Upload your own documents. Rebuild after changing uploads or chunk settings. Uploaded document bytes are processed in memory, and extracted chunks persist under `data/`.

PDF sources show actual page numbers. DOCX/TXT/Markdown sources currently show section 1, not actual rendered page numbers. Scanned PDFs require external OCR; this app does not implement OCR.

## 5. Repository map

| File | Purpose |
|---|---|
| `app.py` | Upload, index, question and evidence interface |
| `core.py` | Extraction, chunking, embeddings, retrieval and generation |
| `evaluate.py` | Compare chunk size, overlap and top-k configurations |
| `eval_questions.json` | Small labeled demo question set |
| `sample_docs/` | Two fictional documents with known answers |
| `tests/test_core.py` | Lightweight identity, threshold and citation checks |
| `.env.example` | Provider and model configuration |

Indexes are isolated by a hash of filenames, content, chunk settings and embedding model. Deterministic chunk IDs prevent duplicate entries on repeated indexing. This does not implement authentication or isolation between untrusted users: run as a personal local application.

## 6. Evaluation: make the second résumé bullet real

```bash
python -m unittest discover -s tests -v
python evaluate.py
# Optional generation; requires a working configured model:
python evaluate.py --generate
```

The script compares chunk sizes 256/384/512, overlaps 32/64/64, and top-k 2/4: six configurations. It exports passages, distances and metrics to `results/evaluation.json`.

- **Evidence hit@k:** fraction of answerable questions where a retrieved chunk has both the expected filename and the labeled evidence substring. This is a lightweight evidence check, not exhaustive relevance recall.
- **MRR@k:** mean inverse rank of that first matching chunk, zero if absent.
- **Unanswerable questions:** kept separately through per-case rejection/answer records; they do not enter retrieval hit/MRR denominators.
- **Answer correctness:** manually compare generated answers to `expected_answer`.
- **Grounding:** manually check whether every factual claim is actually supported by its cited passage.
- **Citation ID validation:** only ensures a citation exists and refers to an available passage. It cannot prove correctness or semantic support.

The sample set is intentionally tiny and cannot support an impressive performance claim. Create at least 30–50 questions across PDFs, DOCX and text documents: exact facts, paraphrases, ambiguous wording, multi-passage questions and questions with no answer. Use your own lawful non-sensitive documents; no training dataset is necessary because this system uses pretrained models without fine-tuning.

For each custom answerable question, label `source`, a distinctive `evidence` substring and `expected_answer`; use null source/evidence for unanswerable questions. Run:

```bash
python evaluate.py --docs my_documents --questions my_questions.json --generate
```

Use one development set to select settings and a separate held-out set for the final report. Keep model/provider and threshold constant while comparing chunking. Then separately tune the distance threshold on development data and evaluate abstention on held-out data. Record model name, installed package versions, machine specifications and run date. After a successful install, freeze your environment with `pip freeze > requirements-lock.txt`.

Build a comparison table with baseline and selected settings, evidence hit@k, MRR, manual correctness, manual grounding and unanswerable abstention rate. Do not insert invented measurements. Tiny sample documents may produce identical scores across chunk sizes; longer realistic documents are needed to test chunking meaningfully.

## 7. Debugging and next improvements

- No answers: inspect retrieval-only mode, relax the distance threshold carefully or improve source text/chunking. Cosine distance is not calibrated answer confidence.
- Wrong passage: inspect the saved evaluation examples; tune chunk size, overlap and top-k.
- Ollama connection refused/model missing: start Ollama and run the model pull command.
- OpenAI authentication/rate-limit errors: check key, model availability, credit and limits.
- First run slow: MiniLM downloads and local inference can take time.
- Empty PDF text: supply OCR text first.
- Complex tables/layouts: basic extraction can lose reading order; inspect extracted evidence before evaluating.
- Multi-hop answers: increase top-k only if enough related evidence can be retrieved; no multi-step retrieval exists yet.
- Prompt injection: the prompt treats document text as data, but this is a mitigation, not a guarantee. The model has no action tools.

Useful extensions after establishing a baseline: lexical BM25 + vector hybrid search, reciprocal-rank fusion, cross-encoder reranking, page-aware OCR, incremental ingestion, document filters and an answer-grounding evaluator with manual calibration. Measure each extension independently.

## 8. Verification and limits

During project creation, Python compilation and four standard-library unit tests passed. The supplied environment did not have Streamlit/LangChain/LlamaIndex/Chroma installed; live indexing, embedding downloads, provider calls and UI execution were not verified. Dependency ranges are installation constraints, not a tested lockfile. Perform the setup and demonstration before presenting this as a completed project.

This starter is single-user, uses synchronous ingestion, does not handle OCR or image-based questions, does not stream tokens, and does not guarantee hallucination-free answers. Data persists locally; remove `data/` to remove the indexes when the app is stopped. No uploaded originals are written by the app.

## 9. Honest résumé wording

Once you run and validate the implementation:

> Built a document Q&A application using Python, LangChain, LlamaIndex and Chroma, with document ingestion, local embeddings, source-cited generation and an inspectable retrieval interface.

Only after running your evaluation and selecting a measurably better configuration:

> Evaluated retrieval across [N] questions and [document types], improving evidence hit@k from [baseline]% to [selected]% by tuning chunk size, overlap and retrieval depth.

Use actual numbers. Retain failed examples and evaluation output for interviews.

## 10. Official references

- LlamaIndex nodes/chunking: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_nodes/
- LangChain Chroma interface: https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma
- LangChain embeddings: https://docs.langchain.com/oss/python/integrations/embeddings
- Ollama: https://ollama.com/
