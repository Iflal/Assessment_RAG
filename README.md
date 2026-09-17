# RAG Generator

A runtime-configurable retrieval-augmented generation application. It ingests
`.txt`, `.md`, and text-based `.pdf` documents, builds a local Chroma index with
`sentence-transformers/all-MiniLM-L6-v2`, and answers questions through Groq's
`openai/gpt-oss-120b` using only retrieved document chunks.

## Prerequisites

- Python 3.10 or newer
- Internet access for the initial embedding-model download and Groq requests
- A Groq API key for question answering (ingestion does not require one)

Embeddings are generated locally. When you ask a question, the retrieved text
chunks are sent to Groq to generate the answer; do not ingest sensitive material
unless that external processing is acceptable.

## Setup with Conda

Windows PowerShell:

```powershell
conda create -n hrc_rag python=3.10 -y
conda activate hrc_rag
Set-Location "C:\path\to\Assessment_RAG"
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Windows Command Prompt:

```bat
conda create -n hrc_rag python=3.10 -y
conda activate hrc_rag
cd /d C:\path\to\Assessment_RAG
python -m pip install -r requirements.txt
copy .env.example .env
```

macOS or Linux:

```bash
conda create -n hrc_rag python=3.10 -y
conda activate hrc_rag
cd /path/to/Assessment_RAG
python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set the Groq key:

```text
GROQ_API_KEY=your-key-here
```

`HF_TOKEN` is optional. You may add it to `.env` to raise Hugging Face download
limits. Without it, the initial embedding-model download may display a
rate-limit warning, but local embedding still works.

Confirm the CLI is installed correctly:

```text
python cli.py --help
```

## Run

Try the included sample document:

```text
python cli.py ingest "data/alice-in-wonderland.pdf"
python cli.py ask "What animal did Alice follow down the rabbit-hole?"
```

To use your own corpus, provide either a supported file or a folder. Folder
ingestion is recursive:

```text
python cli.py ingest "path/to/documents"
```

Ask a question against the latest successfully ingested document set:

```text
python cli.py ask "What is the retention period?"
```

Tune retrieval without changing source code:

```text
python cli.py ask "What is the retention period?" --top-k 5 --min-similarity 0.40
```

Each successful ingestion creates a complete versioned collection and then
switches `active-index.json` to it. Ingesting another file or folder therefore
replaces the searchable corpus without mixing it with the previous document set
and without requiring code changes.

Useful environment settings are documented in `.env.example`, including index
location, embedding model, chunk size, overlap, top-k, similarity threshold,
Groq model, and maximum completion tokens. Equivalent CLI flags take precedence
where provided.

## Grounding and citations

The query path applies two safeguards:

1. Chroma returns the nearest chunks, but only chunks meeting the minimum cosine
   similarity are accepted. If none qualify, the application returns a fixed
   not-found response and does not call Groq.
2. The generation prompt labels document chunks as untrusted reference data,
   forbids outside knowledge, requires `[S#]` citations, and tells the model to
   refuse when the supplied context is insufficient.

The application appends its own deterministic source list to grounded answers.
It includes the filename, PDF page when available, chunk index, stable chunk ID,
and similarity score, so citations do not depend solely on model formatting.

## Architecture

```text
Ingestion
runtime file/folder
  -> TXT/Markdown/PDF loaders
  -> recursive structure-aware chunker
  -> all-MiniLM-L6-v2 embeddings
  -> versioned Chroma collection
  -> atomic active-index manifest

Question answering
question
  -> normalized MiniLM query embedding
  -> Chroma top-k cosine search
  -> minimum-similarity refusal gate
  -> grounded Groq prompt
  -> answer plus deterministic source list
```

The main responsibilities are separated under `src/rag_generator`:

- `ingestion`: file discovery, text extraction, recursive chunking, orchestration
- `indexing`: local embeddings and persistent Chroma access
- `retrieval`: similarity conversion, top-k selection, threshold filtering
- `generation`: refusal gate, grounded prompt, Groq adapter, citation rendering
- `evaluation.py`: isolated five-case end-to-end evaluation
- `cli.py`: argument parsing, dependency wiring, and user-facing error boundary

## Error handling

Expected failures return exit code `1` with a short message and no traceback:

- Empty folder: reports that no files were found.
- Unsupported-only or empty documents: reports that no readable documents were
  found and lists the supported extensions.
- No active index: tells the user to run `ingest` first.
- Embedding/model failure: suggests checking installation, memory, connectivity,
  or the local Hugging Face cache.
- Groq failure: distinguishes missing/invalid authentication, rate limiting,
  connection failures, rejected requests, and other API failures.

Set `RAG_DEBUG=1` when an unexpected failure requires a developer traceback.

## Tests and evaluation

Run the unit and component tests:

```bat
python -m unittest discover -s tests -v
```

Run the five-case live evaluation:

```bat
python eval.py
```

The evaluation ingests `tests/fixtures/eval_docs/operations_handbook.md` into a
separate `.eval-chroma` index, so it does not replace the normal application
index. It checks four answerable questions for expected facts and citations, and
one unrelated question for explicit refusal. Every case prints `PASS` or `FAIL`,
followed by an overall summary. Because four cases call Groq, the result requires
a valid API key and network access.

## Tradeoffs

- **Chroma instead of FAISS:** Chroma supplies persistence and metadata storage
  directly, making file/page/chunk citations and corpus replacement simple.
  The embedded local client is suitable for this assessment but is not a
  multi-process or distributed production database.
- **MiniLM instead of a larger embedding model:** `all-MiniLM-L6-v2` is small,
  fast on CPU, and produces compact 384-dimensional vectors. The tradeoff is
  lower retrieval quality on specialized, multilingual, or subtle questions.
- **800-character chunks with 120-character overlap:** This is simple and stays
  below MiniLM's short input limit for typical English text. Recursive splitting
  preserves Markdown headers, paragraphs, and sentences where possible, but a
  token-aware splitter would control model input length more precisely.
- **A `0.35` similarity threshold:** It gives a practical refusal gate for the
  included sample but is a heuristic, not a universal confidence score. Each
  document domain should calibrate it against labeled questions.
- **Prompt grounding plus deterministic citations:** This is stronger than a
  prompt-only demo, but generation remains probabilistic. The source footer
  lists retrieved context; it does not formally prove every generated claim.
- **Text-only PDFs:** `pypdf` keeps the dependency footprint small, but scanned
  or image-only PDFs require OCR and are currently rejected as having no text.

## With more time

- Build a larger labeled evaluation set and tune thresholds per document domain.
- Add hybrid keyword/vector retrieval and a cross-encoder reranker.
- Validate model citation labels and reject claims without supporting chunks.
- Add OCR, table extraction, richer PDF layout handling, and more file formats.
- Add structured logging, latency/token metrics, tracing, and API retry policy.
- Run Chroma in server mode with concurrency controls, authentication, backups,
  and retention policies.
- Add incremental indexing based on file hashes instead of rebuilding the full
  corpus, while preserving the current atomic active-index switch.
- Add integration tests against pinned Chroma and Groq SDK versions in CI.
