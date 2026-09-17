# RAG Assessment — Live Prompt Sequence for Codex

A step-by-step script for building the RAG Generator with Codex, structured so your exported transcript shows real steering, not a single mega-prompt.

**How to use this:** run each step, actually look at what Codex produces, then use the "checkpoint" prompt to react to it. Don't skip the checkpoints — they're the part that makes the transcript worth reading.

---

## Step 0 — Kickoff prompt

Paste this first. It sets scope and asks Codex to justify its stack choice before writing code — that justification is itself good transcript material.

```
You are helping me build a RAG (Retrieval-Augmented Generation) Generator as a
45-minute-scoped technical assessment. I want production-reasonable code, not a
toy script, but scoped realistically for the time budget.

REQUIREMENTS (hard):
1. Accepts documents at RUNTIME (via CLI arg or folder path) — not hardcoded
   file paths in the source.
2. Builds a RAG index over whatever documents are provided, with NO code
   changes required to switch to a different document set.
3. Lets a user ask natural-language questions and get answers grounded ONLY
   in the ingested documents.
4. If the answer isn't supported by the retrieved context, the system must
   say so explicitly rather than hallucinating from its own knowledge.
5. Answers cite which source chunk(s)/file(s) they came from.

STACK: Python, a lightweight local vector store (Chroma or FAISS), support
for .txt, .md, and .pdf ingestion, OpenAI (or Anthropic) for embeddings and
generation.

Before writing any code: propose the file structure and give a one-paragraph
justification for the embedding model and vector store choice. Wait for my
go-ahead before implementing.
```

Read its proposal. If it's reasonable, reply `Looks good, go ahead and implement ingestion first.` If something's off (e.g. it picks a vector store you can't easily run), say so — that's your first real correction.

---

## Step 1 — Scaffold + ingestion

```
Implement the repo scaffold and the ingestion pipeline:
- Folder structure as proposed
- requirements.txt, .env.example, .gitignore
- Ingestion: load all supported files from a given folder, chunk with
  overlap, embed, persist to the vector store
- CLI command: `python cli.py ingest <folder>`

Keep it simple and readable. Don't build retrieval/generation yet.
```

**Run it.** Actually execute `python cli.py ingest <some_folder>` against a real folder of docs. Read the chunking code.

### Checkpoint A — catch something real

Look for an actual issue (chunking that ignores paragraph/markdown structure, no overlap, silent failure on unsupported files, etc.) and call it out specifically. Example:

```
This chunker is splitting mid-sentence and doesn't respect markdown headers.
Switch to a recursive splitter that tries to break on headers/paragraphs
first, falling back to sentence boundaries. Also: what happens right now if
the folder has a .docx file in it? Handle unsupported extensions with a
clear skip + warning instead of crashing.
```

This line matters more than anything else in the transcript so far — it proves you read the generated code, not just the chat summary.

---

## Step 2 — Retrieval + generation

```
Now implement retrieval and generation:
- Top-k similarity search with a minimum similarity threshold
- If nothing clears the threshold, do not call the generation model with
  irrelevant context — return a clear "not found in the documents" response
- The generation prompt must instruct the model to answer ONLY from the
  provided chunks and explicitly refuse/hedge if the context is insufficient
- Include source file/chunk citations in every answer
- CLI command: `python cli.py ask "<question>"`
```

**Run it** with a question you know is answered in your docs, and confirm the citation shows up.

### Checkpoint B — test the grounding, on purpose

This is the single most important checkpoint. Ask it a question your documents do *not* cover:

```
Now ask it something the ingested documents definitely don't cover — e.g.
"What's the capital of Mongolia?" if your docs are about something unrelated.
Show me the raw output.
```

If it hallucinates an answer instead of refusing, correct it explicitly:

```
It answered from its own knowledge instead of refusing. Tighten the
similarity threshold and make the system prompt explicitly forbid answering
outside the retrieved context — it should respond with something like
"I couldn't find information about this in the provided documents."
Re-run the same question and show me the new output.
```

Don't move on until you've actually seen the refusal happen in the transcript.

---

## Step 3 — Prove the "no code changes" requirement

This is the one hard pass/fail line item in the brief. Verify it live, don't assume it.

```
Now let's verify the core requirement: switching to a completely different,
unrelated document set with zero code changes. I'm going to ingest a second
folder on a totally different topic and ask questions against it — walk me
through running:
  python cli.py ingest <second_folder>
  python cli.py ask "<a question specific to the second doc set>"
Confirm no source files were edited between the two runs.
```

Actually swap in a second, unrelated folder (e.g. if round 1 was product docs, round 2 could be recipes or a legal FAQ — anything unrelated). Run both commands, paste the real output into the transcript, and note explicitly that it worked with no code edits.

---

## Step 4 — Error handling, eval, and README

```
Add the remaining pieces:
1. Error handling for: empty folder, no documents ingested yet, embedding/
   generation API failure — clear messages, no stack traces to the user.
2. A tiny eval script: 5 hand-written Q&A pairs against one of the test doc
   sets (mix of answerable and unanswerable questions), reporting pass/fail
   for each.
3. README.md with: setup/run instructions, architecture overview, the
   tradeoffs I made given the 45-minute time budget, and what I'd add with
   more time.

Run the eval script and show me the results.
```

Skim the README it writes — if the "tradeoffs" section is generic filler, push back:

```
The tradeoffs section is too generic. Be specific: why Chroma over FAISS,
why this chunk size, what you'd change about the refusal threshold with more
time to tune it.
```

---

## Before you export

- Confirm both document-set test runs are actually in the transcript with real output, not summarized.
- Confirm the hallucination-then-refusal exchange from Checkpoint B is intact — don't let it get buried or edited out.
- Skim start to finish once for anything half-finished or embarrassing, then export.
