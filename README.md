
# RAG Against the Machine

## Description

This project builds a retrieval-augmented generation system that answers
questions about the vLLM codebase using retrieved source locations and the
local `Qwen/Qwen3-0.6B` model.

## System Architecture

The CLI connects four pipeline stages:

1. **Indexing:** supported corpus files are split into bounded `Chunk` objects,
   converted into BM25 statistics, and persisted under `data/processed/`.
   An optional transformer embedding index can be stored alongside BM25.
2. **Retrieval:** the saved index is validated and scored for a query; BM25,
   semantic cosine similarity, or a normalized hybrid score selects the
   highest-scoring chunks as `MinimalSource` file paths and character ranges.
3. **Augmentation and generation:** source ranges are restored from the corpus,
   combined into a bounded context, truncated to reserve generation tokens, and
   passed to `Qwen/Qwen3-0.6B` with a context-only prompt.
4. **Batching and evaluation:** Pydantic models validate dataset JSON at every
   boundary, while `tqdm` reports progress and the local evaluator calculates
   per-question Recall@k.

## Chunking Strategy

Python files are parsed with `ast`. Top-level statement ranges are packed while
preserving comments, whitespace, Unicode character offsets, and original line
endings. Syntactically invalid files fall back to fixed-size splitting.

Markdown and text files are divided at headings and blank-line block boundaries.
Fenced code blocks are kept together when they fit; oversized blocks are split
into fixed-size ranges. Both strategies preserve exact source slices. The
configurable chunk size is validated in the range 1–2000 characters because a
single longer retrieved span invalidates the evaluator output.

## Retrieval Method

The default retrieval mode uses BM25 with `k1=1.2` and `b=1.0`. Text is
lowercased and tokenized with Unicode word matching, which keeps
underscore-separated Python identifiers such as `trust_remote_code` intact.
Results are ordered by descending score with corpus order as a deterministic
tie-breaker; BM25 zero-score chunks are omitted.

The semantic bonus uses mean-pooled, L2-normalized embeddings from
`sentence-transformers/all-MiniLM-L6-v2` through the existing `transformers`
dependency. `semantic` ranks by cosine similarity. `hybrid` min-max normalizes
both BM25 and semantic scores and combines them with `SEMANTIC_WEIGHT`.
Semantic model weights are loaded lazily and are not included in the submission.
The pickle loader permits only the project BM25, chunk, and semantic classes and
validates their fields before retrieval.

## Instructions

The source code and Makefile targets are split into mandatory and bonus parts:

- `src/mandatory/` and `Makefile.mandatory` contain the core BM25 RAG pipeline.
- `src/bonus/` and `Makefile.bonus` contain semantic or hybrid retrieval,
  incremental indexing, retrieval caching, and the optional HTTP API.
- The two Makefiles are independently runnable. The root `Makefile` includes
  both as a compatibility entry point for `make` commands.

Prerequisites:

- Python 3.10 or later
- uv
- Enough disk space for the vLLM corpus, dependencies, and model cache

### Data layout

The runtime expects the following directory structure:

```text
data/
├── raw/
│   └── vllm-0.10.1/          # Provided source corpus
├── datasets/
│   ├── AnsweredQuestions/    # Public evaluation datasets
│   └── UnansweredQuestions/  # Public query datasets
├── processed/                # Generated indexes and query cache
└── output/                   # Generated search results and answers
```

The evaluator provides the vLLM corpus. For a local run, place the provided
repository at exactly `data/raw/vllm-0.10.1` before indexing because source
paths are compared verbatim during evaluation. A checkout contains the empty
directory layout under `data/`, represented by `.gitkeep` files. For a local
run, place the provided repository at exactly `data/raw/vllm-0.10.1` and place
the four public dataset JSON files under `data/datasets/` before indexing. The
original dataset ZIP is not required in the repository. The corpus, datasets,
generated indexes, and generated outputs are intentionally excluded from Git.

Install the project dependencies:

```bash
make -f Makefile.mandatory install
```

Build the default full-corpus index through the Makefile:

```bash
make -f Makefile.mandatory run
```

The mandatory Makefile creates the fast BM25-only index. Use the bonus
Makefile for semantic and incremental indexing:

```bash
# Build BM25 plus semantic embeddings (downloads the model on first use).
make -f Makefile.bonus run-semantic

# Reuse unchanged files from data/processed/index.pkl.
make -f Makefile.bonus run-incremental

# Combine both optional indexing features.
make -f Makefile.bonus run-incremental-semantic
```

The available commands are:

```bash
# Build the BM25 index.
uv run python -m src index \
  --file_path data/raw/vllm-0.10.1 \
  --max_chunk_size 2000

# Search one question.
uv run python -m src search \
  "What is the default value of trust_remote_code?" --k 5

# Semantic or hybrid search requires an index created with run-semantic.
uv run python -m src.bonus search \
  "Which option controls adapter loading?" --k 5 \
  --retrieval_mode hybrid --semantic_weight 0.5

# Search unanswered questions for answer generation.
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions

# Search answered questions to create evaluation input.
uv run python -m src search_dataset \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/AnsweredQuestions

# Evaluate saved retrieval results.
uv run python -m src evaluate \
  --student_search_results_path \
    data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# Generate one source-grounded answer.
uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 --max_new_tokens 64

# Generate answers for saved retrieval results.
uv run python -m src answer_dataset \
  --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory \
    data/output/search_results_and_answer/UnansweredQuestions
```

### CLI argument reference

Mandatory commands use `uv run python -m src <command>`. The complete argument
surface is:

| Command | Required arguments | Optional arguments and defaults |
| --- | --- | --- |
| `index` | None | `--file_path=data/raw/vllm-0.10.1`, `--max_chunk_size=2000`, `--output_directory=data/processed` |
| `search` | Positional `query` | `--k=5`, `--index_path=data/processed` |
| `search_dataset` | `--dataset_path` | `--k=5`, `--save_directory=data/output/search_results`, `--index_path=data/processed` |
| `evaluate` | `--student_search_results_path`, `--dataset_path` | `--iou_threshold=0.05` |
| `answer` | Positional `query` | `--k=5`, `--index_path=data/processed`, `--model_name=Qwen/Qwen3-0.6B`, `--max_new_tokens=256`, `--max_context_characters=12000` |
| `answer_dataset` | `--student_search_results_path` | `--save_directory=data/output/search_results_and_answer`, `--model_name=Qwen/Qwen3-0.6B`, `--max_new_tokens=256`, `--max_context_characters=12000` |

Bonus commands use `uv run python -m src.bonus <command>`. They retain the
mandatory arguments and add:

| Command | Additional arguments and defaults |
| --- | --- |
| `index` | `--semantic_model_name` (disabled by default), `--incremental=False` |
| `search` | `--retrieval_mode=bm25`, `--semantic_weight=0.5` |
| `search_dataset` | `--retrieval_mode=bm25`, `--semantic_weight=0.5` |
| `answer` | `--retrieval_mode=bm25`, `--semantic_weight=0.5` |
| `serve` | `--host=127.0.0.1`, `--port=8000`, `--index_path=data/processed`, `--model_name=Qwen/Qwen3-0.6B`, `--max_new_tokens=256`, `--max_context_characters=12000`, `--retrieval_mode=bm25`, `--semantic_weight=0.5` |

The bonus CLI also inherits `evaluate` and `answer_dataset` unchanged.

Run the project checks before submission:

```bash
make -f Makefile.mandatory lint-strict
make -f Makefile.mandatory test
```

## Makefile usage

Use `Makefile.mandatory` for the core workflow and `Makefile.bonus` for the
optional features. They can be invoked independently. The root `Makefile`
includes both files as a compatibility entry point.

The core workflow is:

```bash
make -f Makefile.mandatory install
make -f Makefile.mandatory run
make -f Makefile.mandatory search QUERY="What is the default value of trust_remote_code?"
make -f Makefile.mandatory answer QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
```

Run a mandatory target with `make -f Makefile.mandatory <target> NAME=value`.
The mandatory targets are:

| Target | Purpose |
| --- | --- |
| `install` | Install or synchronize project dependencies with `uv`. |
| `run` | Build the default BM25 index. |
| `search` | Search one query with BM25 and print retrieved source ranges. |
| `search-dataset` | Search every question in `DATASET_PATH` and save JSON results. |
| `evaluate` | Evaluate saved retrieval results. |
| `answer` | Retrieve sources and generate one BM25-grounded answer with Qwen. |
| `answer-dataset` | Generate answers for a saved search-results JSON file. |
| `lint` | Run Flake8 and the configured mypy checks. |
| `lint-strict` | Run Flake8 and strict mypy checks. |
| `test` | Run the test suite. |
| `debug` | Start the mandatory CLI under Python's debugger. |
| `clean` | Remove Python bytecode, tool caches, and coverage artifacts. |

The bonus targets are run with `make -f Makefile.bonus <target> NAME=value`:

| Target | Purpose |
| --- | --- |
| `run-semantic` | Build a BM25 plus semantic-embedding index. |
| `run-incremental` | Reuse unchanged files while rebuilding an index. |
| `run-incremental-semantic` | Combine incremental indexing and semantic embeddings. |
| `search-semantic` | Search one query with semantic retrieval. |
| `search-hybrid` | Search one query with BM25 and semantic scores combined. |
| `search-dataset-semantic` | Run semantic retrieval over a dataset. |
| `search-dataset-hybrid` | Run hybrid retrieval over a dataset. |
| `answer-semantic` | Generate an answer using semantic retrieval. |
| `answer-hybrid` | Generate an answer using hybrid retrieval. |
| `serve` | Start the optional local HTTP API on `HOST:PORT`. |

### Makefile variable reference

Override a value by appending `NAME=value` to a Make command. The user-facing
variables are:

| Variable | Default |
| --- | --- |
| `FILE_PATH` | `data/raw/vllm-0.10.1` |
| `MAX_CHUNK_SIZE` | `2000` |
| `QUERY` | The example LoRA endpoint question |
| `K` | `5` |
| `INDEX_PATH` | `data/processed` |
| `DATASET_PATH` | `data/datasets/UnansweredQuestions/dataset_docs_public.json` |
| `SEARCH_DIRECTORY` | `data/output/search_results/UnansweredQuestions` |
| `SEARCH_RESULTS_PATH` | `$(SEARCH_DIRECTORY)/$(notdir $(DATASET_PATH))` |
| `ANSWER_DIRECTORY` | `data/output/search_results_and_answer/UnansweredQuestions` |
| `EVALUATION_RESULTS_PATH` | `data/output/search_results/AnsweredQuestions/dataset_docs_public.json` |
| `EVALUATION_DATASET_PATH` | `data/datasets/AnsweredQuestions/dataset_docs_public.json` |
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` |
| `MAX_NEW_TOKENS` | `256` |
| `MAX_CONTEXT_CHARACTERS` | `12000` |
| `SEMANTIC_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` (bonus) |
| `RETRIEVAL_MODE` | `bm25` (bonus) |
| `SEMANTIC_WEIGHT` | `0.5` (bonus) |
| `HOST` | `127.0.0.1` (bonus API) |
| `PORT` | `8000` (bonus API) |

Use the maintenance targets directly:

```bash
make -f Makefile.mandatory lint
make -f Makefile.mandatory lint-strict
make -f Makefile.mandatory test
make -f Makefile.mandatory debug
make -f Makefile.mandatory clean
```

`clean` removes only local bytecode, test/lint caches, and coverage files; it
does not remove `data/processed/` or `data/output/`.

For dataset processing, run the search step before the answer step:

```bash
make -f Makefile.mandatory search-dataset \
  DATASET_PATH=data/datasets/UnansweredQuestions/dataset_docs_public.json \
  SEARCH_DIRECTORY=data/output/search_results/UnansweredQuestions
make -f Makefile.mandatory answer-dataset \
  SEARCH_RESULTS_PATH=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  ANSWER_DIRECTORY=data/output/search_results_and_answer/UnansweredQuestions
```

Semantic, hybrid, incremental, and cached retrieval require the bonus
Makefile. The mandatory targets use BM25 and do not download an embedding
model. Answer targets load Qwen only when an answer is requested.

For example, override retrieval and generation settings as follows:

```bash
make -f Makefile.mandatory search QUERY="What is the default value of trust_remote_code?"
make -f Makefile.bonus search-hybrid \
  SEMANTIC_WEIGHT=0.5
make -f Makefile.bonus search-dataset-semantic \
  DATASET_PATH=data/datasets/UnansweredQuestions/dataset_docs_public.json \
  SEARCH_DIRECTORY=data/output/search_results/UnansweredQuestions
make -f Makefile.bonus search-dataset-hybrid \
  DATASET_PATH=data/datasets/UnansweredQuestions/dataset_docs_public.json \
  SEARCH_DIRECTORY=data/output/search_results/UnansweredQuestions
make -f Makefile.bonus answer-semantic \
  QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  MAX_NEW_TOKENS=64
make -f Makefile.bonus answer-hybrid \
  QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  MAX_NEW_TOKENS=64
make -f Makefile.mandatory answer-dataset \
  SEARCH_RESULTS_PATH=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  ANSWER_DIRECTORY=data/output/search_results_and_answer/UnansweredQuestions
```

Answer generation is intentionally separate from indexing: indexing has no
question or answer dataset, while answer targets load the local Qwen model only
when an answer is requested.

### Bonus: local HTTP API

The optional local API exposes the same retrieval and answer-generation stages
without adding another dependency. Start it after building the index:

```bash
make -f Makefile.bonus serve
```

It listens on `http://127.0.0.1:8000` by default. The available endpoints are:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the default value of trust_remote_code?","k":5}'

curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?","k":5}'
```

Use `HOST`, `PORT`, `INDEX_PATH`, `MODEL_NAME`, `MAX_NEW_TOKENS`, and
`MAX_CONTEXT_CHARACTERS` to customize the server. Set `RETRIEVAL_MODE` and
`SEMANTIC_WEIGHT` to use hybrid or semantic retrieval through the API.

### Bonus: semantic and hybrid retrieval

Build the optional vectors once, then select the retrieval strategy without
changing the query format:

```bash
make -f Makefile.bonus run-semantic
make -f Makefile.bonus search-semantic
make -f Makefile.bonus search-hybrid SEMANTIC_WEIGHT=0.7
```

If a BM25-only index is used with `semantic` or `hybrid`, the command reports
that the index must be rebuilt with `semantic_model_name`.

### Bonus: incremental indexing

`make -f Makefile.bonus run-incremental` stores each source file's nanosecond
modification time and size. Unchanged files reuse their existing chunks (and
matching semantic vectors when the model name is unchanged), changed/new files
are reprocessed, and deleted files disappear from the new index. A legacy index
without this metadata is safely rebuilt in full.

### Bonus: retrieval cache

Loaded indexes are cached in the process. Query results are also written to
`query_cache.json` beside `index.pkl`, with the index file signature and the
retrieval options in the cache key. Stale, malformed, or unwritable cache data
is ignored; at most 256 recent query results are retained.

## Performance Analysis

Measurements were taken on the local CPU-only environment with the bundled
`data/raw/vllm-0.10.1` corpus. Building the complete BM25 index with
`max_chunk_size=2000` took 4.61 seconds and used approximately 221 MiB maximum
resident memory, below the five-minute requirement.

With `k=5`, the 100-question documentation batch took 27.48 seconds and the
99-question code batch took 30.48 seconds. Summing the independently observed
times gives 57.96 seconds for 199 questions, consistent with the 90-second
target for 200 questions.

Recall@5 was 0.8200 for the answered documentation set and 0.5051 for the
answered code set, meeting the respective targets of 0.80 and 0.50.

A real offline run with the cached `Qwen/Qwen3-0.6B` model, `k=5`, and
`max_new_tokens=64` took 29.72 seconds including retrieval, model loading, and
generation, with approximately 2.36 GiB maximum resident memory. For the
question "What HTTP endpoint is used to dynamically load a LoRA adapter in
vLLM?", it correctly returned `/v1/load_lora_adapter`. Model download time is
excluded because the weights were already cached.

## Design Decisions

- BM25 retrieval returns source paths and character ranges, which are restored
  into bounded context blocks before answer generation.
- The answer generator loads `Qwen/Qwen3-0.6B` lazily so indexing and tests do
  not download model weights.
- If no usable context is available, the answer generator returns an explicit
  insufficient-context answer instead of asking the model to guess.
- Recall matches sources only when paths are identical and range IoU is at
  least 0.05.

## Challenges Faced

- Python AST columns use UTF-8 byte offsets, while evaluator locations use
  Python character offsets. Line-local byte offsets are therefore converted
  back to character positions.
- Universal-newline conversion changed source positions for CRLF files. Corpus
  files are now read with newline translation disabled during both indexing and
  context restoration.
- Python Fire can parse positional values such as `123` and `None` as non-string
  objects despite type hints. Runtime validation converts these cases into
  concise CLI errors without tracebacks.
- Persisted pickle objects can have valid classes but invalid fields. Chunk and
  BM25 metadata are validated before they are used.
- A generic `trust_remote_code` question retrieved conflicting defaults from a
  test helper and the production configuration, and the small model selected
  the test value. Using a question whose top-ranked source directly contains
  the requested fact produced the correct grounded answer; this highlights the
  effect of retrieval noise on generation quality.

## Example Usage

Run the mandatory documentation retrieval and evaluation workflow with:

```bash
make -f Makefile.mandatory install
make -f Makefile.mandatory run
make -f Makefile.mandatory search-dataset \
  DATASET_PATH=data/datasets/AnsweredQuestions/dataset_docs_public.json \
  SEARCH_DIRECTORY=data/output/search_results/AnsweredQuestions
make -f Makefile.mandatory evaluate
```

After indexing, retrieve sources and generate one grounded answer with:

```bash
make -f Makefile.mandatory answer \
  QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  MAX_NEW_TOKENS=64
```

## Resources

- TF-IDF : <https://qiita.com/ebimontblanc/items/80938fc8f56b5a0a9217>
- BM25 : <https://zenn.dev/m_nakano_teppei/articles/b91b8ab59c15e7>
- RAG : <https://qiita.com/jagaimo_daisuki/items/39c3e7f0634e25e0e3a1>
- Chunking : <https://note.com/uns3k/n/n31a8363184b2>
- Indexer : <https://zenn.dev/keisuke_suzuki/articles/40ade82616aaee>

## AI Usage

AI was used for requirements analysis, implementation assistance, test-case
generation, function and variable naming, edge-case review, and README drafting
across the indexing, retrieval, evaluation, and answer-generation stages. The
resulting code was checked with pytest, mypy, flake8, CLI edge-case runs, and
retrieval evaluation against the provided public datasets.
