# Topic 4 — Async Research Assistant

Ask a research question, retrieve excerpts from Wikipedia, arXiv, and web search
concurrently, and receive one synthesized answer with numbered references. The
application adds persistent source caching, retries with exponential backoff,
per-source timeouts, graceful degradation, source filtering, and a Docker CLI
around the provided `ai/` package.

## Architecture overview

| Component | Responsibility |
| --- | --- |
| `Settings` / `get_settings()` | Validate environment and `.env` settings; cache the settings instance. |
| `FileSystemCacheStore` | Persist source results as JSON with canonical query keys and a TTL. |
| `AIService` | Wrap `ai.*` fetching and synthesis with retries, timeouts, and logging. |
| `SourceOrchestrator` | Fetch requested sources concurrently using a semaphore and shared HTTP client; collect failure notes. |
| `ResearchAssistant` | Validate the question, reuse cached sources, fetch missing sources, and request synthesis. |
| `src.cli` | Construct the application, parse arguments, and render answers, references, and errors. |
| `researcher` package | Delegate `python -m researcher` to `src.cli.main()` without moving the `src/` code. |

See [docs/architecture.md](docs/architecture.md) for the diagram and request flow.
The supplied `ai/` package remains unchanged.

## Prerequisites and setup

The verified environment is **Python 3.13.15 on macOS ARM64**; the Docker image uses
Python 3.13.15 on Debian Bookworm. Use Python 3.13 for the pinned dependencies.
Other Python versions have not been verified. Docker is optional for local Python
usage and required for the container workflow.

Run commands from the repository root. For a new checkout, using Bash or Zsh:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Keep an existing `.env` instead of replacing it. The finalized `requirements.txt`
pins runtime dependencies, all supported provider SDKs, test tools, and their
transitive dependencies. Use it for this application; `requirements-ai.txt` is the
original AI-module dependency reference.

For live research, edit `.env` with your provider selection, model, and credentials.
Use plain `KEY=value` lines with comments on separate lines: move the inline
comments in the copied template off the value lines before using Docker's
`--env-file`. Keep `.env` local; it is excluded from Git and the Docker build.

The SE settings read `.env` automatically, but the provided AI adapters use
`os.getenv()`. Export your edited `.env` before running a local live question:

```bash
set -a
source .env
set +a
```

Help, tests, and the offline demo do not need provider credentials. Docker's
`--env-file` supplies the environment inside the container without the export step.

## Streamlit UI

The project includes an optional Streamlit interface. From the repository root
in a normal Windows PowerShell terminal, create and activate a Python 3.13
environment and install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.env` does not already exist, create it with `Copy-Item .env.example .env`.
Edit it to configure your LLM provider, model, and required provider credentials
(see below). Keep comments on separate lines and never commit API keys. The UI
loads `.env` automatically; existing environment variables take precedence.

Start the UI:

```powershell
python -m streamlit run ui/app.py
```

Streamlit prints a local URL, normally [http://localhost:8501](http://localhost:8501).
Enter a research question, choose Wikipedia, arXiv, and/or Web, click **Research**,
and view the generated answer and citations. External sources may fail
independently; the application supports graceful degradation using remaining
sources when possible. If no sources are available or synthesis fails, the UI
shows an error.

## Environment variables

For live synthesis, supply credentials for the chosen LLM provider. Wikipedia and
arXiv do not require keys. Web search uses the selected provider's credentials,
unless DuckDuckGo is selected. Selecting only `wiki,arxiv` avoids web-search
credentials but still requires an LLM for synthesis.

| Provider variable | Default / supported values | When needed |
| --- | --- | --- |
| `LLM_PROVIDER` | `anthropic`; also `openai`, `gemini` (`google` alias) | Selects live synthesis provider. |
| `LLM_MODEL` | Adapter defaults: `claude-sonnet-4-6`, `gpt-4o-mini`, or `gemini-2.0-flash` | Set a model available to your selected provider/account; update it when switching provider. These are code defaults, not verified live model availability. |
| `ANTHROPIC_API_KEY` | No default | Anthropic credential. |
| `OPENAI_API_KEY` | No default | OpenAI credential. |
| `GOOGLE_API_KEY` | No default | Preferred Gemini credential. |
| `GEMINI_API_KEY` | No default | Gemini fallback after `GOOGLE_API_KEY`. |
| `LLM_API_KEY` | No default | Final credential fallback for the selected LLM provider. |
| `WEB_SEARCH_PROVIDER` | `tavily`; also `serper`, `duckduckgo` (`ddg` alias) | Selects web search when `web` is requested. |
| `TAVILY_API_KEY` | No default | Required for Tavily web search. |
| `SERPER_API_KEY` | No default | Required for Serper web search. |

DuckDuckGo has no application API-key setting. Provider-specific LLM keys take
precedence over `LLM_API_KEY`. The provided package also has embedding adapters,
but this research pipeline does not use embeddings or `EMBEDDING_*` settings.

| Optional SE setting | Default | Meaning |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; case is normalized. |
| `CACHE_DIR` | `./.cache` | Persistent source-cache directory, relative to the working directory. |
| `CACHE_TTL_SECONDS` | `86400` | Positive cache lifetime in seconds. |
| `PER_SOURCE_TIMEOUT_SECONDS` | `10` | Positive per-source fetch budget in seconds. |
| `MAX_SOURCES_PER_QUERY` | `3` | Positive limit used both for concurrent source tasks and the default result count per source. |

Environment values take precedence over `.env` values for SE settings. Invalid
settings are rejected by Pydantic.

## CLI usage

These help commands run without network or provider access:

```bash
python -m researcher --help
python -m researcher ask --help
```

The following commands perform live research and require the configuration above:

```bash
python -m researcher ask "What is photosynthesis?"
python -m researcher ask "How do transformers handle long context?" --sources wiki,arxiv
python -m researcher ask "What is photosynthesis?" --no-cache
```

By default, the assistant requests `wiki`, `arxiv`, and `web`, reusing unexpired
cached sources. `--sources` accepts a non-empty comma-separated selection of those
names; unknown names and empty entries are rejected. Duplicates are removed in
first-occurrence order. `--no-cache` bypasses both cache reads and writes. Synthesis
still runs for each question even when all sources are cached.

Questions are stripped and must contain 1–2,000 characters. Output includes the
answer and a `References:` section sorted by citation index, with origin, title,
and URL. Terminal escape/control sequences are removed while normal Unicode and
citation markers are preserved. If a source fails or times out but others provide
results, the answer includes a note identifying the unavailable source and reason.
An empty successful source response alone does not create a degradation note.

Exit codes are `0` for success, `2` for argument/invalid-question errors, and `1`
when no research sources are available. These handled errors are concise stderr
messages. Application diagnostics use logging on stderr.

## Reproducible offline demo

```bash
python scripts/demo.py
```

The script processes all five records in
[data/research_questions.json](data/research_questions.json) through the real SE
components, using the explicit canned sources and templated LLM from `demo_ai.py`.
Only external fetch boundaries and LLM selection are replaced. Real synthesis
formats the prompt and builds citations. Its temporary cache is removed afterward.

Each run creates or refreshes these deterministic outputs:

```text
artefacts/q1_offline.json
artefacts/q2_offline.json
artefacts/q3_offline.json
artefacts/q4_offline.json
artefacts/q5_offline.json
```

Unrelated files are preserved, and output symlinks are rejected. Each JSON file
contains `question_id`, `mode`, `demo_note`, `question`, `answer`, nested
`citations` (`index` and `source`), and `degradation_notes`.
These are **offline demonstration fixtures, not live research results**: some
references are generic or illustrative and have not been verified as evidence.

## Tests and verified results

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

Verified on **2026-09-19** with the pinned environment: **141 passed** in each run;
**97% statement coverage for `src`** (485 statements, 13 missed). This percentage
does not measure the provided `ai/` package or scripts.

Tests are offline. CLI tests replace the assistant boundary, while integration
tests retain the real SE pipeline and replace external fetch/LLM boundaries. The
integration tests cover all five dataset questions, filesystem-cache reuse across
assistant instances, deterministic demo reruns, and preservation of unrelated
files. They also block socket connections and DNS lookups. The provided smoke
tests remain unchanged.

## Docker quick-start

With Docker Desktop or a compatible Docker engine running:

```bash
docker build -t async-research-assistant .
docker run --rm async-research-assistant --help
docker run --rm async-research-assistant ask --help
```

The single-stage image installs pinned dependencies before copying source into
`/app`, checks installed dependency consistency, and uses
`ENTRYPOINT ["python", "-m", "researcher"]` with default `CMD ["--help"]`.
The build context excludes `.env`, virtual environments, Git metadata, caches,
coverage, and generated artefacts. Source code and question data remain included.

For live research with an edited, valid `.env`:

```bash
docker run --rm --env-file .env async-research-assistant \
  ask "What is photosynthesis?"
```

Build and both container help commands were verified on 2026-09-19. No live
research query was executed during documentation verification. The quick-start
container's cache is discarded on removal; it does not reuse the host `.cache`.
This is a CLI application; no HTTP server or exposed port is required.

## Sequential versus concurrent benchmark

```bash
python scripts/benchmark.py
```

This is an **offline simulated-I/O benchmark**, not a measurement of real API
latency. P2's `BenchmarkAIService` sleeps for 0.5 seconds per source and returns
canned results. For each dataset question, the script measures three sequential
fetches and then three fetches through the real `SourceOrchestrator`. The parallel
run uses a concurrency limit of three. Client setup and scheduling overhead are
included; HTTP requests, LLM synthesis, cache operations, and the real `AIService`
retry facade are not measured.

Raw values printed by the original Step 4 run on 2026-09-19, retained without adjustment:

| Dataset question | Sequential | Parallel |
| --- | --- | --- |
| q1 — Photosynthesis | 1.56s | 0.51s |
| q2 — Transformer context | 1.51s | 0.51s |
| q3 — Financial crisis | 1.51s | 0.51s |
| q4 — Fusion research | 1.51s | 0.51s |
| q5 — CRISPR-Cas9 | 1.51s | 0.51s |
| **Reported total** | **7.61s** | **2.55s** |

The script reported **2.98x speedup**. It prints two decimal places and calculates
totals/speedup from underlying measurements, so displayed rows need not sum
exactly to the displayed total. The simulated waits are fixed; measured wall time
varies between runs. Independent I/O waits overlap under asyncio, making the
parallel duration approach the longest wait rather than the sum of all waits.

The resumed verification run on the same date also succeeded. Its latest raw
output was:

```text
Question 1: Sequential: 1.54s | Parallel: 0.51s
Question 2: Sequential: 1.51s | Parallel: 0.51s
Question 3: Sequential: 1.51s | Parallel: 0.51s
Question 4: Sequential: 1.51s | Parallel: 0.51s
Question 5: Sequential: 1.51s | Parallel: 0.51s
Sequential total: 7.58s
Parallel total:   2.53s
Speedup:          2.99x
```

Both measurements use the same simulated delays; neither represents live
Wikipedia, arXiv, web-search, or LLM latency.

## Type checking

Neither mypy nor pyright was initially available. Mypy **2.3.1** was installed only
in the local `.venv` as a development verification tool; it is not in the
application requirements. To reproduce this check in a new environment:

```bash
python -m pip install mypy==2.3.1
python -m mypy --version
python -m mypy src researcher scripts
```

The exact check above was run on 2026-09-19 and **failed with exit code 1**:

```text
Found 30 errors in 4 files (checked 19 source files)
```

| File | Errors | Reported issue |
| --- | --- | --- |
| `ai/providers/anthropic.py` | 25 | Optional model types, SDK content-block union narrowing, and a vision content argument type. |
| `ai/providers/openai.py` | 2 | Optional model values passed to dictionary lookup and the embeddings SDK. |
| `demo_ai.py` | 1 | A gather result may be `BaseException` when passed to `list.extend`. |
| `scripts/demo.py` | 2 | Mypy rejects the `_env_file` keyword in its inferred `Settings` constructor; runtime tests pass with this argument. |

The command follows local imports, so diagnostics also include supplied modules
outside the three requested directories. No errors were reported in `src/` or
`researcher/` in this run. This is the default mypy check, not strict checking or
proof that all unannotated code is type-safe. No code was rewritten or errors
suppressed to force a clean result.

## Limitations and design notes

- Live research depends on valid provider credentials, available models, and
  reachable external services. Source degradation does not guarantee synthesis
  will succeed; provider/configuration/cache errors outside the two handled CLI
  domain exceptions can surface as tracebacks.
- Cached data consists of retrieved sources, not final answers. Real LLM answers
  need not be deterministic, and numeric citation tracking does not verify claims.
- The supplied offline fixtures demonstrate integration and serialization rather
  than research quality. Static typing findings above remain unresolved.

## AI-assistance acknowledgement

AI coding assistance was used substantially for implementation, tests, debugging,
and documentation. The team remains responsible for reviewing the code and
explaining the design and verification results.
