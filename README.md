## POLYMER – Agent Architecture (Perception → Signification → Mémoire → Raisonneurs → Orchestration)

Professional reference implementation aligned with `documentation/` (architecture, mémoire hybride, raisonneurs, planification, méta‑cognition) and constrained to run locally on Windows with an RTX 5070 (8 GB VRAM) and ~10 GB RAM.

### Features
- Intent/Plan IR (structured Pydantic models)
- Semantic parser (heuristics + optional LLM assist via Ollama)
- Hybrid memory: vector search (FastEmbed + SQLite), episodic SQLite store, knowledge graph (GraphML)
- Reasoners: symbolic checks and Program‑of‑Thought sandbox (RestrictedPython)
- Planner: simple HTN planner
- Orchestrator: metacognitive selection and verification + alignment guardrails
- FastAPI server: `/query`, `/ingest`, `/status`
- Typer CLI: `serve`, `ingest`, `query`

### Hardware & Constraints
- GPU: RTX 5070 with 8 GB VRAM (fits 7–8B quantized models via Ollama)
- RAM: ~10 GB (pipeline is memory‑lean; vectors persisted on disk)

### Quickstart (Windows)
1) Install Python 3.11 (recommended)
2) Install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Install Ollama (optional but recommended for GPU inference)
- Download and install: `https://ollama.com/download`
- Pull a tiny model (1.1B) for fast prototyping:
```bash
ollama pull tinyllama:1.1b
```
Ollama uses GPU automatically on Windows if available. Ensure the Ollama service is running.

4) Configure
Edit `config/default.yaml` as needed. Key fields:
```yaml
llm:
  provider: ollama  # or "mock"
  model: tinyllama:1.1b
memory:
  data_dir: ./data
server:
  host: 127.0.0.1
  port: 8000
```

5) Run the server
```bash
python -m polymer.cli serve
```
Open: `http://127.0.0.1:8000/docs`

6) Ingest and Query via CLI
```bash
# Ingest a file or text
python -m polymer.cli ingest --text "La photosynthèse convertit l'énergie lumineuse en énergie chimique."

# Ask a question
python -m polymer.cli query --question "Explique la photosynthèse en trois étapes."
```

### ACHMRA-Base-Solo Training Pipeline
- Config: update `config/training/achmra-base-solo.yaml` (nouveaux blocs internal_language et thought_graph) et referencez les jeux jsonl generes avec `scripts/build_achmra_dataset.py`.
- Dataset builder: ajustez `datasets/achmra/scenarios.yaml` puis lancez `python scripts/build_achmra_dataset.py --outdir data/achmra` (ou `samples/achmra`) pour obtenir un corpus fiable: chaque entree combine `[THOUGHT]/[FINAL]/[CONF]/[NEXT]`, passes latentes, segments de langue interne (`<il_begin>/<il_bridge>/<il_end>`) et graph-of-thought multi-branche (`<got_section>/<got_node>/<got_edge>`).
- CLI garde-fous: `python -m polymer.cli achmra --stage status` audite jeux/export sans demarrer l entrainement; les autres stages (`sft`, `preference`, `rl`, `evaluate`, `export`) restent manuels.
- Export: `python -m polymer.cli achmra --stage export --checkpoint artifacts/achmra_base_solo/sft` (ou tout checkpoint LoRA) construit les `.gguf` via `llama_cpp_python`.
- Automatisation: `python scripts/run_achmra_pipeline.py --stage status` reproduit le flux pour l integration continue; utilisez `python scripts/train_achmra_full.py --help` pour enchainer SFT/Preference/RL/Export depuis Python.

### Modele Runtime
- Telechargez le GGUF `Qwen3-4B-Instruct-2507-GGUF` depuis https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF (par exemple fichier `Qwen3-4B-Instruct-Q4_K_M.gguf`).
- Utilisez `llama.cpp` release b6096 (https://github.com/ggml-org/llama.cpp/releases/tag/b6096) ou un runner compatible base sur cette version pour l inference locale.
- L entrainement se fait sur le checkpoint HF `lmstudio-community/Qwen3-4B-Instruct` defini dans `config/training/achmra-base-solo.yaml`, puis l export `achmra --stage export` reconstruit le GGUF voulu.

### Endpoints
- `POST /ingest` body: `{ text: str, metadata?: dict }` – add to vector memory and episodic store
- `POST /query` body: `{ question: str }` – returns structured result `{ intent, plan, answer, verification }`
- `GET /status` – health and configuration summary

### Design Notes (mapping to documentation/)
- Perception → Signification: `parser/semantic_parser.py` builds an Intention Graph with uncertainty fields
- Mémoire Hybride: `memory/vector_store.py`, `memory/episodic.py`, `memory/knowledge_graph.py`
- Raisonneurs: `reasoners/symbolic.py`, `reasoners/pot_sandbox.py`
- Planner (HTN): `planner/htn.py`
- Orchestrateur Métacognitif: `orchestrator.py`
- Vérif & Critique: `verification/verify.py` + `alignment/guardrails.py`

### Model Size Guidance (8 GB VRAM)
- Recommended for prototyping (fast): `tinyllama:1.1b`
- Alternatives: `phi3:mini` (≈3.8B q4), `mistral:7b-instruct-q4_K_M` (heavier), `llama3.1:8b-instruct-q4_K_M` (heavier)

If Ollama is unavailable, set `llm.provider: mock` for a deterministic stub.

### Development
Run locally with auto‑reload:
```bash
python -m uvicorn polymer.server.app:app --host 127.0.0.1 --port 8000 --reload
```

Run CLI help:
```bash
python -m polymer.cli --help
```

### Data Locations
- Vector DB: `data/vector.db`
- Episodes DB: `data/episodes.db`
- Knowledge Graph: `data/kg.graphml`

### License
For professional use; add your company’s license policy as needed.



