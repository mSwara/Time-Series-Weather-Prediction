# Phase 1 — Placement JD Skill-Frequency Analysis

## Sample

- Index page: `https://ajithsaip.github.io/Placement_JD_2025_2026/` — ~690 rows total.
- ~105-110 rows judged plausibly DS/ML/AI-relevant by title (ambiguous titles were
  opened and checked against JD content; 2 were excluded after reading despite an
  AI-sounding name/title because the JD itself had no ML/AI content — Aditya Birla
  S&T "Scientist - Modelling & Simulation" turned out to be ANSYS/Fluent/COMSOL
  engineering simulation, and Ayna.AI "Analyst" turned out to be plain Excel/data
  analysis work).
- **58 JD pages fetched and used** in the tally below (60 fetched, 2 excluded per
  above) — a broad, non-cherry-picked sample across big tech/MNC, Indian IT
  services, AI-native startups, BFSI, and D2C/product companies. ~10 of the 58
  returned only placeholder text ("refer JD" / an unresolvable linked doc) with no
  extractable skill detail — they're kept in the denominator, which means the
  frequencies below are, if anything, a slight *undercount*.
- Full (company, role, URL) list of all 60 fetched JDs is preserved in the agent
  transcript this report summarizes; available on request if a specific claim needs
  spot-checking.

## Skill frequency (out of 58 JDs)

| Skill (from the original tracked list) | Frequency | Companies (sample) | Verdict |
|---|---|---|---|
| PyTorch | 14/58 | 10x Construction.ai, Chariot, DENSO, ESRI, HiLabs, Meesho, Navi, NoBroker, SuperAGI, TURIUM AI, Telus | Most-named DL framework |
| TensorFlow | 10/58 | Augnito, DENSO, HiLabs, Meesho, MPHASIS, Navi, NoBroker, SuperAGI, TURIUM AI, Telus | Second most-named; usually offered as "TF or PyTorch" |
| CNN | 3/58 | MPHASIS, JAVIS (Vision AI), Avaada | Named explicitly, not dominant |
| LSTM | **1/58** | MPHASIS | One mention total — genuinely rare here |
| CNN-LSTM (hybrid) | **0/58** | — | Not named anywhere in the sample |
| Keras | 1/58 | MPHASIS | Named once, as a third option |
| Keras Tuner | **0/58** | — | Not mentioned anywhere |
| Dropout | **0/58** | — | Not named explicitly (implementation detail, not a marketed keyword) |
| EarlyStopping | **0/58** | — | Not named explicitly |
| Regularization / L2 | **0/58** | — | Not named explicitly |
| ModelCheckpoint | **0/58** | — | Not named anywhere |
| Synthetic data / augmentation | 3 companies | 10x Construction.ai, Augnito, Miko | Framed as CV/speech/NLP augmentation, not tabular class-imbalance |
| SMOTE | **0/58** | — | See below |
| ADASYN | **0/58** | — | See below |
| Time-series windowing | 2/58 | Commonwealth Bank of Australia, AGENT MIRA | Both say "time series" generically; nobody uses the word "windowing" |
| Adam | **0/58** | — | Not named explicitly |
| SGD | **0/58** | — | Not named explicitly |
| RMSprop | **0/58** | — | Not named explicitly |

### Additional skills that surfaced strongly and were NOT in the original list

| Skill | Frequency | Why it matters |
|---|---|---|
| LLM / GenAI / RAG / prompt engineering / fine-tuning (SFT, RLHF, LoRA) | ~20+/58 | By far the dominant theme in this JD pool — more common than every classic supervised-DL skill combined |
| Transformer / Attention / BERT / GPT-family / ViT | ~8/58 | More common than CNN (3/58) or LSTM (1/58) individually |
| Computer vision (detection, segmentation, classification) | ~7-9/58 | Recurring vertical, especially product/e-commerce and geospatial roles |
| Classical stats / EDA / feature engineering / regression / classification | ~15+/58 | Near-universal baseline for "Data Scientist"-titled roles, more common than any single DL architecture |
| Vector databases (FAISS, Pinecone, Milvus, ...) | 4/58 | Now a standard part of the RAG stack |
| Basic MLOps (Docker, MLflow, SageMaker, Airflow) | 5/58 | Present but secondary, framed as deployment hygiene |
| GANs / VAEs / diffusion | 3/58 | Present in the more research-flavored roles |

## SMOTE / ADASYN — honest verdict

**Zero hits for both, across all 58 JDs.** This JD pool skews toward GenAI/LLM work,
CV/speech, and broadly-described "classical DS generalist" work (regression,
classification, clustering) — none of it framed in explicit "imbalanced dataset" or
"fraud/churn classification" language that would summon SMOTE/ADASYN by name, even
though a few JDs mention fraud-adjacent use cases in passing (Meesho, American
Express). Per your instruction not to force a technique where it doesn't fit: SMOTE
and ADASYN are **not evidenced as a JD-driven priority at all** in this sample — they
remain in this project only because they're the textbook-correct tool for the
auxiliary heat-stress classification sub-task on genuinely physical grounds, not
because the JDs asked for them. That sub-task stays scoped and secondary, not the
project's headline.

## Priority ranking for this project

**High priority, JD-evidenced:** PyTorch/TensorFlow familiarity (either), classical
ML/EDA/feature-engineering fundamentals, and (where the problem domain allows it)
Transformer/attention-style architectures — all three well ahead of CNN/LSTM
individually in this pool.

**Low priority / not JD-evidenced here:** CNN-LSTM as a named hybrid, Keras Tuner,
Dropout/EarlyStopping/L2/ModelCheckpoint/Adam/SGD/RMSprop as *named* keywords (they
remain correct engineering practice regardless — just not something these JDs
market by name), SMOTE/ADASYN as a general priority (kept only for the scoped
classification sub-task).

**Not applicable to this project, despite being the single largest JD theme:**
GenAI/LLM/RAG. It dominates this JD pool, but forcing a language-model component
into a 300-row numeric meteorological regression problem would be exactly the kind
of shoehorning this brief asked me to avoid for SMOTE/ADASYN — wrong data modality,
no legitimate task for it here. Named and set aside rather than silently ignored.

## How this changes the Phase 2 design actually being built

The brief's own Phase 2 spec was written in detail *before* this analysis came back,
with the instruction to "confirm/adjust the stack per Phase 1 findings" — not to
replace the architecture outright. Given the findings above:

- **Kept as specified:** TensorFlow/Keras, CNN-LSTM hybrid (plus plain CNN/LSTM
  baselines), Dropout/L2/EarlyStopping/ModelCheckpoint, Keras Tuner. These are not
  JD-frequent *by name* in this pool, but they were independently justified by the
  dataset's own structure (§3-4 of `phase2_design.md`), not by JD frequency — and
  they remain correct small-sample engineering practice regardless of whether a JD
  happens to name them.
- **One concrete addition, justified by this analysis:** a lightweight
  **self-attention layer** over the 4-timestep sequence, ahead of the LSTM
  output/pooling step. Attention/Transformer-style architectures appeared in 8/58
  JDs — more than plain CNN (3/58) and far more than LSTM alone (1/58) — making
  this the one architectural adjustment actually evidenced by Phase 1 data, and it
  is small enough (a handful of extra parameters over a 4-step sequence) not to
  compromise the small-sample discipline in §4 of the design doc.
- **Explicitly not adopted:** GenAI/LLM/RAG components (wrong data modality for
  this problem, as above) and SMOTE/ADASYN as a headline skill (kept scoped to the
  auxiliary classification sub-task only, per the honesty check requested).
- **Framing change:** the write-up will not claim CNN-LSTM/Keras Tuner/SMOTE as
  "what these JDs are asking for" — because they mostly aren't, in this sample. The
  project's actual pitch to a recruiter reading this pool is: sound small-sample DL
  engineering discipline + explicit awareness of what the market actually asks for
  (PyTorch/TensorFlow fluency, attention-based architectures, classical DS
  fundamentals) — demonstrated honestly rather than keyword-stuffed.
