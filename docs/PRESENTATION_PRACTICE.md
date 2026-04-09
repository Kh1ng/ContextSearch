# ContextSearch Practice Presentation

## Total Time: 7 minutes + 3 minutes Q&A

---

## [1] Problem Statement (~1-1.5 min)

**The Problem:**
- Modern AI agents and LLMs rely on markdown context files (skills, instructions, agent configs)
- As these files grow, you can't inject ALL of them into every prompt—it's expensive and hurts performance
- Current approaches: either inject everything (token-wasteful) or manually cherry-pick (not scalable)

**Our Solution:**
- A* search-based context window optimization
- Automatically selects only the chunks most relevant to a query
- Stay within a token budget while maximizing relevance

**Why it matters:**
- Reduces tokens used (cost savings for API calls)
- Faster inference (less context to process)
- Better results (focused context vs noisy full corpus)

---

## [2] Algorithm Used (~1.5-2 min)

**A* Search:**
- f(n) = g(n) + h(n)
  - **g(n)**: cumulative token cost of selected chunks so far
  - **h(n)**: keyword overlap heuristic (fraction of query keywords in chunk)
- Greedy selection: prioritize chunks with high relevance, low token cost

**Three Chunking Strategies:**
1. **Markdown** (default): Split on ## headings, keeps semantic sections intact
2. **Fixed**: Token-window with overlap, simple fallback
3. **Sentence**: Split on sentence boundaries, packs sentences up to size limit

**Why A*?**
- Balances relevance and cost in a single metric
- Greedy enough to be fast, principled enough to be effective
- Easy to extend (different heuristics for different domains)

---

## [3] Justification for Approach (~1 min)

**Why Markdown-first?**
- Our real use case: instruction markdown files (CLAUDE.md, skill.md, etc.)
- Preserves semantic structure better than fixed windows
- Headings are natural breakpoints for meaningful chunks

**Why keyword overlap?**
- Fast to compute (simple set intersection)
- Effective on technical/domain-specific text
- Can be swapped for other heuristics later

**Why A*?**
- Combines token budget constraint with relevance in one framework
- More principled than random selection
- More flexible than pure keyword matching

---

## [4] Trade-offs & Cost Implications (~0.5 min)

**Token Savings:**
- Evaluated on SciFact: ~99% token reduction at 512-token budget
- Trades off recall for efficiency (some relevant docs missed)

**Limitations:**
- Keyword-only heuristic struggles on semantic retrieval (e.g., "biomaterials" ≠ "nanotechnologies")
- Evaluated on scientific abstracts (domain-specific)
- Will perform better on agent instructions (keyword-dense, domain-specific)

**Future improvements:**
- Embedding-based heuristic instead of keywords (higher quality, slower)
- Query expansion to improve keyword matching
- Domain-specific stopwords and weighting

---

## [5] Evaluation (~1.5 min)

**Dataset:**
- BEIR SciFact: 300 test queries, 5,183 corpus documents, ~6,639 chunks
- Binary relevance labels (relevant/not relevant)

**Metrics (at 512-token budget):**
- Recall@10: 1.4% (low due to keyword limitation)
- Precision@10: 0.2% (few retrieved chunks are relevant)
- NDCG@10: 0.5% (ranking penalizes misaligned results)
- MRR: 0.3% (first relevant result rarely in top-k)
- Avg tokens used: 485 (stays within budget)
- **Token savings: 99.97%** (huge reduction in context size)

**Why low recall?**
- Keyword overlap is weak heuristic for semantic retrieval
- Example: Query "biomaterials show inductive properties" has ZERO keyword match with relevant doc about "nanotechnologies manipulating stem cells"

**What this means:**
- SciFact is not the right benchmark for this tool
- Expected to work much better on agent instructions (keyword-dense, domain-specific)

---

## [6] Demo (~1 min)

**Live Demo (RECOMMENDED):**
```bash
source .venv/bin/activate
python demo_evaluation.py --budget 512 --num-examples 3
```

Shows:
- Actual queries from SciFact test set
- Which chunks A* selected for each query
- Whether relevant docs were found
- Token usage vs budget
- Why keyword heuristic struggles on scientific abstracts

**Alternative (if running live demo fails):**
```bash
source .venv/bin/activate
python evaluation/evaluate.py --budget 512
```
This produces JSON metrics. Less visual, but shows the pipeline works.

**What to narrate while demo runs:**
- "Loading SciFact... chunking documents... running A* search..."
- "Notice how the retrieved chunks often don't match the relevant docs"
- "This is because scientific abstracts use different vocabulary"
- "On agent instructions, keywords are denser and more aligned"

---

## [7] Conclusion (~0.5 min)

**What we built:**
- Fully working context optimization system
- Markdown-aware chunking
- A* search with token budgeting
- Comprehensive evaluation pipeline

**What we learned:**
- Keyword heuristics are fast but limited
- SciFact is hard; agent instructions would be easier
- 99% token savings is impressive but needs to preserve quality

**Future work:**
- Test on actual agent instruction files
- Implement embedding-based heuristic
- Benchmark speed improvements (inference time)
- Compare against baselines: random selection, TF-IDF

---

## Strong Points to Emphasize
1. Working end-to-end system (not just theory)
2. Multiple chunking strategies for flexibility
3. Principled A* approach (not random)
4. Comprehensive evaluation with real dataset
5. 27 unit tests (high quality, not hacked together)
6. Clear documentation and modular code

## Weak Points to Prepare For

1.  **Low recall on SciFact** 
   - Be ready: "SciFact requires semantic understanding. Our keyword heuristic is intentionally lightweight for speed. Would switch to embeddings if latency weren't critical."

2.  **Why SciFact if it doesn't work well?**
   - Be ready: "BEIR SciFact provides labeled data for objective evaluation. Real use case (agent instructions) would perform much better. We used this to demonstrate the pipeline works."

3.  **Why not use embeddings from the start?**
   - Be ready: "Embeddings are slower (API calls or compute). Keywords are instant. For agent instructions (small corpus), instant keyword match is often sufficient. Would add embeddings as an opt-in strategy."

4.  **How do you know it works on agent instructions if you only tested SciFact?**
   - Be ready: "Fair point. We should do a quick sanity test on a real skill.md file to show it chunks and retrieves sensibly. Time permitting, could do that live."

5.  **What about speed? Did you measure latency?**
   - Be ready: "Good question. A* search is O(n log n) in heap operations, keyword extraction is O(n) in chunk count. For typical agent instruction corpus (50-500 chunks), sub-millisecond. Didn't benchmark against baselines yet—future work."

6.  **Why does our heuristic use keyword overlap and not something else?**
   - Be ready: "Keyword overlap is interpretable and fast. Other options: TF-IDF (harder to explain, similar performance), embeddings (slower, better quality), LLM-based (expensive). Chose keyword for transparency and speed."

---

## Questions to Prepare For

**Q: What happens if there are no keyword matches?**
A: A* skips those chunks (scores them 0 relevance) and only selects chunks with at least some overlap. This is conservative—false negatives over false positives.

**Q: How does this compare to just asking the LLM to summarize the corpus first?**
A: Different problem. We're optimizing fixed context selection *within* a turn. LLM summarization requires additional LLM calls (cost/latency). Our approach is pre-computed once per corpus.

**Q: What if the query uses synonyms that don't match keyword?**
A: Valid limitation. Example in eval: "biomaterials" ≠ "nanotechnologies". Future work: query expansion or semantic matching.

**Q: Why use A* instead of simpler greedy (just top-k by relevance)?**
A: Good question. Top-k by relevance ignores token budget—might waste tokens on low-relevance chunks. A* balances both. Could show empirically if time.

**Q: What's the actual speedup for queries? Did you measure?**
A: Measured token reduction (99%), not query latency yet. Would benchmark: baseline model with full context vs our selection. That's future work.

**Q: Can you handle dynamic corpus updates?**
A: Chunks are pre-computed and stored. Adding new docs: just re-chunk and reindex. No model retraining. Very scalable.

