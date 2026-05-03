# Waggle RLM-style Benchmark Results

> **Warning:** This benchmark follows the benchmark families used in the RLM paper,
> but uses deterministic synthetic memory tasks mapped to Waggle's graph/transcript
> environment. It should **not** be compared numerically to the RLM paper until the
> exact public datasets and matching model setup are run.

| Benchmark family | Scale | Method | Score | F1 | Ev. Coverage | Tokens returned | Latency (ms) |
|---|---:|---|---:|---:|---:|---:|---:|
| S-NIAH-style | 32 | raw_context | 1.000 | 1.000 | 1.000 | 572 | 2 |
| S-NIAH-style | 32 | query_graph | 1.000 | 1.000 | 1.000 | 93 | 2 |
| S-NIAH-style | 32 | build_context | 1.000 | 1.000 | 1.000 | 251 | 9 |
| S-NIAH-style | 128 | raw_context | 1.000 | 1.000 | 1.000 | 943 | 3 |
| S-NIAH-style | 128 | query_graph | 1.000 | 1.000 | 1.000 | 93 | 5 |
| S-NIAH-style | 128 | build_context | 1.000 | 1.000 | 1.000 | 181 | 21 |
| BrowseComp-Plus-style | 32 | raw_context | 1.000 | 1.000 | 1.000 | 175 | 1 |
| BrowseComp-Plus-style | 32 | query_graph | 1.000 | 1.000 | 1.000 | 98 | 1 |
| BrowseComp-Plus-style | 32 | build_context | 1.000 | 1.000 | 1.000 | 193 | 9 |
| BrowseComp-Plus-style | 128 | raw_context | 1.000 | 1.000 | 1.000 | 175 | 1 |
| BrowseComp-Plus-style | 128 | query_graph | 1.000 | 1.000 | 1.000 | 98 | 1 |
| BrowseComp-Plus-style | 128 | build_context | 1.000 | 1.000 | 1.000 | 193 | 8 |
| OOLONG-Pairs-style | 32 | raw_context | 1.000 | 1.000 | 1.000 | 611 | 1 |
| OOLONG-Pairs-style | 32 | query_graph | 0.000 | 0.000 | 0.000 | 97 | 2 |
| OOLONG-Pairs-style | 32 | build_context | 1.000 | 1.000 | 1.000 | 411 | 9 |
| OOLONG-Pairs-style | 128 | raw_context | 0.000 | 0.000 | 0.000 | 948 | 3 |
| OOLONG-Pairs-style | 128 | query_graph | 0.000 | 0.000 | 0.000 | 98 | 5 |
| OOLONG-Pairs-style | 128 | build_context | 1.000 | 1.000 | 1.000 | 515 | 83 |
| CodeQA-style | 32 | raw_context | 1.000 | 1.000 | 1.000 | 826 | 1 |
| CodeQA-style | 32 | query_graph | 1.000 | 1.000 | 1.000 | 178 | 2 |
| CodeQA-style | 32 | build_context | 1.000 | 1.000 | 1.000 | 378 | 15 |
| CodeQA-style | 128 | raw_context | 0.000 | 0.500 | 0.500 | 920 | 2 |
| CodeQA-style | 128 | query_graph | 1.000 | 1.000 | 1.000 | 178 | 5 |
| CodeQA-style | 128 | build_context | 1.000 | 1.000 | 1.000 | 535 | 33 |

## Token efficiency: build_context vs baselines

| Benchmark family | Scale | Method | Tokens returned | Score |
|---|---:|---|---:|---:|
| BrowseComp-Plus-style | 32 | query_graph | 98 | 1.000 |
| BrowseComp-Plus-style | 32 | raw_context | 175 | 1.000 |
| BrowseComp-Plus-style | 32 | build_context | 193 | 1.000 |
| BrowseComp-Plus-style | 128 | query_graph | 98 | 1.000 |
| BrowseComp-Plus-style | 128 | raw_context | 175 | 1.000 |
| BrowseComp-Plus-style | 128 | build_context | 193 | 1.000 |
| CodeQA-style | 32 | query_graph | 178 | 1.000 |
| CodeQA-style | 32 | build_context | 378 | 1.000 |
| CodeQA-style | 32 | raw_context | 826 | 1.000 |
| CodeQA-style | 128 | query_graph | 178 | 1.000 |
| CodeQA-style | 128 | build_context | 535 | 1.000 |
| CodeQA-style | 128 | raw_context | 920 | 0.000 |
| OOLONG-Pairs-style | 32 | query_graph | 97 | 0.000 |
| OOLONG-Pairs-style | 32 | build_context | 411 | 1.000 |
| OOLONG-Pairs-style | 32 | raw_context | 611 | 1.000 |
| OOLONG-Pairs-style | 128 | query_graph | 98 | 0.000 |
| OOLONG-Pairs-style | 128 | build_context | 515 | 1.000 |
| OOLONG-Pairs-style | 128 | raw_context | 948 | 0.000 |
| S-NIAH-style | 32 | query_graph | 93 | 1.000 |
| S-NIAH-style | 32 | build_context | 251 | 1.000 |
| S-NIAH-style | 32 | raw_context | 572 | 1.000 |
| S-NIAH-style | 128 | query_graph | 93 | 1.000 |
| S-NIAH-style | 128 | build_context | 181 | 1.000 |
| S-NIAH-style | 128 | raw_context | 943 | 1.000 |
