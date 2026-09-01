  # Abuse-Ring Sentinel

Built for the Razorpay Buildathon — Track 02: AI Risk Manager.

## What it solves

Coordinated abuse rings — groups of fake accounts working together to
farm promos, run refund scams, or commit fraud — are hard to catch
one account at a time. A single ring member often looks completely
normal on their own. The pattern only becomes visible when you look
at *relationships between accounts*: shared devices, shared payment
methods, shared addresses.

This project detects abuse rings using graph-based clustering, while
specifically protecting innocent overlaps (like a family sharing a
home Wi-Fi network) from being falsely flagged.

## Why this approach

Coming from a cybersecurity background rather than fintech, this
problem maps directly onto something more familiar: catching
coordinated, adversarial behavior that tries to look independent —
the same underlying pattern as detecting botnets or sybil attacks.
Graph clustering is the natural tool for that: individual accounts
don't reveal fraud, but their connections do.

## Architecture

| Layer | What it does |
|---|---|
| 1. Data | Generates 2,000 synthetic accounts — 95% normal (with some realistic innocent noise, e.g. households sharing IP/address), 5% split across 15 abuse rings sharing device_id/card in a staggered pattern (not every member shares with every other member) |
| 2. Graph | Builds a weighted graph — accounts as nodes, shared attributes as edges. Low weight (0.2) for IP/address (could be innocent), high weight (0.9) for device_id/card (rarely innocent) |
| 3. Detection | Finds connected clusters (`connected_components`), flags a cluster as a ring only if its *average edge weight* crosses a threshold — this is what protects a household of 5 from being flagged the same as a ring and [Ring visualization](ring_visualization.png)|
| 4. Evaluation | Compares flags against ground truth: precision, recall, F1, and false-positive/false-negative cost |

## Results

Precision: 1.000 (zero innocent accounts wrongly flagged)
Recall: 0.780 (caught 78 of 100 actual ring members)
F1 Score: 0.876
Accuracy: 0.989 (misleading alone — 95% of accounts are non-fraud anyway)

Estimated cost: $2,200 total
False positives: 0 ($0)
False negatives: 22 ($2,200 — all missed ring members, no false alarms)


*(Cost figures — $10/false positive, $100/false negative — are illustrative
assumptions for demonstrating cost-aware evaluation, not researched
real-world figures.)*

## What broke, and how I got out

The first version used a hardcoded rule ("flag any group of 4+ connected
accounts"). This would have falsely flagged legitimate households sharing
a home network. Fixed by moving from *group size* to *average edge weight*
as the flagging criterion — a group of 5 sharing only low-weight IP
connections passes safely, while a smaller group sharing high-weight
device/card connections gets flagged.

The detector still misses 22% of real ring members — not a tuning bug,
but a structural limit: because ring sharing is staggered (60-90% of
members, not all), a small fraction of members in every ring share
nothing with the rest of their ring, so they're invisible to a
connectivity-based detector by design. This is disclosed rather than
hidden, since it's an honest property of graph-based detection, not
something more tuning would fix.

## How to run

```bash
python generate_dataset.py   # produces accounts.csv, ground_truth.csv
python build_graph.py        # produces account_graph.gexf
python detect_rings.py       # produces predictions.csv
python evaluate.py           # prints + saves evaluation_summary.csv
```
Requires: `pandas`, `networkx`. Fixed random seed (42) for reproducibility.

## Tool use disclosure

Built with AI assistance for scaffolding, debugging, and code review.
The dataset design (staggered sharing, innocent-noise simulation), the
choice of weighted-average threshold over a hardcoded size rule, and
the interpretation of the results above are my own decisions.
