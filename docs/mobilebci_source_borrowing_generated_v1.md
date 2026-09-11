# Source-expert borrowing: generated-only mechanism gate

2026-09-11. Base2fa9b14. Candidate A only; trial-resolved coupling B is deferred because
feeding new features to the same closed diagonal prior would mainly be feature rescue.
No raw or numeric human cache access is authorized by this generated contract.

## Why this is a new operation, not a novelty claim

The closed context-template candidate reweighted a target person's own support and
was an exact no-op at k1. The closed head-motion candidate changed a target ridge penalty.
Here labeled EEG from different source participants creates class-specific decoders;
target support context selects whose prediction information to borrow. The target's
own decoder remains available as a no-transfer endpoint.

SS-iTRCA already selects source TRCs using EEG correlation and combines subject-general
and target-specific information (arXiv2506.10933v1, pp3–5, eqs6–21). Its target TRCA is
not a k1 operator and its all-low-similarity fallback includes all sources. Source
selection itself is not new, nor is metadata efficacy established by that paper.
Our transfer from that literature to external-M source routing is a falsifiable hypothesis.

## Frozen operator

Use class-specific reference-ridge W learned from each source's five generated trials/class,
λ=.1, identity regularization, covariance-scale normalization. Target W uses one/class.
References are 5.45/8.57/12Hz, three harmonics at128Hz for128samples in this fixture only.
Compute each expert's Gram-corrected projection energy exactly as in the verified primitive,
then softmax(10×score) over classes. A source-order-equivariant shared linear router
maps per-expert support-only features to logits; softmax over experts gives convex weights.
One weight vector is fixed for all queries in a generated episode. Never average W or raw
waveforms across people. Explicit no-transfer mode returns target probabilities directly.

Base router inputs: self-expert indicator, mean own-class support projection compatibility,
and mean squared Frobenius distance between trace-normalized target and expert support
covariances. These are not full support multiclass competence. Q2 adds standard deviation
and minimum own-class compatibility. QM adds two absolute differences of target/source
external context; SHAM uses the same two fields with a deterministic balanced mismatch.
Self-expert auxiliary differences are zero for Q2/QM/SHAM, matched across arms. All arms
share exactly the same expert bank, target support, queries and label resources.

This gate is not the final human Q feature contract. A human experiment would need the
existing richer Q/common information plus valid cache-computable transferability features.
The cache does not contain wrong-class support reference cross-products; querying outer
labels to fabricate support multiclass competence is prohibited.

## One constructive fixture and explicit limitations

Fixed seed20260911. Six channels form two context groups of three class-specific channels.
Two source banks have supervised class-specific signals in different groups. Target k1
support is deliberately ambiguous between groups. A query contains the true stimulus
in the context-selected group and a cyclic wrong-class decoy in the other. M reveals the
group in the positive condition; a balanced deterministic reassignment breaks its pairing
for SHAM. This is deliberately constructed informative side information, not a claim that
head gyro has this physiology or that real source motion predicts transferability.

Eight source-training episodes and eight disjoint generated evaluation episodes, balanced
contexts/classes; six queries/episode. Fit exactly four router heads Q/Q2/QM/SHAM, 100
full-batch AdamW steps each, lr=.05,weight_decay=.01,zero initialization,float64 CPU1.
Source-training-only standardization, population std floor1e−12. No early stopping,
hyperparameter search, alternate seed/generator, head restart or threshold change.

Before declaring gate PASS require all:

1. QM evaluation accuracy exceeds Q,Q2,SHAM and uniform borrowing by at least10pp.
2. Frozen QM with true M beats the same head with mismatched M by at least10pp.
3. Permuting source class assignments while keeping the fixed task references and true M
   degrades QM by at least10pp; otherwise transferred class information is unproven.
4. M-only change alters router weights and relative class margins by >1e−5 at k1.
5. Source permutation equivariance and within-class reference-basis orthogonal rotation
   invariance within1e−9; these are representation checks, not all real latency shifts.
6. Exact no-transfer return; invalid negative/nonfinite weights and self/held-out source
   inclusion rejected; routing consumes no query M or query labels.

Budget: implementation/generated unit suites at most3calls, no fixture>200000numeric
elements or2MiB; one actual capacity experiment with4routerfits/400steps, ≤120s CPUwall,
≤8MiB artifacts, no newdependencies/GPU/network/raw/humanfits/outcomes. Source expert
ridge solves are recorded separately from router fits. First unexpected failure stops and
is preserved, no scientific rescue. Gate failure closes this implementation candidate
before human efficacy. Passing only permits drafting a new finite human contract; it does
not itself authorize or prove human metadata usefulness.

Root sole writer, reviewers read-only. Closed336-fit result/old failures/ChoiPARKED/held60
remain intact. No overall research-goal success is inferred from a generated positive.
