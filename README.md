# A Clear Tutorial on Markov Chains

## 1. What problem are we solving?

Imagine a system that moves between a fixed set of **states**, one step at a
time, and where the next state only depends on the *current* state — not on
the whole history that got you there. That's a **Markov chain**. It's one of
the simplest and most useful models in all of probability: it shows up in
weather forecasting, board games, text generation, Google's PageRank,
population genetics, queueing systems, and — as in the accompanying demo —
predicting an opponent's next move in a fight.

## 2. The building blocks

**States.** A finite list of possible situations the system can be in.
Example: `{Sunny, Rainy}`, or `{Punch, Kick, Falcon Punch}`.

**Transition probabilities.** For every pair of states `i` and `j`, a number
`P(i → j)` = the probability of moving to state `j` given that you are
currently in state `i`.

**The Markov property.** The probability of the next state depends *only* on
the current state:

```
P(next state | current state, all past states) = P(next state | current state)
```

This is what makes the chain "memoryless" — and it's a modeling choice, not a
law of nature. A lot of the art of using Markov chains is choosing states rich
enough that this assumption is reasonable (e.g. "last 3 words typed" instead
of just "last word typed" for a text predictor).

## 3. The transition matrix

If there are `n` states, we collect all the transition probabilities into an
`n × n` matrix `P`, where row `i`, column `j` is `P(i → j)`:

```
        to: S1    S2    S3
from S1  [ 0.9   0.05  0.05 ]
from S2  [ 0.1   0.8   0.1  ]
from S3  [ 0.2   0.2   0.6  ]
```

Two things must always be true of a valid transition matrix:
- Every entry is between 0 and 1.
- **Every row sums to 1** — from any state, you have to go *somewhere*
  (possibly back to the same state).

If you also have a **distribution row-vector** `x` describing how likely you
are to be in each state right now (e.g. `x = [0.5, 0.3, 0.2]`), then your
distribution one step later is:

```
x_next = x @ P
```

...and `k` steps later:

```
x_after_k_steps = x @ P^k
```

This is the single most important formula in this tutorial. `P^k` (matrix
power, *not* element-wise power) tells you everything about how the system
evolves over `k` steps. Row `i` of `P^k` is exactly the distribution over
states you'd end up in, `k` steps later, if you started in state `i` for sure.

## 4. Where do the numbers come from?

Two common ways:
1. **You design them** (as in the ninja demo — you decide "this fighter
   repeats punches 90% of the time").
2. **You estimate them from data**: count how often each transition actually
   happened, and normalize each row to sum to 1. This is exactly how simple
   text-prediction ("Markov chain generators") and many real-world models are
   built.

## 5. Does the chain settle down? Regular vs. periodic chains

A natural question: if you run the chain forever, does the distribution over
states converge to something stable, no matter where you started?

- **Regular chain**: some power `P^k` has *all strictly positive* entries.
  Regular chains are guaranteed to converge to a single **stationary
  distribution** `π`, and it doesn't matter what state you started in.
- **Periodic / non-regular chain**: the chain cycles instead of settling. The
  simplest example is `P = [[0, 1], [1, 0]]` — it just flips forever between
  two states, and `P^k` alternates between the identity matrix and its
  reverse without ever converging.

This distinction matters a lot in practice — e.g. PageRank's underlying chain
is specifically engineered (with a "damping factor") to be regular, precisely
so that the ranking converges to one answer instead of oscillating.

## 6. The stationary distribution

A **stationary distribution** `π` is a row-vector satisfying:

```
π @ P = π         (and the entries of π sum to 1)
```

In words: if you're already distributed according to `π`, applying one more
step of the chain leaves the distribution unchanged. For a regular chain,
`π` is unique, and:

```
lim (k → ∞) x @ P^k = π      for ANY starting distribution x
```

### How to actually compute `π`

**Method 1 — brute force (what the demo does first).** Just raise `P` to a
large power and watch it converge; every row of `P^k` becomes (approximately)
`π`.

```python
import numpy as np
P_100 = np.linalg.matrix_power(P, 100)
print(P_100)   # every row ≈ the stationary distribution
```

**Method 2 — eigenvectors (exact, faster, "the real math").** Rearranging
`π @ P = π` as `Pᵀ @ πᵀ = πᵀ` shows that `πᵀ` is an eigenvector of `Pᵀ` with
eigenvalue exactly `1`. Every regular transition matrix has exactly one such
eigenvalue-1 eigenvector, so you find it, then normalize it to sum to 1:

```python
evals, evecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(evals - 1))     # the eigenvalue closest to 1
pi = np.real(evecs[:, idx])
pi = pi / pi.sum()                     # normalize
```

This is exactly the `eig(P')` trick used in the original MATLAB demo, just
written in `numpy`.

## 7. A worked mini-example

Weather model with two states, `Sunny` and `Rainy`:

```
P = [ [0.9, 0.1],      # from Sunny: 90% stay sunny, 10% rain
      [0.5, 0.5] ]     # from Rainy: 50/50 either way
```

Row sums are 1, good. Since every entry is already positive, this chain is
regular after just one step. Solving `π @ P = π`:

```
π1 * 0.9 + π2 * 0.5 = π1
π1 * 0.1 + π2 * 0.5 = π2
π1 + π2 = 1
```

Solving gives `π ≈ [0.833, 0.167]` — in the long run, it's sunny about 5/6 of
the time and rainy 1/6 of the time, regardless of today's weather. (The
`markov_chain_new_example.ipynb` notebook uses a slightly richer 3-state
weather chain and confirms this numerically.)

## 8. Beyond "does it converge?": absorbing states

Some chains have a state that, once entered, you can *never leave* (e.g. `KO`
in the ninja combo demo, or `Churned` in a subscription business). These are
called **absorbing states**, and a chain containing at least one is called an
**absorbing Markov chain**. These chains don't have a single interesting
stationary distribution in the usual sense — instead the natural questions
become:

- What's the probability of eventually being absorbed into each absorbing
  state, starting from each non-absorbing state?
- What's the *expected number of steps* until absorption?

Both questions have clean closed-form answers using something called the
**fundamental matrix** `N = (I − Q)⁻¹`, where `Q` is the sub-matrix of `P`
restricted to the non-absorbing ("transient") states. The
`markov_chain_new_example.ipynb` notebook works through this with a customer
subscription-churn example.

## 9. Summary cheat-sheet

| Concept | Formula / Check |
|---|---|
| Valid transition matrix | every row sums to 1, all entries ≥ 0 |
| Distribution after k steps | `x @ P^k` |
| Regular chain | some `P^k` has all-positive entries |
| Stationary distribution | solves `π @ P = π`, `sum(π) = 1` |
| Find `π` via eigenvectors | eigenvector of `Pᵀ` for eigenvalue 1, normalized |
| Absorbing state | a state `s` with `P(s → s) = 1` |
| Absorption probabilities | via fundamental matrix `N = (I − Q)⁻¹` |

## 10. Where to go next

- `markov_chain_colab.ipynb` — runs the original ninja-fighting-style demo in
  Python, with animated diagrams.
- `markov_chain_new_example.ipynb` — a fresh, non-ninja example (subscription
  churn) that also introduces absorbing chains.
- `markov_chain_pretty_graphs.py` — a polished, reusable graphing script for
  visualizing any transition matrix (heatmap, network diagram, convergence
  plot, and stationary-distribution bar chart in one figure).
