# Reproducibility & Scope Record

This document records the deterministic conventions, primitive resource definitions, execution environment, and scope boundaries for the 13-qubit Pauli-graph QHED reference in [`pauli_graph_qhed.py`](./pauli_graph_qhed.py) and [`Pauli_Graph_QHED_Reference.ipynb`](./Pauli_Graph_QHED_Reference.ipynb).

---

## 1. Deterministic Ordering & Decomposition Choices

1. **Low-weight Pauli basis ($K = 37$):** Generated in Cartesian product order over `("I", "X", "Y", "Z")` on $n = 3$ qubits filtered by Pauli weight $w \le 2$.
2. **PTM coordinate flattening ($K^2 = 1,369$):** Row-major index $u = 37 i_\alpha + i_\beta \in \{0, \dots, 1368\}$ for ordered pair $(P_{i_\alpha}, P_{i_\beta})$, zero-padded at addresses $1369 \dots 2047$ ($N_{\mathrm{pad}} = 2^{11} = 2048$).
3. **Undirected edge ordering:** Ascending lexicographic pairs $(u, v)$ with $0 \le u < v < 1369$.
4. **Matching decomposition:** Sequential greedy decomposition over the sorted edge list (`greedy_matching_decomposition`), producing $18$ matchings on $G_H$ ($9,324$ edges) and $9$ matchings on $G_C$ ($4,662$ edges). Note that on $G_C$, the weight-$2 \times 2$ components have $3^4 = 81$ (odd) vertices and degree $8$ ($324$ edges), so $\lceil 324 / \lfloor 81/2 \rfloor \rceil = \lceil 324 / 40 \rceil = 9$ matchings is also the chromatic-index lower bound.
5. **Binary Gray-path controlled transpositions:** Each pair $(u, v)$ at binary Hamming distance $d = \operatorname{wt}(u \oplus v)$ is expanded into $2d - 1$ 11-controlled-$X$ primitives (`aux = 1` plus the $10$ unchanged address bits along the LSB-to-MSB Gray walk).
6. **Roundoff cleanup:** Extracted feature magnitudes below `1e-14` are set to `0.0` before passing the identical array to both the circuit compiler and the classical baselines.

---

## 2. Statevector Simulation vs. Physical Sampling Costs

- **Gate-level statevector execution:** `execute_pauli_graph_qhed_circuit` prepares the postselected 13-qubit state once in memory and applies each matching circuit sequentially to verify exact circuit realization against the classical graph quadratic form.
- **Physical hardware sampling:** Because measurement collapses the quantum state and FRQI pixel-$1$ postselection succeeds with probability $p_{\mathrm{load}} = \sum_j \widetilde{S}_j^2 / N_{\mathrm{pad}}$, collecting $N_{\mathrm{shots}}$ postselected samples per matching across $L$ matchings requires $L \cdot N_{\mathrm{shots}} / p_{\mathrm{load}}$ expected loader trials, in addition to native multi-controlled gate synthesis, qubit routing, and finite-shot binomial variance.
- **Boundary conditions:**
  - When $S = 0$ (as occurs for the cross-weight antisymmetric feature under product local unital drift), $p_{\mathrm{load}} = 0$ and the normalized score $\mathcal{A}_{\mathrm{edge}}^G(S)$ is undefined.
  - When $S$ is a positive constant across all $1,369$ graph vertices, $\mathcal{A}_{\mathrm{edge}}^G(S) = 0$ even though padded addresses $1369 \dots 2047$ hold zero amplitude, because every matching involution fixes all padded addresses.

---

## 3. Relationship to HOST 2027 Manuscript (`arXiv:2609.40283`)

- The manuscript reports a 13-qubit gate-level statevector verification across its evaluated channel and feature sweep with a maximum error of $1.4 \times 10^{-14}$ (Figure 10).
- Because the original Figure 10 input manifest and per-case numerical arrays are not archived in precursor repository commits, this reference does not claim to replay Figure 10 literally. Instead, it evaluates five explicitly specified 3-qubit unitaries (`local_drift`, `zx_crosstalk`, `yx_crosstalk`, `zz_crosstalk`, `noncommuting_combo`) across three feature images and both Pauli graphs (28 nonzero + 2 zero-image cases), recording all parameters and outputs in [`results/pauli_graph_qhed_validation.json`](./results/pauli_graph_qhed_validation.json) and [`results/notebook_execution_record.json`](./results/notebook_execution_record.json).
