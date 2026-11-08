# QuantumImageProcessing: FRQI/NEQR Encoding & 13-Qubit Pauli-Graph QHED Reference

This repository contains two distinct open-source artifacts for quantum image representation and operator-space edge detection:

1. **[`NEQR_FRQI.ipynb`](./NEQR_FRQI.ipynb)** (Precursor Material): Generalized Qiskit implementations of **Flexible Representation of Quantum Images (FRQI)** and **Novel Enhanced Quantum Representation (NEQR)** encoding, decoding, and measurement-based image reconstruction.
2. **[`Pauli_Graph_QHED_Reference.ipynb`](./Pauli_Graph_QHED_Reference.ipynb)** & **[`pauli_graph_qhed.py`](./pauli_graph_qhed.py)**: An independently specified, gate-level **13-qubit Pauli-graph Quantum Hadamard Edge Detection (QHED)** reference implementation and classical baseline validation following Section VII of:
   > Syed Emad Uddin Shubha and Tasnuva Farheen, *"Distinguishing Coherent Crosstalk from Calibration Drift via Pauli-Transfer Signatures and Quantum Edge Detection,"* IEEE International Symposium on Hardware Oriented Security and Trust (HOST), 2027 ([arXiv:2609.40283](https://arxiv.org/abs/2609.40283)).

---

## 1. Mathematical Construction (13-Qubit Pauli-Graph QHED)

For $n = 3$ physical qubits and Pauli weight cutoff $k = 2$, the number of low-weight Pauli strings is

$$K = \sum_{w=0}^{2} 3^w \binom{3}{w} = 1 + 9 + 27 = 37,$$

yielding $K^2 = 1,369$ Pauli transfer matrix (PTM) feature coordinates. Zero-padding to $N_{\mathrm{pad}} = 2,048 = 2^{11}$ addresses gives a **13-qubit** register ($11\text{ address} + 1\text{ pixel} + 1\text{ auxiliary}$, $8,192$ statevector amplitudes):

| Graph | Vertices | Undirected Edges | Greedy Matchings | 11-Controlled-$X$ Primitives |
|---|---:|---:|---:|---:|
| Hamming-1 ($G_H$) | 1,369 | 9,324 | 18 | 62,400 |
| Anticommuting-factor ($G_C$) | 1,369 | 4,662 | 9 | 30,638 |

### FRQI Loading & Postselection
Given a nonnegative, nonzero feature vector $S \in \mathbb{R}_{\ge 0}^{1369}$, define $\widetilde{S}_j = S_j / \max_k S_k$. Applying Hadamards on the 11 address qubits followed by address-controlled rotations $R_y(2\arcsin \widetilde{S}_j) = e^{-i\arcsin(\widetilde{S}_j)Y}$ prepares

$$\frac{1}{\sqrt{N_{\mathrm{pad}}}} \sum_{j=0}^{N_{\mathrm{pad}}-1} |j\rangle \left(\sqrt{1 - \widetilde{S}_j^2}\,|0\rangle + \widetilde{S}_j\,|1\rangle\right)|0\rangle_{\mathrm{aux}}.$$

Conditioning on pixel outcome $|1\rangle$ prepares normalized address amplitudes $a_j = S_j / \|S\|_2$ with success probability $p_{\mathrm{load}} = \sum_j \widetilde{S}_j^2 / N_{\mathrm{pad}}$.

### Matching Involutions & Normalized Dirichlet Energy
Each matching $M_\ell$ defines an address involution $D_\ell$ that swaps paired addresses $(u \leftrightarrow v) \in M_\ell$ and fixes all unpaired and padded addresses. Expanding each swap at binary Hamming distance $d = \operatorname{wt}(u \oplus v)$ along a binary Gray path uses $2d - 1$ open-controlled $X$ primitives (11 controls: `aux = 1` plus the 10 unchanged address bits). Applying $H_{\mathrm{aux}}, \operatorname{controlled-}D_\ell, H_{\mathrm{aux}}$ yields auxiliary-$1$ probability

$$P_\ell = \frac{1}{4}\sum_{j=0}^{N_{\mathrm{pad}}-1} |a_j - a_{D_\ell(j)}|^2 = \frac{1}{2}\sum_{\{u,v\}\in M_\ell} |a_u - a_v|^2,$$

so summing over all matchings recovers the normalized Pauli-graph Dirichlet energy:

$$\mathcal{A}_{\mathrm{edge}}^G(S) = \frac{1}{2}\sum_{\ell} P_\ell = \frac{S^{\mathsf{T}} L_G S}{4 S^{\mathsf{T}} S}.$$

---

## 2. Numerical Validation Summary

`pauli_graph_qhed.py` executes explicit Hadamard, open-controlled $R_y$, and Gray-path open-controlled $X$ primitives directly on the 8,192-amplitude statevector without invoking the classical edge formula, comparing against both a direct undirected edge-difference sum and a dense $1369 \times 1369$ Laplacian quadratic form across five three-qubit unitaries (local coherent drift, $ZX$, $YX$, $ZZ$, and a noncommuting combination) and three feature images (28 nonzero cases + 2 zero-image local-drift cross-weight cases):

- **Observed maximum circuit-vs-classical discrepancy (28 nonzero cases):** $1.78 \times 10^{-15}$ (edge sum) and $1.33 \times 10^{-15}$ (dense Laplacian).
- **Analytical $U = e^{-i(0.23)ZXI/2}$ cross-weight antisymmetric benchmark ($8$ nonzero pixels, $p_{\mathrm{load}} = 8/2048 = 1/256$):**
  - Hamming-1 ($G_H$): classical $3.750000000000000$, circuit $3.749999999999999$ ($|\Delta| = 8.88 \times 10^{-16}$), expected physical loader attempts for one sample per matching $= 4,608$.
  - Anticommuting-factor ($G_C$): classical $1.500000000000000$, circuit $1.4999999999999996$ ($|\Delta| = 4.44 \times 10^{-16}$), expected physical loader attempts for one sample per matching $= 2,304$.
- **Scope note:** The paper's reported $1.4 \times 10^{-14}$ maximum discrepancy applies to its original Figure 10 sweep; this reference validates independently specified examples at the same 13-qubit circuit width (see [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)).

---

## 3. Quick Start & Reproduction

```bash
pip install -r requirements.txt

# Run the unit and regression test suite
python3 -m unittest discover -s tests -p 'test_*.py' -v

# Re-execute all 8 notebook code cells and regenerate JSON validation records
python3 scripts/execute_reference_notebook.py
```

Optional Qiskit 4-qubit statevector cross-validation runs automatically when `requirements-optional-qiskit.txt` is installed.
