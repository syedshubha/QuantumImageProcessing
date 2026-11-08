"""Build and execute Pauli_Graph_QHED_Reference.ipynb sequentially, recording outputs and JSON artifacts."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pauli_graph_qhed as qhed  # noqa: E402


def _md_cell(cell_id: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }


def _code_cell(cell_id: str, code: str) -> dict:
    lines = code.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook_template() -> dict:
    cells = [
        _md_cell(
            "md-intro",
            """# 13-Qubit Pauli-Graph Quantum Hadamard Edge Detection (QHED) Reference

This notebook provides an independently specified, gate-level 13-qubit reference implementation of the **FRQI + Pauli-graph Quantum Hadamard Edge Detection (QHED)** construction from Section VII of:

> Syed Emad Uddin Shubha and Tasnuva Farheen, *"Distinguishing Coherent Crosstalk from Calibration Drift via Pauli-Transfer Signatures and Quantum Edge Detection,"* IEEE HOST 2027 (`arXiv:2609.40283`).

### Scope and Distinction from Precursor Material
- **`NEQR_FRQI.ipynb`** in this repository is earlier precursor material demonstrating general FRQI and NEQR image encoding, decoding, and measurement-based image reconstruction.
- **This notebook (`Pauli_Graph_QHED_Reference.ipynb`)** implements the 13-qubit operator-space Pauli-graph edge-readout circuit ($11$ address + $1$ pixel + $1$ auxiliary qubit, $8,192$ complex statevector amplitudes) and verifies it against two independent classical baselines across five explicitly specified three-qubit unitary channels. It validates the circuit identity on independently specified examples rather than replaying the original manuscript's Figure 10 sweep.
""",
        ),
        _code_cell(
            "code-01-imports",
            """import platform
import numpy as np
import pauli_graph_qhed as qhed

labels = qhed.low_weight_pauli_labels(n_qubits=3, max_weight=2)
print(f"Python: {platform.python_version()} | NumPy: {np.__version__}")
print(f"Physical qubits n = 3, max Pauli weight k = 2 -> K = {len(labels)} low-weight Paulis")
print(f"Feature coordinates K^2 = {len(labels)**2}, padded to N_pad = 2048 (11 address + 1 pixel + 1 aux = 13 qubits)")
print("First 10 ordered Pauli labels:", labels[:10])
""",
        ),
        _md_cell(
            "md-graphs",
            """## 1. Pauli Graphs and Deterministic Matching Decompositions

Each pixel $j \\in \\{0, \\dots, 1368\\}$ indexes an ordered pair $(P_\\alpha, P_\\beta)$ in the $37 \\times 37$ low-weight Pauli transfer matrix (PTM) block:
- **Hamming-1 graph ($G_H$):** connects coordinates where one Pauli label is unchanged and the other differs in exactly one tensor factor.
- **Anticommuting-factor graph ($G_C$):** restricts $G_H$ to pairs whose differing single-qubit tensor factors anticommute.
""",
        ),
        _code_cell(
            "code-02-graphs",
            """graphs = qhed.build_pauli_graphs(labels)
for name, spec in graphs.items():
    cx_count = sum(2 * ((u ^ v).bit_count()) - 1 for u, v in spec.edges)
    print(
        f"{name:22s} | vertices={spec.n_vertices:4d} | undirected_edges={len(spec.edges):5d} "
        f"| matchings={len(spec.matchings):2d} | 11-controlled-X={cx_count:5d}"
    )
""",
        ),
        _md_cell(
            "md-zx-walkthrough",
            """## 2. Analytical & Gate-Level Walkthrough: $U = e^{-i(0.23)ZXI/2}$

For a nonnegative, nonzero feature vector $S$ normalized by $\\widetilde{S}_j = S_j / \\max_k S_k$, the FRQI loader applies $R_y(2\\arcsin\\widetilde{S}_j)$ conditioned on each 11-bit address $|j\\rangle$. Conditioning on pixel outcome $|1\\rangle$ prepares $a_j = S_j / \\|S\\|_2$ with probability $p_{\\mathrm{load}} = \\sum_j \\widetilde{S}_j^2 / N_{\\mathrm{pad}}$.

For each matching $M_\\ell$, an address involution $D_\\ell$ swaps paired addresses along binary Gray paths ($2d - 1$ controlled-$X$ primitives for Hamming distance $d$). Applying $H_{\\mathrm{aux}}, \\text{controlled-}D_\\ell, H_{\\mathrm{aux}}$ yields auxiliary-$1$ probability $P_\\ell = \\frac{1}{2}\\sum_{\\{u,v\\}\\in M_\\ell} |a_u - a_v|^2$, so:
$$\\mathcal{A}_{\\mathrm{edge}}^G(S) = \\frac{1}{2}\\sum_\\ell P_\\ell = \\frac{S^{\\mathsf{T}} L_G S}{4 S^{\\mathsf{T}} S}.$$
""",
        ),
        _code_cell(
            "code-03-zx-example",
            """unitaries = qhed.reference_unitary_examples()
ptm_zx = qhed.low_weight_ptm(unitaries["zx_crosstalk"]["unitary"], labels)
feat_zx = qhed.extract_feature_images(ptm_zx, labels)["cross_weight_antisymmetric"]

print(f"Nonzero pixels in ZX cross-weight antisymmetric image: {int(np.count_nonzero(feat_zx))}")
for g_name, g_spec in graphs.items():
    cl_edge = qhed.classical_edge_sum_score(feat_zx, g_spec.edges)
    cl_lap = qhed.classical_laplacian_score(feat_zx, g_spec.laplacian)
    res = qhed.execute_pauli_graph_qhed_circuit(feat_zx, g_spec, n_address_qubits=11)
    print(
        f"{g_name:22s} | classical={cl_edge:.15f} | laplacian={cl_lap:.15f} "
        f"| circuit={res.edge_score:.15f} | |diff|={abs(res.edge_score - cl_edge):.3e}"
    )
""",
        ),
        _code_cell(
            "code-04-zx-resources",
            """for g_name, g_spec in graphs.items():
    res = qhed.execute_pauli_graph_qhed_circuit(feat_zx, g_spec, n_address_qubits=11)
    print(
        f"{g_name:22s} | width={res.n_qubits} | loader_H={res.loader_hadamards} | loader_mcry={res.loader_controlled_ry} "
        f"| p_load={res.p_load:.8f} (1/{round(1.0 / res.p_load)}) | matchings={res.matching_circuits} "
        f"| aux_H={res.aux_hadamards} | 11-ctrl-X={res.controlled_x_primitives} "
        f"| expected_attempts={res.expected_loader_attempts:.1f}"
    )
""",
        ),
        _md_cell(
            "md-full-suite",
            """## 3. Full Five-Unitary Benchmark Suite (28 Nonzero + 2 Zero-Image Cases)

We evaluate five 3-qubit unitaries (local drift, $ZX$, $YX$, $ZZ$, and a noncommuting combination) across three low-weight PTM feature images and both Pauli graphs. Under local coherent drift, the cross-weight antisymmetric feature vanishes identically by Proposition 1, producing the two zero-image cases ($p_{\\mathrm{load}} = 0$).
""",
        ),
        _code_cell(
            "code-05-full-benchmark",
            """benchmark = qhed.run_reference_benchmark_suite()
print(f"Total cases: {benchmark['total_cases']} ({benchmark['nonzero_cases']} nonzero, {benchmark['zero_image_cases']} zero-image)")
print(f"Max |circuit - classical edge sum| : {benchmark['max_abs_diff_circuit_vs_edge_sum']:.16e}")
print(f"Max |circuit - classical Laplacian|: {benchmark['max_abs_diff_circuit_vs_laplacian']:.16e}")
""",
        ),
        _code_cell(
            "code-06-case-table",
            """print(f"{'Unitary':18s} | {'Feature':26s} | {'Graph':20s} | {'Circuit Score':>16s} | {'|Diff|':>10s}")
print("-" * 102)
for row in benchmark["cases"]:
    if row["circuit_score"] is None:
        print(f"{row['unitary']:18s} | {row['feature']:26s} | {row['graph']:20s} | {'undefined (S=0)':>16s} | {'n/a':>10s}")
    else:
        print(
            f"{row['unitary']:18s} | {row['feature']:26s} | {row['graph']:20s} "
            f"| {row['circuit_score']:16.12f} | {row['abs_diff_edge_sum']:10.2e}"
        )
""",
        ),
        _md_cell(
            "md-boundary",
            """## 4. Boundary Conditions and Optional Qiskit Small-Width Cross-Check

We verify two distinct boundary behaviors:
1. **$S = 0$ (zero feature image):** pixel-$1$ postselection has probability $p_{\\mathrm{load}} = 0$, so the normalized score is undefined.
2. **Constant $S > 0$ across all $1,369$ graph vertices:** every graph edge connects equal amplitudes, so $\\mathcal{A}_{\\mathrm{edge}}^G(S) = 0$ even though padded addresses $1369 \\dots 2047$ hold zero amplitude (because matching involutions fix all padded addresses).
""",
        ),
        _code_cell(
            "code-07-boundary-cases",
            """zero_vec = np.zeros(len(labels) ** 2, dtype=np.float64)
zero_res = qhed.execute_pauli_graph_qhed_circuit(zero_vec, graphs["hamming_1"])
print("Zero feature S = 0 -> p_load =", zero_res.p_load, "| edge_score =", zero_res.edge_score)

const_vec = np.full(len(labels) ** 2, 0.5, dtype=np.float64)
for g_name, g_spec in graphs.items():
    c_res = qhed.execute_pauli_graph_qhed_circuit(const_vec, g_spec)
    c_cl = qhed.classical_edge_sum_score(const_vec, g_spec.edges)
    print(f"Constant feature S > 0 ({g_name:20s}) -> classical={c_cl:.16f} | circuit={c_res.edge_score:.16e}")
""",
        ),
        _code_cell(
            "code-08-qiskit-crosscheck",
            """qk_info = qhed.qiskit_small_width_crosscheck()
if qk_info is None:
    print("Optional Qiskit package not installed; core NumPy reference verified.")
else:
    print(
        f"Optional 4-qubit Qiskit {qk_info['qiskit_version']} statevector check: "
        f"loader max |Δψ| = {qk_info['loader_max_amplitude_diff']:.3e}, "
        f"full circuit max |Δψ| = {qk_info['full_circuit_max_amplitude_diff']:.3e}"
    )
""",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": platform.python_version(),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build_notebook_template()
    namespace: dict = {"__name__": "__main__"}
    exec_records = []
    exec_num = 0
    t_start = time.perf_counter()

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec_num += 1
        code_str = "".join(cell["source"])
        buf = io.StringIO()
        cell_t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code_str, f"<notebook-cell-{cell['id']}>", "exec"), namespace)
        except Exception as exc:
            tb = traceback.format_exc()
            raise RuntimeError(f"Error executing {cell['id']}: {exc}\n{tb}") from exc
        elapsed_s = time.perf_counter() - cell_t0
        stdout_text = buf.getvalue()
        cell["execution_count"] = exec_num
        cell["outputs"] = (
            [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_text.splitlines(keepends=True),
                }
            ]
            if stdout_text
            else []
        )
        exec_records.append(
            {
                "execution_count": exec_num,
                "cell_id": cell["id"],
                "status": "ok",
                "elapsed_seconds": round(elapsed_s, 6),
                "stdout_lines": len(stdout_text.splitlines()),
            }
        )

    total_elapsed = time.perf_counter() - t_start
    nb_path = REPO_ROOT / "Pauli_Graph_QHED_Reference.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    validation_payload = dict(namespace["benchmark"])
    validation_payload["environment"] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_crosscheck": namespace.get("qk_info"),
    }
    (results_dir / "pauli_graph_qhed_validation.json").write_text(
        json.dumps(validation_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    execution_record = {
        "notebook": "Pauli_Graph_QHED_Reference.ipynb",
        "runner": "scripts/execute_reference_notebook.py",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "executed_code_cells": exec_num,
        "errors": 0,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "cells": exec_records,
    }
    (results_dir / "notebook_execution_record.json").write_text(
        json.dumps(execution_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Successfully executed {exec_num} code cells in {total_elapsed:.2f}s and wrote JSON records.")


if __name__ == "__main__":
    main()
