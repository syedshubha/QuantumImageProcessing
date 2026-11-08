"""Reproducible 13-qubit Pauli-graph Quantum Hadamard Edge Detection (QHED) reference.

Implements the mathematical construction described in Section VII of:
  Syed Emad Uddin Shubha and Tasnuva Farheen,
  "Distinguishing Coherent Crosstalk from Calibration Drift via Pauli-Transfer
   Signatures and Quantum Edge Detection," IEEE HOST 2027 (arXiv:2609.40283).

Scope and conventions:
- Physical system size n = 3, Pauli weight cutoff k = 2 -> K = 37 low-weight Paulis
  ordered deterministically in Cartesian product order over ("I", "X", "Y", "Z").
- Feature image size K^2 = 1,369 PTM coordinates flattened row-major and zero-padded
  to N_pad = 2,048 = 2^11 addresses -> 11 address + 1 pixel + 1 auxiliary = 13 qubits
  (8,192 complex statevector amplitudes).
- Two Pauli graphs on the 1,369 PTM coordinates:
    * Hamming-1 graph G_H: 1,369 vertices, 9,324 undirected edges, 18 greedy matchings.
    * Anticommuting-factor graph G_C: 1,369 vertices, 4,662 undirected edges, 9 greedy matchings.
- Gate-level circuit execution applies explicit Hadamard, open-controlled R_y, and
  Gray-path open-controlled X primitives directly to the 8,192-amplitude statevector
  without invoking the classical edge-energy formula.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PAULI_LABELS_1Q: Tuple[str, ...] = ("I", "X", "Y", "Z")
ROUNDOFF_CUTOFF: float = 1e-14


def single_qubit_pauli_matrices() -> Dict[str, np.ndarray]:
    """Return standard 2x2 single-qubit Pauli matrices."""
    return {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }


def pauli_weight(label: str) -> int:
    """Count non-identity tensor factors in a Pauli string."""
    return sum(1 for ch in label if ch != "I")


def low_weight_pauli_labels(n_qubits: int = 3, max_weight: int = 2) -> Tuple[str, ...]:
    """Return low-weight n-qubit Pauli strings in deterministic product order over I, X, Y, Z."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if max_weight < 0 or max_weight > n_qubits:
        raise ValueError("max_weight must satisfy 0 <= max_weight <= n_qubits")
    return tuple(
        "".join(chars)
        for chars in itertools.product(PAULI_LABELS_1Q, repeat=n_qubits)
        if sum(1 for ch in chars if ch != "I") <= max_weight
    )


def pauli_string_matrix(label: str) -> np.ndarray:
    """Construct the 2^n x 2^n matrix for a Pauli tensor-product label."""
    p1 = single_qubit_pauli_matrices()
    mat = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for ch in label:
        mat = np.kron(mat, p1[ch])
    return mat


def rotation_gate_1q(axis: str, angle: float) -> np.ndarray:
    """Return exp(-i * angle * P / 2) for P in {X, Y, Z}."""
    p = single_qubit_pauli_matrices()[axis]
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return c * np.eye(2, dtype=np.complex128) - 1.0j * s * p


def unitary_from_hermitian_generator(hermitian: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Compute exp(-i * scale * H / 2) via exact Hermitian eigendecomposition."""
    herm = np.asarray(hermitian, dtype=np.complex128)
    if not np.allclose(herm, herm.conj().T, atol=1e-14):
        raise ValueError("Generator must be Hermitian")
    eigvals, eigvecs = np.linalg.eigh(herm)
    phases = np.exp(-0.5j * scale * eigvals)
    return (eigvecs * phases) @ eigvecs.conj().T


def reference_unitary_examples() -> Dict[str, Dict[str, object]]:
    """Return five explicitly specified 3-qubit unitary test channels."""
    rx = rotation_gate_1q("X", 0.19)
    rz = rotation_gate_1q("Z", -0.14)
    ry = rotation_gate_1q("Y", 0.11)
    u_drift = np.kron(np.kron(rx, rz), ry)

    u_zx = unitary_from_hermitian_generator(pauli_string_matrix("ZXI"), scale=0.23)
    u_yx = unitary_from_hermitian_generator(pauli_string_matrix("YXI"), scale=0.21)
    u_zz = unitary_from_hermitian_generator(pauli_string_matrix("ZZI"), scale=0.25)

    h_combo = (
        0.18 * pauli_string_matrix("ZXI")
        + 0.13 * pauli_string_matrix("IYZ")
        + 0.09 * pauli_string_matrix("XXI")
    )
    u_combo = unitary_from_hermitian_generator(h_combo, scale=1.0)

    return {
        "local_drift": {
            "formula": "R_x(0.19) ⊗ R_z(-0.14) ⊗ R_y(0.11)",
            "unitary": u_drift,
        },
        "zx_crosstalk": {
            "formula": "exp(-i * 0.23 * ZXI / 2)",
            "unitary": u_zx,
        },
        "yx_crosstalk": {
            "formula": "exp(-i * 0.21 * YXI / 2)",
            "unitary": u_yx,
        },
        "zz_crosstalk": {
            "formula": "exp(-i * 0.25 * ZZI / 2)",
            "unitary": u_zz,
        },
        "noncommuting_combo": {
            "formula": "exp(-i * (0.18 ZXI + 0.13 IYZ + 0.09 XXI) / 2)",
            "unitary": u_combo,
        },
    }


def low_weight_ptm(
    unitary: np.ndarray,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute the K x K low-weight Pauli transfer matrix R_{a,b} = Tr(P_a U P_b U^†) / 2^n."""
    if labels is None:
        labels = low_weight_pauli_labels(3, 2)
    dim = unitary.shape[0]
    mats = [pauli_string_matrix(lbl) for lbl in labels]
    conjugated = [unitary @ m @ unitary.conj().T for m in mats]
    k_len = len(labels)
    ptm = np.empty((k_len, k_len), dtype=np.float64)
    for i, ma in enumerate(mats):
        for j, ub in enumerate(conjugated):
            # Both ma and ub are Hermitian, so Tr(ma @ ub) = sum(ma * ub.T) is real.
            ptm[i, j] = float(np.real(np.sum(ma * ub.T)) / dim)
    return ptm


def extract_feature_images(
    ptm: np.ndarray,
    labels: Sequence[str] | None = None,
    cutoff: float = ROUNDOFF_CUTOFF,
) -> Dict[str, np.ndarray]:
    """Construct three nonnegative K^2 feature vectors with documented roundoff cleanup.

    1. cross_weight_antisymmetric: |(R_{a,b} - R_{b,a}) / 2| restricted to {w(a), w(b)} = {1, 2}.
    2. low_weight_antisymmetric:   |(R_{a,b} - R_{b,a}) / 2| across all K x K low-weight pairs.
    3. off_diagonal_residual:      |(R - I)_{a,b}| on off-diagonal pairs a != b.
    """
    if labels is None:
        labels = low_weight_pauli_labels(3, 2)
    weights = [pauli_weight(lbl) for lbl in labels]
    k_len = len(labels)
    anti = 0.5 * (ptm - ptm.T)

    cross_anti = np.zeros((k_len, k_len), dtype=np.float64)
    for i in range(k_len):
        for j in range(k_len):
            if (weights[i] == 1 and weights[j] == 2) or (weights[i] == 2 and weights[j] == 1):
                cross_anti[i, j] = abs(anti[i, j])

    low_anti = np.abs(anti)
    residual = np.abs(ptm - np.eye(k_len, dtype=np.float64))
    np.fill_diagonal(residual, 0.0)

    out: Dict[str, np.ndarray] = {}
    for name, mat in (
        ("cross_weight_antisymmetric", cross_anti),
        ("low_weight_antisymmetric", low_anti),
        ("off_diagonal_residual", residual),
    ):
        vec = mat.reshape(-1).copy()
        vec[vec < cutoff] = 0.0
        out[name] = vec
    return out


@dataclass(frozen=True)
class PauliGraphSpec:
    """Specification of a Pauli graph and its deterministic matching decomposition."""

    name: str
    n_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    matchings: Tuple[Tuple[Tuple[int, int], ...], ...]
    laplacian: np.ndarray


def _single_factor_diff_site(label_a: str, label_b: str) -> int:
    """Return differing tensor site index if labels differ in exactly one site, else -1."""
    diff_site = -1
    for idx, (ca, cb) in enumerate(zip(label_a, label_b)):
        if ca != cb:
            if diff_site != -1:
                return -1
            diff_site = idx
    return diff_site


def greedy_matching_decomposition(
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Decompose sorted undirected edges deterministically into disjoint matchings."""
    remaining = list(edges)
    layers: List[Tuple[Tuple[int, int], ...]] = []
    while remaining:
        used = set()
        current: List[Tuple[int, int]] = []
        leftover: List[Tuple[int, int]] = []
        for u, v in remaining:
            if u not in used and v not in used:
                used.add(u)
                used.add(v)
                current.append((u, v))
            else:
                leftover.append((u, v))
        layers.append(tuple(current))
        remaining = leftover
    return tuple(layers)


def build_pauli_graphs(labels: Sequence[str] | None = None) -> Dict[str, PauliGraphSpec]:
    """Construct Hamming-1 and anticommuting-factor Pauli graphs on K^2 PTM coordinates."""
    if labels is None:
        labels = low_weight_pauli_labels(3, 2)
    k_len = len(labels)
    n_vertices = k_len * k_len

    edges_h: List[Tuple[int, int]] = []
    edges_c: List[Tuple[int, int]] = []

    for u in range(n_vertices):
        ia, ib = divmod(u, k_len)
        pa, pb = labels[ia], labels[ib]
        for v in range(u + 1, n_vertices):
            ja, jb = divmod(v, k_len)
            if ia == ja:
                site = _single_factor_diff_site(pb, labels[jb])
                if site >= 0:
                    edges_h.append((u, v))
                    if pb[site] != "I" and labels[jb][site] != "I":
                        edges_c.append((u, v))
            elif ib == jb:
                site = _single_factor_diff_site(pa, labels[ja])
                if site >= 0:
                    edges_h.append((u, v))
                    if pa[site] != "I" and labels[ja][site] != "I":
                        edges_c.append((u, v))

    specs: Dict[str, PauliGraphSpec] = {}
    for name, edge_list in (
        ("hamming_1", edges_h),
        ("anticommuting_factor", edges_c),
    ):
        lap = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for u, v in edge_list:
            lap[u, v] -= 1.0
            lap[v, u] -= 1.0
            lap[u, u] += 1.0
            lap[v, v] += 1.0
        specs[name] = PauliGraphSpec(
            name=name,
            n_vertices=n_vertices,
            edges=tuple(edge_list),
            matchings=greedy_matching_decomposition(edge_list),
            laplacian=lap,
        )
    return specs


def classical_edge_sum_score(feature: np.ndarray, edges: Sequence[Tuple[int, int]]) -> float:
    """Compute normalized edge score (1 / (4 ||S||_2^2)) sum_{(u,v) in E} |S_u - S_v|^2."""
    s = np.asarray(feature, dtype=np.float64).reshape(-1)
    norm_sq = float(np.dot(s, s))
    if norm_sq == 0.0:
        raise ValueError("Normalized QHED edge score is undefined for the zero feature S = 0.")
    if not edges:
        return 0.0
    edge_arr = np.asarray(edges, dtype=np.intp)
    diffs = s[edge_arr[:, 0]] - s[edge_arr[:, 1]]
    diff_sq = float(np.sum(diffs * diffs))
    return diff_sq / (4.0 * norm_sq)


def classical_laplacian_score(feature: np.ndarray, laplacian: np.ndarray) -> float:
    """Compute normalized quadratic form (S^T L_G S) / (4 S^T S) via dense Laplacian."""
    s = np.asarray(feature, dtype=np.float64).reshape(-1)
    norm_sq = float(np.dot(s, s))
    if norm_sq == 0.0:
        raise ValueError("Normalized Laplacian score is undefined for the zero feature S = 0.")
    return float(s @ (laplacian @ s)) / (4.0 * norm_sq)


@dataclass(frozen=True)
class PrimitiveGate:
    """Explicit gate primitive in {H, open-controlled R_y, open-controlled X}.

    Qubit indexing convention on total width M = m + 2:
      - qubits 0 .. m-1: address register (qubit b represents binary weight 2^b of address j)
      - qubit m:         FRQI pixel qubit
      - qubit m+1:       QHED auxiliary qubit
    """

    kind: str  # "h", "mcry", or "mcx"
    target: int
    controls: Tuple[Tuple[int, int], ...] = ()
    angle: float = 0.0


def expand_controlled_transposition_gray(
    u: int,
    v: int,
    n_address_qubits: int,
    aux_qubit: int,
) -> List[PrimitiveGate]:
    """Expand controlled address transposition (u <-> v) into 2d - 1 (m)-controlled X gates.

    Along the binary Gray path w_0 = u, w_1, ..., w_d = v, consecutive addresses w_t and w_{t+1}
    differ solely at bit b_t. Swapping |w_t> <-> |w_{t+1}> conditioned on aux == 1 is a single
    X gate on qubit b_t controlled on aux == 1 and the remaining m - 1 address bits of w_t.
    Conjugating the final step by the prefix steps gives 2d - 1 primitive gates.
    """
    if u == v:
        return []
    xor_val = u ^ v
    diff_bits = [b for b in range(n_address_qubits) if (xor_val >> b) & 1]
    d = len(diff_bits)

    step_gates: List[PrimitiveGate] = []
    cur = u
    for b_t in diff_bits:
        controls: List[Tuple[int, int]] = [(aux_qubit, 1)]
        for q in range(n_address_qubits):
            if q != b_t:
                controls.append((q, (cur >> q) & 1))
        step_gates.append(PrimitiveGate(kind="mcx", target=b_t, controls=tuple(controls)))
        cur ^= 1 << b_t

    return step_gates[:-1] + [step_gates[-1]] + list(reversed(step_gates[:-1]))


def compile_frqi_loader_gates(
    feature: np.ndarray,
    n_address_qubits: int = 11,
) -> Tuple[List[PrimitiveGate], float]:
    """Compile explicit FRQI state-preparation gates and theoretical loading probability p_load."""
    s = np.asarray(feature, dtype=np.float64).reshape(-1)
    if np.any(s < 0.0):
        raise ValueError("FRQI feature values must be nonnegative.")
    n_pad = 1 << n_address_qubits
    if s.size > n_pad:
        raise ValueError(f"Feature length {s.size} exceeds N_pad = {n_pad}.")

    pixel_qubit = n_address_qubits
    gates: List[PrimitiveGate] = [
        PrimitiveGate(kind="h", target=q) for q in range(n_address_qubits)
    ]

    max_val = float(np.max(s)) if s.size else 0.0
    if max_val == 0.0:
        return gates, 0.0

    s_tilde = s / max_val
    for addr, val in enumerate(s_tilde):
        if val > 0.0:
            phi = 2.0 * math.asin(min(1.0, float(val)))
            controls = tuple((q, (addr >> q) & 1) for q in range(n_address_qubits))
            gates.append(PrimitiveGate(kind="mcry", target=pixel_qubit, controls=controls, angle=phi))

    p_load_theory = float(np.sum(s_tilde**2)) / float(n_pad)
    return gates, p_load_theory


def compile_matching_readout_gates(
    matching: Sequence[Tuple[int, int]],
    n_address_qubits: int = 11,
) -> List[PrimitiveGate]:
    """Compile H_aux -> controlled-D_ell -> H_aux into primitive H and open-controlled X gates."""
    aux_qubit = n_address_qubits + 1
    gates: List[PrimitiveGate] = [PrimitiveGate(kind="h", target=aux_qubit)]
    for u, v in matching:
        gates.extend(expand_controlled_transposition_gray(u, v, n_address_qubits, aux_qubit))
    gates.append(PrimitiveGate(kind="h", target=aux_qubit))
    return gates


def apply_primitive_gates(
    state: np.ndarray,
    gates: Iterable[PrimitiveGate],
    n_qubits: int,
) -> np.ndarray:
    """Apply explicit H, open-controlled R_y, and open-controlled X gates to a 2^n statevector."""
    psi = np.array(state, dtype=np.complex128, copy=True).reshape((2,) * n_qubits)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for gate in gates:
        # Axis index in (2,)*n_qubits where qubit 0 is the least-significant bit (last axis).
        target_axis = n_qubits - 1 - gate.target
        idx0: List[int | slice] = [slice(None)] * n_qubits
        idx1: List[int | slice] = [slice(None)] * n_qubits
        for ctrl_q, ctrl_val in gate.controls:
            c_axis = n_qubits - 1 - ctrl_q
            idx0[c_axis] = ctrl_val
            idx1[c_axis] = ctrl_val
        idx0[target_axis] = 0
        idx1[target_axis] = 1
        t0 = tuple(idx0)
        t1 = tuple(idx1)

        if gate.kind == "h":
            a0 = psi[t0].copy()
            a1 = psi[t1].copy()
            psi[t0] = inv_sqrt2 * (a0 + a1)
            psi[t1] = inv_sqrt2 * (a0 - a1)
        elif gate.kind == "mcx":
            tmp = psi[t0].copy()
            psi[t0] = psi[t1]
            psi[t1] = tmp
        elif gate.kind == "mcry":
            c = math.cos(gate.angle / 2.0)
            s = math.sin(gate.angle / 2.0)
            a0 = psi[t0].copy()
            a1 = psi[t1].copy()
            psi[t0] = c * a0 - s * a1
            psi[t1] = s * a0 + c * a1
        else:
            raise ValueError(f"Unsupported primitive gate kind: {gate.kind!r}")

    return psi.reshape(-1)


@dataclass(frozen=True)
class CircuitQHEDResult:
    """Output of a full 13-qubit statevector Pauli-graph QHED execution."""

    n_qubits: int
    p_load: float
    matching_probabilities: Tuple[float, ...]
    edge_score: float | None
    loader_hadamards: int
    loader_controlled_ry: int
    matching_circuits: int
    aux_hadamards: int
    controlled_x_primitives: int
    expected_loader_attempts: float | None


def execute_pauli_graph_qhed_circuit(
    feature: np.ndarray,
    graph: PauliGraphSpec,
    n_address_qubits: int = 11,
) -> CircuitQHEDResult:
    """Execute the 13-qubit FRQI loader and all matching circuits on an 8,192-amplitude statevector."""
    n_qubits = n_address_qubits + 2
    dim = 1 << n_qubits
    pixel_qubit = n_address_qubits
    aux_qubit = n_address_qubits + 1

    loader_gates, _ = compile_frqi_loader_gates(feature, n_address_qubits=n_address_qubits)
    psi_0 = np.zeros(dim, dtype=np.complex128)
    psi_0[0] = 1.0 + 0.0j
    psi_loaded = apply_primitive_gates(psi_0, loader_gates, n_qubits=n_qubits)

    # Condition on pixel == 1 (keeping full 13-qubit 8,192-amplitude statevector).
    reshaped = psi_loaded.reshape((2,) * n_qubits)
    pix_axis = n_qubits - 1 - pixel_qubit
    aux_axis = n_qubits - 1 - aux_qubit

    sel_pix1: List[int | slice] = [slice(None)] * n_qubits
    sel_pix1[pix_axis] = 1
    p_load = float(np.sum(np.abs(reshaped[tuple(sel_pix1)]) ** 2))

    n_loader_h = sum(1 for g in loader_gates if g.kind == "h")
    n_loader_ry = sum(1 for g in loader_gates if g.kind == "mcry")

    compiled_matchings = [
        compile_matching_readout_gates(m, n_address_qubits=n_address_qubits)
        for m in graph.matchings
    ]
    total_aux_h = sum(sum(1 for g in mg if g.kind == "h") for mg in compiled_matchings)
    total_mcx = sum(sum(1 for g in mg if g.kind == "mcx") for mg in compiled_matchings)

    if p_load == 0.0:
        return CircuitQHEDResult(
            n_qubits=n_qubits,
            p_load=0.0,
            matching_probabilities=(),
            edge_score=None,
            loader_hadamards=n_loader_h,
            loader_controlled_ry=n_loader_ry,
            matching_circuits=len(graph.matchings),
            aux_hadamards=total_aux_h,
            controlled_x_primitives=total_mcx,
            expected_loader_attempts=None,
        )

    psi_post = np.zeros_like(reshaped)
    psi_post[tuple(sel_pix1)] = reshaped[tuple(sel_pix1)] / math.sqrt(p_load)
    psi_post_flat = psi_post.reshape(-1)

    probs: List[float] = []
    sel_aux1: List[int | slice] = [slice(None)] * n_qubits
    sel_aux1[pix_axis] = 1
    sel_aux1[aux_axis] = 1
    t_aux1 = tuple(sel_aux1)

    for mg in compiled_matchings:
        out_state = apply_primitive_gates(psi_post_flat, mg, n_qubits=n_qubits).reshape((2,) * n_qubits)
        p_ell = float(np.sum(np.abs(out_state[t_aux1]) ** 2))
        probs.append(p_ell)

    edge_score = 0.5 * float(sum(probs))
    expected_attempts = float(len(graph.matchings)) / p_load

    return CircuitQHEDResult(
        n_qubits=n_qubits,
        p_load=p_load,
        matching_probabilities=tuple(probs),
        edge_score=edge_score,
        loader_hadamards=n_loader_h,
        loader_controlled_ry=n_loader_ry,
        matching_circuits=len(graph.matchings),
        aux_hadamards=total_aux_h,
        controlled_x_primitives=total_mcx,
        expected_loader_attempts=expected_attempts,
    )


def run_reference_benchmark_suite() -> Dict[str, object]:
    """Run all 5 unitaries x 3 feature images x 2 Pauli graphs and summarize validation metrics."""
    labels = low_weight_pauli_labels(3, 2)
    graphs = build_pauli_graphs(labels)
    unitaries = reference_unitary_examples()

    records: List[Dict[str, object]] = []
    max_circuit_vs_edge_err = 0.0
    max_circuit_vs_laplacian_err = 0.0
    nonzero_count = 0
    zero_count = 0

    for u_name, u_spec in unitaries.items():
        ptm = low_weight_ptm(u_spec["unitary"], labels)
        features = extract_feature_images(ptm, labels)
        for f_name, f_vec in features.items():
            nz_pixels = int(np.count_nonzero(f_vec))
            for g_name, g_spec in graphs.items():
                circ = execute_pauli_graph_qhed_circuit(f_vec, g_spec, n_address_qubits=11)
                if circ.edge_score is None:
                    zero_count += 1
                    records.append(
                        {
                            "unitary": u_name,
                            "formula": u_spec["formula"],
                            "feature": f_name,
                            "graph": g_name,
                            "nonzero_pixels": nz_pixels,
                            "p_load": 0.0,
                            "classical_edge_sum_score": None,
                            "classical_laplacian_score": None,
                            "circuit_score": None,
                            "abs_diff_edge_sum": None,
                            "abs_diff_laplacian": None,
                        }
                    )
                else:
                    nonzero_count += 1
                    cl_edge = classical_edge_sum_score(f_vec, g_spec.edges)
                    cl_lap = classical_laplacian_score(f_vec, g_spec.laplacian)
                    err_edge = abs(circ.edge_score - cl_edge)
                    err_lap = abs(circ.edge_score - cl_lap)
                    max_circuit_vs_edge_err = max(max_circuit_vs_edge_err, err_edge)
                    max_circuit_vs_laplacian_err = max(max_circuit_vs_laplacian_err, err_lap)
                    records.append(
                        {
                            "unitary": u_name,
                            "formula": u_spec["formula"],
                            "feature": f_name,
                            "graph": g_name,
                            "nonzero_pixels": nz_pixels,
                            "p_load": circ.p_load,
                            "classical_edge_sum_score": cl_edge,
                            "classical_laplacian_score": cl_lap,
                            "circuit_score": circ.edge_score,
                            "abs_diff_edge_sum": err_edge,
                            "abs_diff_laplacian": err_lap,
                            "matching_circuits": circ.matching_circuits,
                            "aux_hadamards": circ.aux_hadamards,
                            "controlled_x_primitives": circ.controlled_x_primitives,
                            "expected_loader_attempts": circ.expected_loader_attempts,
                        }
                    )

    return {
        "n_physical_qubits": 3,
        "max_pauli_weight": 2,
        "num_low_weight_paulis": len(labels),
        "num_ptm_pixels": len(labels) ** 2,
        "padded_addresses": 2048,
        "circuit_qubits": 13,
        "statevector_amplitudes": 8192,
        "graphs": {
            g_name: {
                "vertices": g_spec.n_vertices,
                "undirected_edges": len(g_spec.edges),
                "matchings": len(g_spec.matchings),
                "controlled_x_primitives": sum(
                    2 * ((u ^ v).bit_count()) - 1 for u, v in g_spec.edges
                ),
            }
            for g_name, g_spec in graphs.items()
        },
        "total_cases": len(records),
        "nonzero_cases": nonzero_count,
        "zero_image_cases": zero_count,
        "max_abs_diff_circuit_vs_edge_sum": max_circuit_vs_edge_err,
        "max_abs_diff_circuit_vs_laplacian": max_circuit_vs_laplacian_err,
        "cases": records,
    }


def qiskit_small_width_crosscheck() -> Dict[str, object] | None:
    """Optionally cross-check a 4-qubit loader and matching circuit against Qiskit Statevector.

    Returns None if Qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import MCXGate, RYGate
        from qiskit.quantum_info import Statevector
        import qiskit
    except ImportError:
        return None

    n_addr = 2
    n_qubits = n_addr + 2  # 4 qubits: addr[0..1], pixel=2, aux=3
    feature = np.array([0.25, 0.90, 0.40, 0.70], dtype=np.float64)
    matching = ((0, 3), (1, 2))

    loader_gates, _ = compile_frqi_loader_gates(feature, n_address_qubits=n_addr)
    readout_gates = compile_matching_readout_gates(matching, n_address_qubits=n_addr)
    all_gates = loader_gates + readout_gates

    psi_zero = np.zeros(1 << n_qubits, dtype=np.complex128)
    psi_zero[0] = 1.0 + 0.0j
    ref_loader_state = apply_primitive_gates(psi_zero, loader_gates, n_qubits=n_qubits)
    ref_full_state = apply_primitive_gates(psi_zero, all_gates, n_qubits=n_qubits)

    def to_qiskit_circuit(primitives: Sequence[PrimitiveGate]) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for gate in primitives:
            if gate.kind == "h":
                qc.h(gate.target)
            elif gate.kind == "mcry":
                ctrl_wires = [q for q, _ in gate.controls]
                # Qiskit ctrl_state string is ordered from highest-index control wire in list to lowest
                ctrl_state = "".join(str(val) for _, val in reversed(gate.controls))
                mc_ry = RYGate(gate.angle).control(len(ctrl_wires), ctrl_state=ctrl_state)
                qc.append(mc_ry, ctrl_wires + [gate.target])
            elif gate.kind == "mcx":
                ctrl_wires = [q for q, _ in gate.controls]
                ctrl_state = "".join(str(val) for _, val in reversed(gate.controls))
                mcx = MCXGate(len(ctrl_wires), ctrl_state=ctrl_state)
                qc.append(mcx, ctrl_wires + [gate.target])
            else:
                raise ValueError(f"Unknown gate kind: {gate.kind}")
        return qc

    qk_loader_state = Statevector.from_instruction(to_qiskit_circuit(loader_gates)).data
    qk_full_state = Statevector.from_instruction(to_qiskit_circuit(all_gates)).data

    loader_max_diff = float(np.max(np.abs(ref_loader_state - qk_loader_state)))
    full_max_diff = float(np.max(np.abs(ref_full_state - qk_full_state)))
    return {
        "qiskit_version": qiskit.__version__,
        "n_qubits": n_qubits,
        "loader_max_amplitude_diff": loader_max_diff,
        "full_circuit_max_amplitude_diff": full_max_diff,
    }

