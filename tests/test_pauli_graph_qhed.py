"""Focused regression and mathematical verification tests for Pauli-graph QHED."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pauli_graph_qhed as qhed  # noqa: E402


class PauliGraphQHEDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = qhed.low_weight_pauli_labels(3, 2)
        cls.graphs = qhed.build_pauli_graphs(cls.labels)
        cls.benchmark = qhed.run_reference_benchmark_suite()

    def test_low_weight_pauli_counts_and_padding_dimensions(self) -> None:
        expected_k = sum(3**w * math.comb(3, w) for w in range(3))
        self.assertEqual(expected_k, 37)
        self.assertEqual(len(self.labels), 37)
        self.assertEqual(len(self.labels) ** 2, 1369)
        self.assertEqual(self.benchmark["padded_addresses"], 2048)
        self.assertEqual(self.benchmark["circuit_qubits"], 13)
        self.assertEqual(self.benchmark["statevector_amplitudes"], 8192)

    def test_independent_graph_reconstruction_and_matching_counts(self) -> None:
        h_spec = self.graphs["hamming_1"]
        c_spec = self.graphs["anticommuting_factor"]

        self.assertEqual(h_spec.n_vertices, 1369)
        self.assertEqual(len(h_spec.edges), 9324)
        self.assertEqual(len(h_spec.matchings), 18)

        self.assertEqual(c_spec.n_vertices, 1369)
        self.assertEqual(len(c_spec.edges), 4662)
        self.assertEqual(len(c_spec.matchings), 9)

        # Independent Cartesian-product adjacency reconstruction from 1D low-weight graphs
        n1 = len(self.labels)
        adj1_h = np.zeros((n1, n1), dtype=int)
        adj1_c = np.zeros((n1, n1), dtype=int)
        anticomm = {("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y"), ("Z", "X"), ("X", "Z")}
        for i, a in enumerate(self.labels):
            for j, b in enumerate(self.labels):
                diffs = [pos for pos in range(3) if a[pos] != b[pos]]
                if len(diffs) == 1:
                    adj1_h[i, j] = 1
                    if (a[diffs[0]], b[diffs[0]]) in anticomm:
                        adj1_c[i, j] = 1

        eye1 = np.eye(n1, dtype=int)
        adj2_h = np.kron(adj1_h, eye1) + np.kron(eye1, adj1_h)
        adj2_c = np.kron(adj1_c, eye1) + np.kron(eye1, adj1_c)
        self.assertEqual(int(np.sum(adj2_h)) // 2, 9324)
        self.assertEqual(int(np.sum(adj2_c)) // 2, 4662)

        # Verify each matching is disjoint and matchings partition the undirected edge set
        for spec in (h_spec, c_spec):
            union_edges = []
            for m in spec.matchings:
                touched = [x for edge in m for x in edge]
                self.assertEqual(len(touched), len(set(touched)))
                union_edges.extend(m)
            self.assertEqual(sorted(union_edges), sorted(spec.edges))

    def test_exhaustive_small_width_controlled_gray_transpositions(self) -> None:
        rng = np.random.default_rng(20261108)
        n_addr = 3
        aux_q = n_addr + 1
        n_qubits = n_addr + 2
        dim = 1 << n_qubits

        base_state = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
        base_state /= np.linalg.norm(base_state)

        for u, v in itertools.combinations(range(1 << n_addr), 2):
            d = (u ^ v).bit_count()
            gates = qhed.expand_controlled_transposition_gray(u, v, n_addr, aux_q)
            self.assertEqual(len(gates), 2 * d - 1)
            out = qhed.apply_primitive_gates(base_state, gates, n_qubits=n_qubits)

            expected = base_state.copy()
            for idx in range(dim):
                aux_val = (idx >> aux_q) & 1
                if aux_val == 1:
                    addr = idx & ((1 << n_addr) - 1)
                    if addr == u:
                        partner = (idx & ~((1 << n_addr) - 1)) | v
                        expected[idx] = base_state[partner]
                    elif addr == v:
                        partner = (idx & ~((1 << n_addr) - 1)) | u
                        expected[idx] = base_state[partner]
            np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-15)

    def test_zx_analytical_scores_and_primitive_gate_counts(self) -> None:
        u_zx = qhed.reference_unitary_examples()["zx_crosstalk"]["unitary"]
        ptm = qhed.low_weight_ptm(u_zx, self.labels)
        feat = qhed.extract_feature_images(ptm, self.labels)["cross_weight_antisymmetric"]
        self.assertEqual(int(np.count_nonzero(feat)), 8)

        res_h = qhed.execute_pauli_graph_qhed_circuit(feat, self.graphs["hamming_1"], 11)
        res_c = qhed.execute_pauli_graph_qhed_circuit(feat, self.graphs["anticommuting_factor"], 11)

        self.assertAlmostEqual(res_h.p_load, 8.0 / 2048.0, places=15)
        self.assertEqual(res_h.loader_hadamards, 11)
        self.assertEqual(res_h.loader_controlled_ry, 8)
        self.assertEqual(res_h.matching_circuits, 18)
        self.assertEqual(res_h.aux_hadamards, 36)
        self.assertEqual(res_h.controlled_x_primitives, 62400)
        self.assertAlmostEqual(res_h.expected_loader_attempts, 4608.0, places=10)
        self.assertAlmostEqual(res_h.edge_score, 3.75, places=14)

        self.assertEqual(res_c.matching_circuits, 9)
        self.assertEqual(res_c.aux_hadamards, 18)
        self.assertEqual(res_c.controlled_x_primitives, 30638)
        self.assertAlmostEqual(res_c.expected_loader_attempts, 2304.0, places=10)
        self.assertAlmostEqual(res_c.edge_score, 1.5, places=14)

    def test_all_twenty_eight_nonzero_and_two_zero_cases_match_classical_baselines(self) -> None:
        self.assertEqual(self.benchmark["total_cases"], 30)
        self.assertEqual(self.benchmark["nonzero_cases"], 28)
        self.assertEqual(self.benchmark["zero_image_cases"], 2)
        self.assertLess(self.benchmark["max_abs_diff_circuit_vs_edge_sum"], 1e-14)
        self.assertLess(self.benchmark["max_abs_diff_circuit_vs_laplacian"], 1e-14)

        for row in self.benchmark["cases"]:
            if row["unitary"] == "local_drift" and row["feature"] == "cross_weight_antisymmetric":
                self.assertIsNone(row["circuit_score"])
                self.assertEqual(row["p_load"], 0.0)
            else:
                self.assertIsNotNone(row["circuit_score"])
                self.assertLess(row["abs_diff_edge_sum"], 1e-14)
                self.assertLess(row["abs_diff_laplacian"], 1e-14)

    def test_zero_and_constant_boundary_cases(self) -> None:
        zero_vec = np.zeros(1369, dtype=np.float64)
        with self.assertRaises(ValueError):
            qhed.classical_edge_sum_score(zero_vec, self.graphs["hamming_1"].edges)
        with self.assertRaises(ValueError):
            qhed.classical_laplacian_score(zero_vec, self.graphs["hamming_1"].laplacian)

        zero_circ = qhed.execute_pauli_graph_qhed_circuit(zero_vec, self.graphs["hamming_1"], 11)
        self.assertEqual(zero_circ.p_load, 0.0)
        self.assertIsNone(zero_circ.edge_score)

        # Constant S > 0 on the 1,369 graph vertices has zero edge energy despite zero-padded tail
        const_vec = np.full(1369, 0.73, dtype=np.float64)
        for spec in self.graphs.values():
            cl = qhed.classical_edge_sum_score(const_vec, spec.edges)
            circ = qhed.execute_pauli_graph_qhed_circuit(const_vec, spec, 11)
            self.assertEqual(cl, 0.0)
            self.assertIsNotNone(circ.edge_score)
            self.assertLess(abs(circ.edge_score), 1e-15)
            for p_ell in circ.matching_probabilities:
                self.assertGreaterEqual(p_ell, 0.0)
                self.assertLessEqual(p_ell, 1.0)

    def test_optional_qiskit_small_width_validation_and_precursor_preservation(self) -> None:
        precursor_path = REPO_ROOT / "NEQR_FRQI.ipynb"
        self.assertTrue(precursor_path.exists())
        precursor_nb = json.loads(precursor_path.read_text(encoding="utf-8"))
        self.assertEqual(len(precursor_nb["cells"]), 43)

        qk_res = qhed.qiskit_small_width_crosscheck()
        if qk_res is not None:
            self.assertLess(qk_res["loader_max_amplitude_diff"], 1e-14)
            self.assertLess(qk_res["full_circuit_max_amplitude_diff"], 1e-14)


if __name__ == "__main__":
    unittest.main()
