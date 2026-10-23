/-
  Radial-Directional Averaging as an outer-learning-rate multiplier (paper Proposition `prop:rda_multiplier`).

  Real-valued statements over an inner-product space; they need Mathlib and are therefore kept apart from
  `lean4/HiveMindSpec.lean` (Mathlib-free). Checked with the partial Mathlib build of the AutoevolveAI
  `formal/` environment through `anse.formal.lean_runner` (axiom whitelist propext, Classical.choice, Quot.sound);
  the receipt is `artifacts/lean_rda_multiplier_check.json`.

  Statement 1 (`rda_radius_ge_direct_norm`): for non-negative weights, the RDA radius Σ wᵢ‖Δᵢ‖ is at least the
  norm of the direct weighted average ‖Σ wᵢ • Δᵢ‖, i.e. the multiplier r = radius / ‖u‖ is ≥ 1 whenever u ≠ 0.
  Statement 2 (`direct_norm_sq_eq_weighted_inner`): ‖Σ wᵢ • Δᵢ‖² = Σᵢ Σⱼ wᵢ wⱼ ⟪Δᵢ, Δⱼ⟫ (the quadratic form that
  gives r = (wᵀ C w)^(-1/2) in the common-norm case).
  Statement 3 (`orthogonal_equal_norm_direct_norm_sq`): for pairwise-orthogonal Δᵢ of common norm R and equal
  weights 1/M, ‖(1/M) Σ Δᵢ‖² = R²/M, so the direct average shrinks by √M while the RDA radius stays R.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Normed.Group.Basic

open scoped BigOperators
open Finset

namespace HiveMindRDA

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Statement 1: the RDA radius dominates the norm of the direct weighted average (triangle inequality). -/
theorem rda_radius_ge_direct_norm {ι : Type*} (s : Finset ι) (w : ι → ℝ) (Δ : ι → E)
    (hw : ∀ i ∈ s, 0 ≤ w i) :
    ‖∑ i ∈ s, w i • Δ i‖ ≤ ∑ i ∈ s, w i * ‖Δ i‖ := by
  calc ‖∑ i ∈ s, w i • Δ i‖ ≤ ∑ i ∈ s, ‖w i • Δ i‖ := norm_sum_le s _
    _ = ∑ i ∈ s, w i * ‖Δ i‖ := by
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw i hi)]

/-- Statement 2: the squared norm of the direct average is the quadratic form of the Gram matrix. -/
theorem direct_norm_sq_eq_weighted_inner {ι : Type*} (s : Finset ι) (w : ι → ℝ) (Δ : ι → E) :
    ‖∑ i ∈ s, w i • Δ i‖ ^ 2 = ∑ i ∈ s, ∑ j ∈ s, w i * w j * inner ℝ (Δ i) (Δ j) := by
  rw [← real_inner_self_eq_norm_sq, sum_inner]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [inner_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [real_inner_smul_left, real_inner_smul_right]
  ring

/-- Statement 3: pairwise-orthogonal, common-norm deltas with equal weights shrink by √M under direct averaging. -/
theorem orthogonal_equal_norm_direct_norm_sq {ι : Type*} (s : Finset ι) (Δ : ι → E) (R : ℝ)
    (hM : 0 < (s.card : ℝ))
    (hnorm : ∀ i ∈ s, ‖Δ i‖ = R)
    (horth : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → inner ℝ (Δ i) (Δ j) = 0) :
    ‖∑ i ∈ s, (1 / (s.card : ℝ)) • Δ i‖ ^ 2 = R ^ 2 / (s.card : ℝ) := by
  rw [direct_norm_sq_eq_weighted_inner]
  have hdiag : ∀ i ∈ s, ∑ j ∈ s, (1 / (s.card : ℝ)) * (1 / (s.card : ℝ)) * inner ℝ (Δ i) (Δ j)
      = (1 / (s.card : ℝ)) * (1 / (s.card : ℝ)) * R ^ 2 := by
    intro i hi
    rw [Finset.sum_eq_single i]
    · rw [real_inner_self_eq_norm_sq, hnorm i hi]
    · intro j hj hji
      rw [horth i hi j hj (Ne.symm hji)]; ring
    · intro h; exact absurd hi h
  rw [Finset.sum_congr rfl hdiag, Finset.sum_const, nsmul_eq_mul]
  field_simp

end HiveMindRDA

#print axioms HiveMindRDA.rda_radius_ge_direct_norm
#print axioms HiveMindRDA.direct_norm_sq_eq_weighted_inner
#print axioms HiveMindRDA.orthogonal_equal_norm_direct_norm_sq
