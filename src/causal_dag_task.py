"""
Causal DAG task generator with an exact oracle for all three Pearl rungs.
Adapts the design philosophy of CLADDER (Jin et al. 2023) but generates
tasks programmatically and at a scale/difficulty suited to small models.

Design note on which (given/do, target) variable pair is used per topology
(see the Step 1.3 checkpoint requirement -- association and intervention must
genuinely diverge somewhere, or RQ1 is untestable):

- chain (A->B->C): do_var=B, target=C. B->C is a real, unconfounded edge, so
  P(C|B) == P(C|do(B)) up to sampling noise. This is the DELIBERATE control
  case showing association and intervention agree absent confounding.
- fork (A->B, A->C): do_var=A, target=B. A is a root cause of B with no
  confounder of its own effect, so this also gives P(B|A) == P(B|do(A)) --
  a second control case.
- collider (A->C, B->C): the meaningful test is NOT do(A)->C (which also
  matches, A being an unconfounded root). It's explaining-away: conditioning
  on the collider C induces spurious association between A and B where none
  exists unconditionally. See `check_collider_explaining_away`.
- confounded (U->A, U->B, A->C): do_var=A, target=B. The backdoor path
  A<-U->B means P(B|A) != P(B|do(A)) in general -- this is the REQUIRED
  divergence case for the Step 1.5 checkpoint. (Note C is NOT confounded
  with A here: C's only cause is A, so P(C|A)==P(C|do(A)) exactly; B is the
  variable where confounding actually bites.)
"""
import random
from dataclasses import dataclass
from typing import Literal

Topology = Literal["chain", "fork", "collider", "confounded"]
Rung = Literal["association", "intervention", "counterfactual"]


@dataclass
class CausalDAGInstance:
    topology: Topology
    variables: dict
    noise_probs: dict
    edges: list


def sample_structural_equations(topology: Topology, rng: random.Random):
    if topology == "chain":
        p_a = 0.5
        p_noise_b = rng.uniform(0.1, 0.3)
        p_noise_c = rng.uniform(0.1, 0.3)
        edges = [("A", "B"), ("B", "C")]
        eqs = {"A": ("root", p_a), "B": ("xor_parent", "A", p_noise_b),
               "C": ("xor_parent", "B", p_noise_c)}
        return eqs, edges

    if topology == "fork":
        p_a = 0.5
        p_noise_b = rng.uniform(0.1, 0.3)
        p_noise_c = rng.uniform(0.1, 0.3)
        edges = [("A", "B"), ("A", "C")]
        eqs = {"A": ("root", p_a), "B": ("xor_parent", "A", p_noise_b),
               "C": ("xor_parent", "A", p_noise_c)}
        return eqs, edges

    if topology == "collider":
        p_a = 0.5
        p_b = 0.5
        p_noise_c = rng.uniform(0.1, 0.2)
        edges = [("A", "C"), ("B", "C")]
        eqs = {"A": ("root", p_a), "B": ("root", p_b),
               "C": ("or_parents", ["A", "B"], p_noise_c)}
        return eqs, edges

    if topology == "confounded":
        p_u = 0.5
        # Strong confounding on purpose (low noise) so A tightly tracks U,
        # and B tightly tracks U -- this is what makes P(B|A) diverge from
        # P(B|do(A)) by a wide, easily-detectable margin.
        p_noise_a = rng.uniform(0.02, 0.08)
        p_noise_b = rng.uniform(0.02, 0.08)
        p_noise_c = rng.uniform(0.1, 0.3)
        edges = [("U", "A"), ("U", "B"), ("A", "C")]
        eqs = {"U": ("root", p_u), "A": ("xor_parent", "U", p_noise_a),
               "B": ("xor_parent", "U", p_noise_b),
               "C": ("xor_parent", "A", p_noise_c)}
        return eqs, edges

    raise ValueError(f"Unknown topology: {topology}")


def simulate_once(eqs: dict, rng: random.Random, interventions: dict = None):
    interventions = interventions or {}
    values = {}
    noise_draws = {}

    for var, spec in eqs.items():
        if var in interventions:
            values[var] = interventions[var]
            continue

        kind = spec[0]
        if kind == "root":
            p = spec[1]
            noise = rng.random() < p
            values[var] = noise
            noise_draws[var] = noise

        elif kind == "xor_parent":
            parent, p_noise = spec[1], spec[2]
            noise = rng.random() < p_noise
            noise_draws[var] = noise
            values[var] = values[parent] != noise

        elif kind == "or_parents":
            parents, p_noise = spec[1], spec[2]
            noise = rng.random() < p_noise
            noise_draws[var] = noise
            values[var] = any(values[p] for p in parents) or noise

    return values, noise_draws


def estimate_association(eqs, target_var, given_var, given_value, n_samples=20000, seed=0):
    """P(target=True | given_var=given_value) via rejection sampling."""
    rng = random.Random(seed)
    matches, positive = 0, 0
    for _ in range(n_samples):
        values, _ = simulate_once(eqs, rng)
        if values[given_var] == given_value:
            matches += 1
            if values[target_var]:
                positive += 1
    return positive / matches if matches > 0 else None


def estimate_association_conditional(eqs, target_var, conditions: dict, n_samples=20000, seed=0):
    """P(target=True | conditions) for multiple conditioning variables at once."""
    rng = random.Random(seed)
    matches, positive = 0, 0
    for _ in range(n_samples):
        values, _ = simulate_once(eqs, rng)
        if all(values[k] == v for k, v in conditions.items()):
            matches += 1
            if values[target_var]:
                positive += 1
    return positive / matches if matches > 0 else None


def estimate_marginal(eqs, target_var, n_samples=20000, seed=0):
    """P(target=True), no conditioning."""
    rng = random.Random(seed)
    positive = 0
    for _ in range(n_samples):
        values, _ = simulate_once(eqs, rng)
        if values[target_var]:
            positive += 1
    return positive / n_samples


def estimate_intervention(eqs, target_var, do_var, do_value, n_samples=20000, seed=0):
    """P(target=True | do(do_var=do_value)) via direct simulation with intervention."""
    rng = random.Random(seed)
    positive = 0
    for _ in range(n_samples):
        values, _ = simulate_once(eqs, rng, interventions={do_var: do_value})
        if values[target_var]:
            positive += 1
    return positive / n_samples


def compute_counterfactual(eqs, observed: dict, cf_var, cf_value, target_var, rng: random.Random,
                            max_tries=200000):
    """
    Full 3-step counterfactual procedure for ONE consistent noise draw:
    1. Abduction: find noise draws consistent with `observed`.
    2. Action: force cf_var = cf_value.
    3. Prediction: recompute target_var reusing the SAME noise draws.
    Returns None if no consistent world was found within max_tries.
    """
    for _ in range(max_tries):
        values, noise_draws = simulate_once(eqs, rng)
        if all(values[k] == v for k, v in observed.items()):
            break
    else:
        return None

    values_cf = {}
    for var, spec in eqs.items():
        if var == cf_var:
            values_cf[var] = cf_value
            continue
        kind = spec[0]
        if kind == "root":
            values_cf[var] = noise_draws[var]
        elif kind == "xor_parent":
            parent = spec[1]
            values_cf[var] = values_cf[parent] != noise_draws[var]
        elif kind == "or_parents":
            parents = spec[1]
            values_cf[var] = any(values_cf[p] for p in parents) or noise_draws[var]

    return values_cf[target_var]


def estimate_counterfactual_probability(eqs, observed: dict, cf_var, cf_value, target_var,
                                         n_samples=2000, seed=0):
    """
    P(target_cf=True | observed, do(cf_var=cf_value)) as a genuine probability,
    obtained by repeating the abduction-action-prediction procedure across many
    independent noise draws consistent with `observed` (there can be more than
    one noise realization compatible with a given observation) and averaging
    the resulting counterfactual outcome. Falls back to skipping draws where
    abduction fails to find a consistent world within the per-draw budget.
    """
    rng = random.Random(seed)
    positive, total = 0, 0
    for i in range(n_samples):
        outcome = compute_counterfactual(eqs, observed, cf_var, cf_value, target_var, rng,
                                          max_tries=2000)
        if outcome is None:
            continue
        total += 1
        if outcome:
            positive += 1
    return positive / total if total > 0 else None


def check_collider_explaining_away(eqs, n_samples=20000, seed=0):
    """
    Validation check specific to the collider topology: conditioning on the
    collider C should induce association between A and B where none exists
    unconditionally. Returns (p_b_marginal, p_b_given_c_true, p_b_given_c_false).
    """
    p_b_marginal = estimate_marginal(eqs, "B", n_samples=n_samples, seed=seed)
    p_b_given_c_true = estimate_association(eqs, "B", "C", True, n_samples=n_samples, seed=seed + 1)
    p_b_given_c_false = estimate_association(eqs, "B", "C", False, n_samples=n_samples, seed=seed + 2)
    return p_b_marginal, p_b_given_c_true, p_b_given_c_false


# ---- Per-topology question wiring (which vars go in assoc/interv/cf questions) ----

QA_CONFIG = {
    "chain": {"treat_var": "B", "target_var": "C",
              "cf_observed_vars": ["A", "C"], "cf_var": "B"},
    "fork": {"treat_var": "A", "target_var": "B",
             "cf_observed_vars": ["B", "C"], "cf_var": "A"},
    "collider": {"treat_var": "A", "target_var": "C",
                 "cf_observed_vars": ["A", "B"], "cf_var": "A"},
    "confounded": {"treat_var": "A", "target_var": "B",
                   "cf_observed_vars": ["A", "C"], "cf_var": "A"},
}

# ---- Natural language templating ----

VAR_NAMES = {"A": "switch A", "B": "switch B", "C": "light C", "U": "hidden factor U"}


def render_association_question(given_var, given_value, target_var):
    gv = "on" if given_value else "off"
    return (f"We observe that {VAR_NAMES[given_var]} is {gv}. "
            f"What is the probability that {VAR_NAMES[target_var]} is on?")


def render_intervention_question(do_var, do_value, target_var):
    dv = "on" if do_value else "off"
    return (f"Suppose we directly force {VAR_NAMES[do_var]} to be {dv}, "
            f"regardless of what would normally determine it. "
            f"What is the probability that {VAR_NAMES[target_var]} is on?")


def render_counterfactual_question(observed, cf_var, cf_value, target_var):
    obs_str = " and ".join(f"{VAR_NAMES[k]} was {'on' if v else 'off'}" for k, v in observed.items())
    cfv = "on" if cf_value else "off"
    return (f"We observed that {obs_str}. "
            f"If {VAR_NAMES[cf_var]} had been {cfv} instead, "
            f"what is the probability that {VAR_NAMES[target_var]} would have been on?")


def generate_instance(topology: Topology, rung: Rung, rng: random.Random):
    """Generate one (question, oracle_answer) instance for the given topology/rung."""
    eqs, edges = sample_structural_equations(topology, rng)
    cfg = QA_CONFIG[topology]

    if rung == "association":
        given_var, target_var = cfg["treat_var"], cfg["target_var"]
        given_value = rng.random() < 0.5
        question = render_association_question(given_var, given_value, target_var)
        answer = estimate_association(eqs, target_var, given_var, given_value,
                                        n_samples=4000, seed=rng.randrange(1 << 30))
        return {"topology": topology, "rung": rung, "question": question,
                "oracle_answer": answer, "eqs": eqs}

    if rung == "intervention":
        do_var, target_var = cfg["treat_var"], cfg["target_var"]
        do_value = rng.random() < 0.5
        question = render_intervention_question(do_var, do_value, target_var)
        answer = estimate_intervention(eqs, target_var, do_var, do_value,
                                        n_samples=4000, seed=rng.randrange(1 << 30))
        return {"topology": topology, "rung": rung, "question": question,
                "oracle_answer": answer, "eqs": eqs}

    if rung == "counterfactual":
        obs_vars = cfg["cf_observed_vars"]
        cf_var, target_var = cfg["cf_var"], cfg["target_var"]
        sample_rng = random.Random(rng.randrange(1 << 30))
        actual_values, _ = simulate_once(eqs, sample_rng)
        observed = {v: actual_values[v] for v in obs_vars}
        cf_value = not actual_values[cf_var]  # ask about the flipped world, per Step 1.2 style
        question = render_counterfactual_question(observed, cf_var, cf_value, target_var)
        answer = estimate_counterfactual_probability(
            eqs, observed, cf_var, cf_value, target_var,
            n_samples=800, seed=rng.randrange(1 << 30))
        return {"topology": topology, "rung": rung, "question": question,
                "oracle_answer": answer, "eqs": eqs}

    raise ValueError(f"Unknown rung: {rung}")
