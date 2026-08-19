"""
Shared rationale-text builder for Phase F (training-time do-calculus scaffolding
follow-up). Phase D already showed that adding this same kind of explicit
do-calculus framing to the PROMPT at inference time never helped confounded-
topology questions (0% for both implicit and explicit phrasing, at every
checkpoint). This tests a different intervention: baking a short worked
derivation into the model's own TRAINING TARGETS (warm-up SFT + RFT rollouts),
so the model has to learn to produce and act on this content itself, rather
than being handed it externally at eval time.

Only "confounded" has a real hidden shared cause (U) to name explicitly --
naming a nonexistent confound for chain/fork/collider would be fabricating
content the true generative model doesn't have, so those topologies get a
shorter, honestly-scoped version of the same intervention principle.
"""
from causal_dag_task import VAR_NAMES


def build_intervention_rationale(topology, do_var, do_value, target_var):
    do_desc = VAR_NAMES[do_var]
    target_desc = VAR_NAMES[target_var]
    val = "on" if do_value else "off"
    if topology == "confounded":
        return (f"This is an intervention. We force {do_desc} to {val} directly, "
                 f"ignoring whatever would normally cause {do_desc} -- including any "
                 f"hidden shared cause with {target_desc}. So {target_desc}'s "
                 f"probability depends only on the direct causal path from {do_desc}, "
                 f"not on the hidden shared cause.")
    return (f"This is an intervention. We force {do_desc} to {val} directly, "
             f"ignoring whatever would normally cause {do_desc}. {target_desc}'s "
             f"probability then depends only on the causal path from {do_desc}.")
