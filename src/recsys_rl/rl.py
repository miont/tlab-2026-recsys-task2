"""RL fine-tuning (stages 4-5).

- on-policy REINFORCE with a group baseline (RLOO / GRPO) + KL to ``pi_ref``;
- off-policy REINFORCE with multi-step importance weights on logged data;
- DPO on preference pairs (baseline).
"""
