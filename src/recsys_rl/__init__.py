"""RL fine-tuning of a GPT-2 sequential recommender on Yambda-50M.

Modules (see docs/codeagents/research_plan.md for the overall plan):

- ``config``       experiment configuration (dataclasses + YAML)
- ``utils``        seeding, device selection, checkpoint helpers
- ``data``         loading, temporal split, vocabulary, training windows   (stage 1)
- ``metrics``      ranking / beyond-accuracy metrics                       (stage 1)
- ``model``        GPT-2 next-item recommender (reference policy)          (stage 2)
- ``reward_model`` user feedback model: ratio / listen+ / skip / like      (stage 3)
- ``rewards``      reward functions R1-R6 on top of reward-model outputs  (stages 4-6)
- ``rl``           RLOO/GRPO + KL, off-policy REINFORCE, DPO               (stages 4-5)
- ``generate``     top-K inference and multi-step rollouts                 (stage 6)
"""

__version__ = "0.1.0"
