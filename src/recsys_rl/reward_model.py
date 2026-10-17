"""User feedback (reward) model (stage 3).

GPT-2 backbone initialised from the base recommender, input is the history of
(item, feedback) pairs plus a candidate item. Heads: played ratio, listen+, skip,
like and (optionally) session continuation. Separate proxy (used for RL) and gold
(evaluation only) models, plus a small ensemble for uncertainty estimates.
"""
