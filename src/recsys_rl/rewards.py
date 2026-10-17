"""Reward functions R1-R6 built on top of reward-model outputs (stages 4-6).

R1 ratio, R2 listening time, R3 like + ratio/10, R4 length-debiased ratio,
R5 penalties (repeats, dislikes, ensemble uncertainty), R6 long-term proxies
(session continuation, discounted rollout return).
"""
