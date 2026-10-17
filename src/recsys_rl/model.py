"""GPT-2 next-item recommender (stage 2).

Item-ID tokens (+ optional feedback-feature embeddings) -> ``transformers.GPT2Model``
-> tied output projection. Trained with sampled softmax and logQ correction,
evaluated with full softmax over the vocabulary. The trained model is frozen as the
reference policy ``pi_ref`` and as the behaviour policy for off-policy correction.
"""
