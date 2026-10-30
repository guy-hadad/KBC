"""ETAM: Events and Time As Modalities.

A transformer that reads an event history as two interleaved streams of soft
tokens, one *time* token and one *event* token per event, and gives each
stream its own parameters in every layer (norms, Q/K/V/O, FFN) while both meet
in one shared attention softmax. The design transfers the modality-expert
recipe of Mono-InternVL / Mixture-of-Transformers / RAM
(`~/ST/docs_RAM/methodology.md`) from (text, image) and (text, item) to
(time, event). The method contract lives in `docs/etam/methodology.md`.
"""
