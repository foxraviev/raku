"""RareEyeVLM end-to-end model.

Ref: Sec. III and Alg. 1-2. Frozen towers feed HCDM (Eq. 1-7), APSN
(Eq. 8-18) and IPC (Eq. 19-26); only the concept-grounding head trains.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn

from raku.bench.settings import BodyCfg
from raku.slip import lexicon
from raku.wheel.backbones import Encoders, build_encoders
from raku.wheel.calibration import CalibratedHead
from raku.wheel.decompose import Decomposer
from raku.wheel.prompting import PromptAssembler
from raku.wheel.synthesis import PromptSynthesis


class Readout(TypedDict):
    logit: Tensor
    prob: Tensor
    concept_conf: Tensor
    conf: Tensor
    activation: Tensor


class RareEyeVLM(nn.Module):
    e_anat: Tensor
    e_path: Tensor
    e_sev: Tensor
    adjacency: Tensor

    def __init__(self, cfg: BodyCfg, dataset: str, encoders: Encoders | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.encoders = encoders or build_encoders(
            cfg.embed_dim,
            cfg.image_size,
            cfg.patch,
            cfg.offline_backbone,
            cfg.vision_id,
            cfg.text_id,
        )
        phrases = lexicon.concept_phrases()
        self.register_buffer("e_anat", self.encoders.encode_phrases(phrases["anatomical"]))
        self.register_buffer("e_path", self.encoders.encode_phrases(phrases["pathological"]))
        self.register_buffer("e_sev", self.encoders.encode_phrases(phrases["severity"]))
        self.register_buffer("adjacency", _adjacency(cfg.n_pathological, cfg.n_anatomical))

        self.decomposer = Decomposer(cfg.embed_dim, cfg.gat_layers, cfg.gat_heads)
        self.apsn = PromptSynthesis(
            cfg.embed_dim,
            cfg.xmodal_heads,
            cfg.n_anatomical,
            cfg.n_pathological,
            cfg.n_severity,
            cfg.ffn_hidden,
        )
        names = lexicon.class_names(dataset)
        class_words = [self.encoders.word_embeddings(name) for name in names]
        self.prompts = PromptAssembler(cfg.embed_dim, cfg.prompt_len, class_words)
        indicator = torch.tensor(lexicon.indicator_matrix(dataset), dtype=torch.float32)
        self.head = CalibratedHead(indicator, cfg.tau_learn_init, cfg.lambda_init)
        self.static_prompt = nn.Parameter(torch.zeros(cfg.embed_dim))
        if not cfg.use_ipc:
            self.head.set_lambda(1.0)

    def _concepts(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self.cfg.use_hcdm:
            return self.e_anat, self.e_path, self.e_sev
        return self.decomposer(self.e_anat, self.e_path, self.e_sev, self.adjacency)

    def _synthesize(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.encoders.vision(images)
        enhanced = self.apsn.enhance(tokens)
        anat, path, sev = self._concepts()
        p_dyn, activation = self.apsn.synthesize(enhanced, anat, path, sev)
        if not self.cfg.use_apsn:
            p_dyn = self.static_prompt.expand(enhanced.shape[0], -1)
        return enhanced[:, 0, :], p_dyn, activation

    def concept_activation(self, images: Tensor) -> Tensor:
        return self._synthesize(images)[2]

    def forward(self, images: Tensor) -> Readout:
        f_cls, p_dyn, activation = self._synthesize(images)
        seq, mask = self.prompts.build(p_dyn)
        b, c, s, d = seq.shape
        text = self.encoders.encode_sequence(seq.reshape(b * c, s, d), mask.reshape(b * c, s))
        text = text.view(b, c, d)
        out = self.head(f_cls, text, activation)
        return {
            "logit": out["logit"],
            "prob": out["prob"],
            "concept_conf": out["concept_conf"],
            "conf": out["conf"],
            "activation": activation,
        }


def _adjacency(n_path: int, n_anat: int) -> Tensor:
    mask = torch.zeros(n_path, n_anat)
    for k, j in lexicon.grounding_edges():
        mask[k, j] = 1.0
    return mask


def build_model(cfg: BodyCfg, dataset: str, num_classes: int) -> RareEyeVLM:
    if num_classes != len(lexicon.class_names(dataset)):
        raise ValueError("num_classes does not match the dataset concept map")
    return RareEyeVLM(cfg, dataset)
