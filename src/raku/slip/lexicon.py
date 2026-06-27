"""Ophthalmic concept vocabulary and concept-disease grounding.

Ref: Appendix B (M_a=12, M_p=45, M_s=9) and Appendix C / Table 13.
"""

from __future__ import annotations

from dataclasses import dataclass

ANATOMICAL: tuple[str, ...] = (
    "macula",
    "fovea",
    "optic disc",
    "optic cup",
    "neuroretinal rim",
    "retinal arteries",
    "retinal veins",
    "retinal nerve fiber layer",
    "retinal pigment epithelium",
    "choroid",
    "vitreous",
    "peripheral retina",
)

PATHOLOGICAL: tuple[str, ...] = (
    "microaneurysms",
    "dot hemorrhages",
    "blot hemorrhages",
    "flame hemorrhages",
    "cotton wool spots",
    "hard exudates",
    "neovascularization",
    "venous beading",
    "intraretinal microvascular abnormalities",
    "hard drusen in macula",
    "soft drusen in macula",
    "macular pigmentary changes",
    "geographic atrophy in macula",
    "macular edema",
    "epiretinal membrane",
    "macular hole",
    "macular hemorrhage",
    "macular exudates",
    "disc pallor",
    "disc edema",
    "optic disc cupping",
    "disc notching",
    "peripapillary atrophy",
    "disc hemorrhages",
    "disc neovascularization",
    "optic disc drusen",
    "retinal detachment",
    "retinal tears",
    "lattice degeneration",
    "retinoschisis",
    "atrophic holes",
    "retinal thinning",
    "retinal folds",
    "arterial occlusion signs",
    "venous occlusion signs",
    "hollenhorst plaques",
    "arteriovenous nicking",
    "vascular sheathing",
    "bone spicule pigmentation",
    "bull's eye maculopathy",
    "crystalline deposits",
    "chorioretinal atrophy",
    "vitreous hemorrhage",
    "asteroid hyalosis",
    "posterior vitreous detachment",
)

SEVERITY: tuple[str, ...] = (
    "minimal",
    "mild",
    "moderate",
    "severe",
    "early-stage",
    "intermediate-stage",
    "advanced-stage",
    "proliferative",
    "vision-threatening",
)

_PATH_GROUNDING: dict[str, tuple[str, ...]] = {
    "microaneurysms": ("retinal arteries", "retinal veins", "macula"),
    "dot hemorrhages": ("retinal veins", "macula", "peripheral retina"),
    "blot hemorrhages": ("retinal veins", "peripheral retina"),
    "flame hemorrhages": ("retinal nerve fiber layer", "retinal arteries"),
    "cotton wool spots": ("retinal nerve fiber layer", "retinal arteries"),
    "hard exudates": ("macula", "retinal pigment epithelium"),
    "neovascularization": ("retinal veins", "optic disc", "peripheral retina"),
    "venous beading": ("retinal veins",),
    "intraretinal microvascular abnormalities": ("retinal arteries", "retinal veins"),
    "hard drusen in macula": ("macula", "retinal pigment epithelium"),
    "soft drusen in macula": ("macula", "retinal pigment epithelium"),
    "macular pigmentary changes": ("macula", "retinal pigment epithelium"),
    "geographic atrophy in macula": ("macula", "retinal pigment epithelium", "choroid"),
    "macular edema": ("macula", "fovea"),
    "epiretinal membrane": ("macula", "vitreous"),
    "macular hole": ("macula", "fovea"),
    "macular hemorrhage": ("macula", "choroid"),
    "macular exudates": ("macula", "fovea"),
    "disc pallor": ("optic disc", "neuroretinal rim"),
    "disc edema": ("optic disc", "neuroretinal rim"),
    "optic disc cupping": ("optic disc", "optic cup", "neuroretinal rim"),
    "disc notching": ("optic cup", "neuroretinal rim"),
    "peripapillary atrophy": ("optic disc", "retinal pigment epithelium"),
    "disc hemorrhages": ("optic disc", "neuroretinal rim"),
    "disc neovascularization": ("optic disc", "retinal veins"),
    "optic disc drusen": ("optic disc",),
    "retinal detachment": ("peripheral retina", "retinal pigment epithelium"),
    "retinal tears": ("peripheral retina",),
    "lattice degeneration": ("peripheral retina",),
    "retinoschisis": ("peripheral retina",),
    "atrophic holes": ("peripheral retina",),
    "retinal thinning": ("peripheral retina", "retinal pigment epithelium"),
    "retinal folds": ("peripheral retina", "macula"),
    "arterial occlusion signs": ("retinal arteries", "macula"),
    "venous occlusion signs": ("retinal veins",),
    "hollenhorst plaques": ("retinal arteries",),
    "arteriovenous nicking": ("retinal arteries", "retinal veins"),
    "vascular sheathing": ("retinal arteries", "retinal veins"),
    "bone spicule pigmentation": ("peripheral retina", "retinal pigment epithelium"),
    "bull's eye maculopathy": ("macula", "retinal pigment epithelium"),
    "crystalline deposits": ("macula", "retinal pigment epithelium"),
    "chorioretinal atrophy": ("choroid", "retinal pigment epithelium", "peripheral retina"),
    "vitreous hemorrhage": ("vitreous",),
    "asteroid hyalosis": ("vitreous",),
    "posterior vitreous detachment": ("vitreous",),
}


@dataclass(frozen=True, slots=True)
class DiseaseConcepts:
    name: str
    anatomical: tuple[str, ...]
    pathological: tuple[str, ...]
    reduced_weight: bool = False


_ALL_ANAT = ANATOMICAL
_ALL_PATH = PATHOLOGICAL

ODIR5K: tuple[DiseaseConcepts, ...] = (
    DiseaseConcepts("normal", _ALL_ANAT, ()),
    DiseaseConcepts(
        "diabetic retinopathy",
        ("macula", "retinal arteries", "retinal veins", "peripheral retina"),
        (
            "microaneurysms",
            "dot hemorrhages",
            "blot hemorrhages",
            "hard exudates",
            "cotton wool spots",
            "neovascularization",
        ),
    ),
    DiseaseConcepts(
        "glaucoma",
        ("optic disc", "optic cup", "neuroretinal rim", "retinal nerve fiber layer"),
        ("optic disc cupping", "disc notching", "disc pallor", "peripapillary atrophy"),
    ),
    DiseaseConcepts("cataract", _ALL_ANAT, ()),
    DiseaseConcepts(
        "age-related macular degeneration",
        ("macula", "fovea", "choroid", "retinal pigment epithelium"),
        (
            "hard drusen in macula",
            "soft drusen in macula",
            "macular pigmentary changes",
            "geographic atrophy in macula",
            "macular hemorrhage",
        ),
    ),
    DiseaseConcepts(
        "hypertensive retinopathy",
        ("retinal arteries", "retinal veins", "optic disc"),
        ("arteriovenous nicking", "flame hemorrhages", "cotton wool spots", "disc edema"),
    ),
    DiseaseConcepts(
        "pathological myopia",
        ("optic disc", "macula", "choroid", "peripheral retina"),
        (
            "peripapillary atrophy",
            "chorioretinal atrophy",
            "geographic atrophy in macula",
            "lattice degeneration",
        ),
    ),
    DiseaseConcepts("other abnormalities", _ALL_ANAT, _ALL_PATH, reduced_weight=True),
)

RFMID: tuple[DiseaseConcepts, ...] = (
    DiseaseConcepts(
        "choroideremia",
        ("choroid", "retinal pigment epithelium", "peripheral retina"),
        ("chorioretinal atrophy", "macular pigmentary changes", "retinal thinning"),
    ),
    DiseaseConcepts(
        "macular dystrophy",
        ("macula", "fovea", "retinal pigment epithelium"),
        (
            "bull's eye maculopathy",
            "macular pigmentary changes",
            "geographic atrophy in macula",
        ),
    ),
    DiseaseConcepts(
        "central retinal artery occlusion",
        ("retinal arteries", "macula"),
        ("arterial occlusion signs", "disc pallor"),
    ),
    DiseaseConcepts(
        "retinitis pigmentosa",
        ("peripheral retina", "retinal pigment epithelium", "retinal veins"),
        ("bone spicule pigmentation", "vascular sheathing", "retinal thinning"),
    ),
    DiseaseConcepts(
        "eales disease",
        ("retinal veins", "peripheral retina"),
        ("vascular sheathing", "neovascularization", "vitreous hemorrhage"),
    ),
)

JSIEC: tuple[DiseaseConcepts, ...] = (
    DiseaseConcepts(
        "coats disease",
        ("retinal veins", "peripheral retina", "macula"),
        ("hard exudates", "neovascularization", "retinal detachment"),
    ),
    DiseaseConcepts(
        "familial exudative vitreoretinopathy",
        ("peripheral retina", "retinal veins"),
        ("neovascularization", "retinal folds", "retinal detachment"),
    ),
    DiseaseConcepts(
        "morning glory syndrome",
        ("optic disc", "neuroretinal rim"),
        ("optic disc cupping", "peripapillary atrophy", "disc notching"),
    ),
)

DATASETS: dict[str, tuple[DiseaseConcepts, ...]] = {
    "odir5k": ODIR5K,
    "rfmid": RFMID,
    "jsiec": JSIEC,
}


def concept_phrases() -> dict[str, tuple[str, ...]]:
    anat = tuple(f"{a} structure of the fundus" for a in ANATOMICAL)
    path = tuple(f"{p} on fundus photography" for p in PATHOLOGICAL)
    sev = tuple(f"{s} clinical severity" for s in SEVERITY)
    return {"anatomical": anat, "pathological": path, "severity": sev}


def grounding_edges() -> list[tuple[int, int]]:
    a_idx = {a: i for i, a in enumerate(ANATOMICAL)}
    edges: list[tuple[int, int]] = []
    for k, path in enumerate(PATHOLOGICAL):
        for anat in _PATH_GROUNDING[path]:
            edges.append((k, a_idx[anat]))
    return edges


def total_concepts() -> int:
    return len(ANATOMICAL) + len(PATHOLOGICAL) + len(SEVERITY)


def class_names(dataset: str) -> tuple[str, ...]:
    return tuple(d.name for d in DATASETS[dataset])


def indicator_matrix(dataset: str) -> list[list[float]]:
    a_idx = {a: i for i, a in enumerate(ANATOMICAL)}
    p_off = len(ANATOMICAL)
    p_idx = {p: p_off + i for i, p in enumerate(PATHOLOGICAL)}
    width = total_concepts()
    rows: list[list[float]] = []
    for disease in DATASETS[dataset]:
        row = [0.0] * width
        for anat in disease.anatomical:
            row[a_idx[anat]] = 1.0
        for path in disease.pathological:
            row[p_idx[path]] = 1.0
        rows.append(row)
    return rows


def class_weights(dataset: str) -> list[float]:
    return [0.5 if d.reduced_weight else 1.0 for d in DATASETS[dataset]]
