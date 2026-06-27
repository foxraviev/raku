"""Command-line surface: throw, grade, read, cast."""

from __future__ import annotations

from pathlib import Path

import clize

from raku.bench.glaze_io import kiln_log, load_kiln, set_seed
from raku.bench.settings import load_settings
from raku.bench.spinning import maybe_init, pick_device, shutdown
from raku.firing.trainer import Trainer
from raku.sorting.readout import diagnose
from raku.sorting.runner import evaluate
from raku.wedging.dressing import eval_transform
from raku.wedging.feed import build_loader
from raku.wheel.export import export_onnx
from raku.wheel.net import build_model

_LOG = kiln_log("raku.wheelhead")


def throw(config: str, *, resume: str = "") -> None:
    """Train RareEyeVLM from a bench sheet.

    :param config: path to a TOML bench sheet.
    :param resume: checkpoint to resume from.
    """
    maybe_init()
    settings = load_settings(config)
    set_seed(settings.kiln.seed)
    device = pick_device()
    model = build_model(settings.body, settings.seam.dataset, settings.seam.num_classes)
    trainer = Trainer(model, settings, device)
    if resume:
        start = trainer.load(resume)
        _LOG.info("resumed at epoch %d", start)
    trainer.fit(
        build_loader(settings, "train", train=True),
        build_loader(settings, "val", train=False),
    )
    trainer.save(Path(settings.station.out_dir) / "final.pt", settings.kiln.epochs)
    shutdown()


def grade(config: str, weights: str) -> None:
    """Score a checkpoint on the test split.

    :param config: path to a TOML bench sheet.
    :param weights: trained checkpoint.
    """
    settings = load_settings(config)
    device = pick_device()
    model = build_model(settings.body, settings.seam.dataset, settings.seam.num_classes).to(
        device
    )
    model.load_state_dict(load_kiln(weights)["model"])
    result = evaluate(
        model, build_loader(settings, "test", train=False), device, settings.station.threshold
    )
    for key, value in result["report"].items():
        _LOG.info("%s = %.4f", key, value)
    _LOG.info("ece = %.4f mce = %.4f", result["ece"], result["mce"])


def read(config: str, weights: str, image: str) -> None:
    """Diagnose a single fundus image with concept explanation.

    :param config: path to a TOML bench sheet.
    :param weights: trained checkpoint.
    :param image: image file to read.
    """
    from PIL import Image

    settings = load_settings(config)
    device = pick_device()
    model = build_model(settings.body, settings.seam.dataset, settings.seam.num_classes).to(
        device
    )
    model.load_state_dict(load_kiln(weights)["model"])
    with Image.open(image) as raw:
        tensor = eval_transform(settings.body.image_size)(raw.convert("RGB"))
    for finding in diagnose(model, tensor, device, settings.station.threshold):
        _LOG.info(
            "%s conf=%.3f anat=%s path=%s sev=%s",
            finding.disease,
            finding.confidence,
            finding.anatomical,
            finding.pathological,
            finding.severity,
        )


def cast(config: str, weights: str, out: str) -> None:
    """Export the prediction path to ONNX.

    :param config: path to a TOML bench sheet.
    :param weights: trained checkpoint.
    :param out: destination .onnx path.
    """
    settings = load_settings(config)
    model = build_model(settings.body, settings.seam.dataset, settings.seam.num_classes)
    model.load_state_dict(load_kiln(weights)["model"])
    path = export_onnx(model, out, settings.body.image_size)
    _LOG.info("wrote %s", path)


def main() -> None:
    clize.run({"throw": throw, "grade": grade, "read": read, "cast": cast})


if __name__ == "__main__":
    main()
