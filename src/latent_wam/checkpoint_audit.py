from __future__ import annotations

import argparse
import dataclasses
import json

from latent_wam.config import load_config
from latent_wam.models import LatentWAM


def main():
    parser = argparse.ArgumentParser(
        description="Strict-load the local V-JEPA 2.1 ViT-G encoder and paired predictor"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, checkpoint=args.checkpoint),
        )
        config.validate()
    model = LatentWAM.from_config(config)
    report = dataclasses.asdict(model.load_report)
    report["context_tokens"] = config.context_tokens
    report["future_tokens"] = config.future_tokens
    report["future_output_dim"] = 4 * config.model.encoder_embed_dim
    report["strict_load"] = True
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
