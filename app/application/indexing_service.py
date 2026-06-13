"""Minimal offline indexing service: run 5 layer classes in order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.application_layers.offline.build_chunks import BuildChunksLayer
from app.application_layers.offline.build_cleaned_docs import BuildCleanedDocsLayer
from app.application_layers.offline.build_embeddings import BuildEmbeddingsLayer
from app.application_layers.offline.build_parsed_docs import BuildParsedDocsLayer
from app.application_layers.offline.build_vector_index import BuildVectorIndexLayer
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class IndexingService:
    """Orchestrates offline indexing flow from existing layers."""

    def __init__(self, *, config_path: Path | str = Path("configs/config.yaml")) -> None:
        self._config_path = Path(config_path)

    def run(self) -> None:
        BuildParsedDocsLayer(config_path=self._config_path).run()
        BuildCleanedDocsLayer(config_path=self._config_path).run()
        BuildChunksLayer(config_path=self._config_path).run()
        BuildEmbeddingsLayer(config_path=self._config_path).run()
        BuildVectorIndexLayer(config_path=self._config_path).run()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full offline indexing (5 sequential offline layers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml or config directory for all offline layers.",
    )
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    try:
        IndexingService(config_path=args.config).run()
    except Exception:
        logger.exception("IndexingService failed")
        sys.exit(1)
    finally:
        logger.info("IndexingService finished")


if __name__ == "__main__":
    main()
