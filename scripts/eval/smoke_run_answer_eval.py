"""Deprecated smoke wrapper for generation eval."""

from app.application_layers.eval.run_generation_eval import main
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


if __name__ == "__main__":
    setup_logging()
    logger.warning(
        "scripts/eval/smoke_run_answer_eval.py is deprecated; use smoke_run_generation_eval.py"
    )
    main()
