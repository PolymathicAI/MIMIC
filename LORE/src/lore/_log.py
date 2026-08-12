"""Logging for the LORE tooling.

Replaces the private `central_dogma.logger` that the data pipeline was written
against, so the published pipeline has no dependency on internal infrastructure.
Import as `from lore import logger`.
"""

import os
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LORE_LOG_LEVEL", "INFO"),
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{file.name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    ),
)

__all__ = ["logger"]
