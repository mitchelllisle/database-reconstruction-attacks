import os
from logging import basicConfig, getLogger

basicConfig(level=os.getenv('LOG_LEVEL', default='INFO').upper())

logger = getLogger(__name__)
