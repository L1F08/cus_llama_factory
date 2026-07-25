# -*- coding: utf-8 -*-
"""平台日志器: 输出 ModelArts 平台识别的关键阶段日志 (与 llm_ft_longtime 格式一致)。"""

import logging


def get_logger(service_name):
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt='time="%(asctime)s" level="%(levelname)s" msg="%(message)s"',
            datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


PLAT_LOGGER = get_logger("serviceName")
