"""
===================================================================================================================
This file is part of GIAS3. (https://github.com/orgs/musculoskeletal/repositories?language=&q=gias3&sort=&type=all)

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
===================================================================================================================
"""

import logging


def init_log(filename=None, level=logging.INFO):
    """
    Initializes logging configuration.

    Parameters:
        filename - The path to the log file. If None, logging information will be output to the console.
        level - Specifies the levels of log information that will be included.
    """
    log_fmt = '%(levelname)s - %(asctime)s: %(message)s'
    log_level = level

    logging.basicConfig(
        filename=filename,
        level=log_level,
        format=log_fmt,
    )
