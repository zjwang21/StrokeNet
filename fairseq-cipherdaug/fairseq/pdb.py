# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import pdb
import sys


__all__ = ["set_trace"]


# _stdin = [None]
# _stdin_lock = multiprocessing.Lock()
# try:
#     _stdin_fd = sys.stdin.fileno()
# except Exception:
#     _stdin_fd = None


# class MultiprocessingPdb(pdb.Pdb):
#     """A Pdb wrapper that works in a multiprocessing environment.

#     Usage: `from fairseq import pdb; pdb.set_trace()`
#     """

#     def __init__(self):
#         pdb.Pdb.__init__(self, nosigint=True)

#     def _cmdloop(self):
#         stdin_bak = sys.stdin
#         with _stdin_lock:
#             try:
#                 if _stdin_fd is not None:
#                     if not _stdin[0]:
#                         _stdin[0] = os.fdopen(0)  # _stdin_fd
#                     sys.stdin = _stdin[0]
#                 self.cmdloop()
#             finally:
#                 sys.stdin = stdin_bak


# def set_trace():
#     pdb = MultiprocessingPdb()
#     pdb.set_trace(sys._getframe().f_back)


def set_trace():
    pdb = ForkablePdb()
    frame = sys._getframe().f_back  # pop the current stackframe off
    pdb.set_trace(frame)


class ForkablePdb(pdb.Pdb):
    """Pdb that works from a multiprocessing child"""

    def interaction(self, *args, **kwargs):
        original_stdin = sys.stdin
        try:
            sys.stdin = os.fdopen(0)  # 0 should be stdin's file descriptor
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = original_stdin
