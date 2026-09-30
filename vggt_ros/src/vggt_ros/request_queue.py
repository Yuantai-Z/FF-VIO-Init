# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded latest-wins request handoff for the inference worker."""

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
from typing import Callable, Optional, Sequence, Tuple, TypeVar


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class InferenceRequest:
    generation: int
    timestamps: Tuple[float, ...]


class LatestRequestMailbox:
    """Keep one pending request while allowing one request to be in flight.

    New submissions replace a queued request and advance ``generation``.  An
    in-flight worker can therefore discard its complete result if any newer
    request arrived while it was resolving images or running inference.
    """

    def __init__(self) -> None:
        self._queue = Queue(maxsize=1)
        self._generation = 0
        self._generation_lock = threading.Lock()

    def submit(self, timestamps: Sequence[float]) -> InferenceRequest:
        values = tuple(timestamps)
        if not values:
            raise ValueError("an inference request must contain at least one timestamp")

        with self._generation_lock:
            self._generation += 1
            request = InferenceRequest(self._generation, values)
            while True:
                try:
                    self._queue.put_nowait(request)
                    return request
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        # The worker consumed the queued request after put_nowait
                        # observed a full queue. Retry without changing generation.
                        continue
                    self._queue.task_done()

    def get(self, timeout=None) -> InferenceRequest:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> InferenceRequest:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def is_current(self, generation: int) -> bool:
        with self._generation_lock:
            return generation == self._generation

    def publish_if_current(
        self, generation: int, publish: Callable[[], ResultT]
    ) -> Tuple[bool, Optional[ResultT]]:
        """Run one complete publish action only when ``generation`` is current.

        The generation lock makes a multi-message batch atomic with respect to
        request submission: callers either publish the complete current result
        or publish none of a stale result.
        """

        with self._generation_lock:
            if generation != self._generation:
                return False, None
            return True, publish()
