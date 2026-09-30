# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
import unittest

from vggt_ros.request_queue import LatestRequestMailbox


class LatestRequestMailboxTest(unittest.TestCase):
    def test_latest_pending_replaces_older_and_accounting_reaches_join(self):
        mailbox = LatestRequestMailbox()
        first = mailbox.submit((1.0,))
        self.assertIs(mailbox.get_nowait(), first)

        mailbox.submit((2.0,))
        latest = mailbox.submit((3.0,))
        self.assertFalse(mailbox.is_current(first.generation))
        self.assertIs(mailbox.get_nowait(), latest)

        mailbox.task_done()
        mailbox.task_done()
        joined = threading.Event()
        join_thread = threading.Thread(
            target=lambda: (mailbox.join(), joined.set()), daemon=True
        )
        join_thread.start()
        self.assertTrue(joined.wait(timeout=1.0), "queue task accounting did not reach zero")
        join_thread.join(timeout=1.0)

    def test_stale_inference_result_is_not_partially_published(self):
        mailbox = LatestRequestMailbox()
        in_flight = mailbox.submit((1.0, 2.0))
        self.assertIs(mailbox.get_nowait(), in_flight)

        inference_started = threading.Event()
        finish_inference = threading.Event()
        published = []
        accepted_results = []

        def complete_stale_inference():
            inference_started.set()
            finish_inference.wait(timeout=1.0)
            accepted, _ = mailbox.publish_if_current(
                in_flight.generation,
                lambda: published.extend(["old-cloud-1", "old-cloud-2"]),
            )
            accepted_results.append(accepted)
            mailbox.task_done()

        worker = threading.Thread(target=complete_stale_inference)
        worker.start()
        self.assertTrue(inference_started.wait(timeout=1.0))
        latest = mailbox.submit((3.0, 4.0))
        finish_inference.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(accepted_results, [False])
        self.assertEqual(published, [])

        self.assertIs(mailbox.get_nowait(), latest)
        accepted, _ = mailbox.publish_if_current(
            latest.generation, lambda: published.extend(["new-cloud-1", "new-cloud-2"])
        )
        self.assertTrue(accepted)
        self.assertEqual(published, ["new-cloud-1", "new-cloud-2"])
        mailbox.task_done()
        mailbox.join()


if __name__ == "__main__":
    unittest.main()
