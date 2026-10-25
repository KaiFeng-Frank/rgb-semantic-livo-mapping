"""An interrupted startup must reap its actual owned subprocess."""
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from .worker import Client


class LifecycleTests(unittest.TestCase):
    def test_cancel_during_startup_reaps_child(self):
        real_popen = subprocess.Popen
        children = []
        def slow_start(command, **kwargs):
            # A real child that never sends its readiness handshake. This tests
            # process ownership and cancellation without importing CUDA in CI.
            child = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            children.append(child)
            return child
        event = threading.Event()
        timer = threading.Timer(.1, event.set)
        with tempfile.TemporaryDirectory() as root, patch('tools.visual_instances.worker.subprocess.Popen', slow_start):
            timer.start(); started = time.monotonic()
            try:
                with self.assertRaises(InterruptedError):
                    Client(sys.executable, 'not-loaded.pt', Path(root)/'worker.log', cancel_event=event)
                self.assertLess(time.monotonic()-started, 3)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
            finally:
                timer.cancel()
                for child in children:
                    if child.poll() is None:
                        child.kill(); child.wait()


if __name__ == '__main__':
    unittest.main()
