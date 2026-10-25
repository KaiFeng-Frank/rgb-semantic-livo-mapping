"""Cancellation/ownership contracts using real Python children, no GPU or ROS.

Run: python3 -m unittest tools.loop_closure.test_semantic_lifecycle -v
"""
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src import ptv3_client as module


class SemanticLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ptv3-lifecycle-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.marker = self.root/'waiting'
        self.worker = self.root/'worker.py'
        self.processes, self.environments, self.select_waits = [], [], []
        real_popen, real_select = subprocess.Popen, module.select.select

        def launch(*args, **kwargs):
            self.environments.append(kwargs['env'].copy())
            process = real_popen(*args, **kwargs)
            self.processes.append(process)
            return process

        def select(*args):
            self.select_waits.append(args[3])
            return real_select(*args)

        for target, name, value in ((module, 'PTV3_PY', sys.executable),
                                     (module, 'WORKER', str(self.worker)),
                                     (module.subprocess, 'Popen', launch),
                                     (module.select, 'select', select)):
            context = patch.object(target, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        # Only processes created by this test; no process-name matching/signals.
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()
        for env in self.environments:
            try:
                shm = shared_memory.SharedMemory(name=env['PTV3_SHM_NAME'])
            except FileNotFoundError:
                continue
            shm.close()
            shm.unlink()

    def write_worker(self, body):
        self.worker.write_text('import os, struct, sys, time\nfrom pathlib import Path\n'
                               f'marker = Path({str(self.marker)!r})\n'+body)

    def wait_for_marker(self):
        deadline = time.monotonic()+2
        while not self.marker.exists() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(self.marker.exists(), 'dummy worker did not reach its blocking read')

    def assert_cleaned(self):
        self.assertEqual(len(self.processes), 1)
        process = self.processes[0]
        self.assertIsNotNone(process.returncode, 'owned child was not waited/reaped')
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=self.environments[0]['PTV3_SHM_NAME'])

    def cancel_startup(self, prefix):
        self.write_worker(f'os.write(1, {prefix!r})\nmarker.write_text("waiting")\ntime.sleep(60)\n')
        stop, errors = threading.Event(), []

        def construct():
            try:
                client = module.PTv3Client(cancel_event=stop, max_pts=8, startup_timeout_s=120)
            except Exception as error:
                errors.append(error)
            else:
                client.close()

        thread = threading.Thread(target=construct, daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(timeout=2)))
        self.wait_for_marker()
        # Allow the partial header to be consumed before cancellation.
        time.sleep(.02)
        start = time.monotonic()
        stop.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), 'startup cancellation exceeded 2 seconds')
        self.assertLess(time.monotonic()-start, 2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertTrue(self.select_waits)
        self.assertLessEqual(max(self.select_waits), .1)
        self.assert_cleaned()

    def test_cancel_before_ready_kills_reaps_and_unlinks_owned_resources(self):
        self.cancel_startup(b'')

    def test_cancel_partial_ready_header(self):
        self.cancel_startup(b'RD')

    def test_cancel_partial_class_header_after_ready_tag(self):
        self.cancel_startup(b'RDY0\x10')

    def test_runtime_cancel_partial_response_and_owner_finally_cleanup(self):
        self.write_worker(
            'os.write(1, b"RDY0"+struct.pack("<I", 16))\n'
            'request = b""\n'
            'while len(request) < 10:\n'
            '    request += os.read(0, 10-len(request))\n'
            'os.write(1, b"r\\x00\\x00")\n'
            'marker.write_text("waiting")\n'
            'time.sleep(60)\n')
        stop, errors = threading.Event(), []
        client = module.PTv3Client(cancel_event=stop, max_pts=8, response_timeout_s=120)

        def infer():
            try:
                client.segment(np.zeros((1, 4), np.float32))
            except Exception as error:
                errors.append(error)
            finally:
                client.close()

        thread = threading.Thread(target=infer, daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(timeout=2)))
        self.wait_for_marker()
        start = time.monotonic()
        stop.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), 'runtime cancellation/cleanup exceeded 2 seconds')
        self.assertLess(time.monotonic()-start, 2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertIsNone(client.shm)
        self.assert_cleaned()

    def test_already_cancelled_does_not_spawn_or_allocate_shm(self):
        stop = threading.Event()
        stop.set()
        with patch.object(shared_memory, 'SharedMemory') as allocate:
            with self.assertRaises(InterruptedError):
                module.PTv3Client(cancel_event=stop, max_pts=8)
        allocate.assert_not_called()
        self.assertEqual(self.processes, [])

    def test_unset_event_polling_does_not_create_spurious_timeouts(self):
        self.write_worker('time.sleep(.25)\nos.write(1, b"RDY0"+struct.pack("<I", 16))\n'
                          'assert os.read(0, 1) == b"Q"\n')
        client = module.PTv3Client(cancel_event=threading.Event(), max_pts=8, startup_timeout_s=2)
        self.assertEqual(client.num_classes, 16)
        client.close()
        self.assertGreaterEqual(len(self.select_waits), 3)
        self.assertLessEqual(max(self.select_waits), .1)
        self.assert_cleaned()

    def test_default_keeps_blocking_wait_and_worker_environment_contract(self):
        self.write_worker('time.sleep(.15)\nos.write(1, b"RDY0"+struct.pack("<I", 16))\n'
                          'assert os.read(0, 1) == b"Q"\n')
        with patch.dict(os.environ, {'PYTHONPATH': '/unwanted/ros',
                                     'PTV3_MEMORY_CHECKPOINT': '/unwanted/model',
                                     'PTV3_MEMORY_SHADOW': '1'}):
            client = module.PTv3Client(max_pts=8, startup_timeout_s=2)
        client.close()
        self.assertGreater(self.select_waits[0], 1)
        env = self.environments[0]
        for name in ('PYTHONPATH', 'PTV3_MEMORY_CHECKPOINT', 'PTV3_MEMORY_SHADOW'):
            self.assertNotIn(name, env)
        self.assertEqual(env['PTV3_SHM_MAXPTS'], '8')
        self.assert_cleaned()

    def test_polling_retains_original_timeout_and_cleans_up(self):
        self.write_worker('os.write(1, b"RD")\ntime.sleep(60)\n')
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            module.PTv3Client(cancel_event=threading.Event(), max_pts=8, startup_timeout_s=.25)
        self.assertLess(time.monotonic()-start, 2)
        self.assert_cleaned()

    def test_popen_and_stderr_open_failures_unlink_preallocated_shm(self):
        real_allocate = shared_memory.SharedMemory
        for failure in ('popen', 'stderr'):
            names = []

            def allocate(*args, **kwargs):
                shm = real_allocate(*args, **kwargs)
                names.append(shm.name)
                return shm

            with self.subTest(failure=failure), patch.object(shared_memory, 'SharedMemory', allocate):
                if failure == 'popen':
                    with patch.object(module.subprocess, 'Popen', side_effect=OSError('cannot spawn')):
                        with self.assertRaises(OSError):
                            module.PTv3Client(max_pts=8)
                else:
                    with self.assertRaises(FileNotFoundError):
                        module.PTv3Client(max_pts=8, stderr_path=self.root/'missing'/'worker.log')
            self.assertEqual(len(names), 1)
            with self.assertRaises(FileNotFoundError):
                real_allocate(name=names[0])


if __name__ == '__main__':
    unittest.main()
