"""End-to-end test for mooncake_master --put_wait_for_space.

Each test starts its own master, a provider that lends a small memory segment
and a client, all over TCP on localhost. With put_wait_for_space the master
parks a put that finds no memory instead of failing it with
NO_AVAILABLE_HANDLE (-200), and the client waits for it without an RPC
timeout (MC_PUT_START_RPC_TIMEOUT_MS=-1).
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MC_FORCE_TCP", "1")
os.environ.setdefault("MC_MS_AUTO_DISC", "0")
os.environ["MC_PUT_START_RPC_TIMEOUT_MS"] = "-1"

import mooncake
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

MiB = 1024 * 1024
SEGMENT_SIZE = 32 * MiB
NO_AVAILABLE_HANDLE = -200

PROVIDER = """
import sys
from mooncake.engine import TransferEngine
from mooncake.store import MooncakeDistributedStore
engine = TransferEngine()
assert engine.initialize('127.0.0.1', 'P2PHANDSHAKE', 'tcp', '') == 0
store = MooncakeDistributedStore()
assert store.setup(local_hostname='127.0.0.1', metadata_server='P2PHANDSHAKE',
                   global_segment_size=int(sys.argv[2]), local_buffer_size=64 * 1024 * 1024,
                   protocol='tcp', rdma_devices='', master_server_addr=sys.argv[1],
                   engine=engine.get_engine(), enable_ssd_offload=sys.argv[3] == '1') == 0
print('PROVIDER_READY', flush=True)
sys.stdin.readline()
store.close()
"""


def master_binary():
    found = shutil.which("mooncake_master")
    if found:
        return found
    return str(Path(mooncake.__file__).parent / "mooncake_master")


def open_log(path):
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


class Cluster:
    """A master, one memory provider and a client store."""

    def __init__(self, master_flags, ssd_offload=False):
        self.workdir = Path(tempfile.mkdtemp(prefix="put_wait_"))
        rpc_port, metrics_port = free_ports(2)
        self.address = f"127.0.0.1:{rpc_port}"
        self.log_fds = [open_log(self.workdir / "master.log")]
        self.master = subprocess.Popen(
            [
                master_binary(),
                "--rpc_address=127.0.0.1",
                f"--rpc_port={rpc_port}",
                "--metrics_host=127.0.0.1",
                f"--metrics_port={metrics_port}",
                "--put_wait_for_space=true",
                *master_flags,
            ],
            stdout=self.log_fds[0],
            stderr=subprocess.STDOUT,
        )
        self.provider = None
        self.store = None
        try:
            self._wait_for_port(rpc_port)
            self._start_provider(ssd_offload)
            self.store = MooncakeDistributedStore()
            assert (
                self.store.setup(
                    {
                        "local_hostname": "127.0.0.1",
                        "metadata_server": "P2PHANDSHAKE",
                        "master_server_addr": self.address,
                        "global_segment_size": 0,
                        "local_buffer_size": 64 * MiB,
                        "protocol": "tcp",
                        "enable_ssd_offload": False,
                    }
                )
                == 0
            )
        except BaseException:
            self.close()
            raise

    def _wait_for_port(self, port):
        deadline = time.monotonic() + 20
        while True:
            assert self.master.poll() is None, "master exited"
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    return
            assert time.monotonic() < deadline, "master startup timed out"
            time.sleep(0.1)

    def _start_provider(self, ssd_offload):
        env = dict(os.environ)
        if ssd_offload:
            env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] = str(self.workdir / "ssd")
            # The default bucket backend holds small objects back until a whole
            # bucket fills; write each key at once instead.
            env["MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"] = (
                "file_per_key_storage_backend"
            )
            env["MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS"] = "1"
            env["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = str(64 * MiB)
            (self.workdir / "ssd").mkdir()
        self.log_fds.append(open_log(self.workdir / "provider.log"))
        self.provider = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-u",
                "-c",
                PROVIDER,
                self.address,
                str(SEGMENT_SIZE),
                "1" if ssd_offload else "0",
            ],
            stdin=subprocess.PIPE,
            stdout=self.log_fds[-1],
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + 30
        while True:
            if "PROVIDER_READY" in (self.workdir / "provider.log").read_text():
                return
            assert self.provider.poll() is None, "provider exited"
            assert time.monotonic() < deadline, "provider startup timed out"
            time.sleep(0.1)

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None
        if self.provider is not None and self.provider.poll() is None:
            try:
                self.provider.communicate("\n", timeout=10)
            except subprocess.TimeoutExpired:
                self.provider.kill()
                self.provider.wait()
        self.master.terminate()
        try:
            self.master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.master.kill()
            self.master.wait()
        for fd in self.log_fds:
            os.close(fd)
        shutil.rmtree(self.workdir, ignore_errors=True)


def replicate_config(hard_pin):
    config = ReplicateConfig()
    config.replica_num = 1
    config.with_hard_pin = hard_pin
    return config


class PutWaitForSpaceTest(unittest.TestCase):
    def start(self, *master_flags, ssd_offload=False):
        cluster = Cluster(master_flags, ssd_offload=ssd_offload)
        self.addCleanup(cluster.close)
        return cluster

    def test_put_waits_until_removal_frees_space(self):
        cluster = self.start(
            "--eviction_ratio=0",
            "--eviction_high_watermark_ratio=1",
            "--default_kv_lease_ttl=500",
        )
        store = cluster.store
        config = replicate_config(hard_pin=True)
        size = 5 * MiB
        # Six fit in the segment; a seventh cannot.
        for i in range(6):
            self.assertEqual(store.put(f"fill-{i}", os.urandom(size), config=config), 0)

        value = os.urandom(size)
        outcome = {}
        late = threading.Thread(
            target=lambda: outcome.update(ret=store.put("late", value, config=config))
        )
        late.start()
        late.join(timeout=1.5)
        self.assertTrue(late.is_alive(), "the put should wait for space")

        self.assertEqual(store.remove("fill-0", force=True), 0)
        late.join(timeout=30)
        self.assertFalse(late.is_alive(), "the put should finish once space frees up")
        self.assertEqual(outcome["ret"], 0)
        self.assertEqual(store.get("late"), value)

    def test_concurrent_puts_beyond_capacity_succeed_with_eviction(self):
        cluster = self.start(
            "--eviction_ratio=0.2",
            "--eviction_high_watermark_ratio=0.9",
            "--default_kv_lease_ttl=500",
        )
        store = cluster.store
        config = replicate_config(hard_pin=False)
        size = 2 * MiB
        # Five times the segment, from eight threads at once.
        keys = [f"evict-{i}" for i in range(80)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda key: store.put(key, os.urandom(size), config=config), keys
                )
            )
        self.assertEqual(results, [0] * len(keys))

    def test_puts_beyond_capacity_succeed_and_stay_readable_with_offload(self):
        cluster = self.start(
            "--enable_offload=true",
            "--offload_on_evict=true",
            "--eviction_ratio=0.2",
            "--eviction_high_watermark_ratio=0.9",
            "--default_kv_lease_ttl=500",
            ssd_offload=True,
        )
        store = cluster.store
        config = replicate_config(hard_pin=False)
        size = 2 * MiB
        values = {f"offload-{i}": os.urandom(size) for i in range(40)}

        for key, value in values.items():
            self.assertEqual(store.put(key, value, config=config), 0, key)
        for key, value in values.items():
            self.assertEqual(store.get(key), value, key)


if __name__ == "__main__":
    unittest.main()
