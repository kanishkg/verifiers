# Patched vllm_serve.py for trl==0.17.0, integrating weight sync and chat functionality

import argparse
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)

if is_fastapi_available():
    from fastapi import FastAPI

if is_pydantic_available():
    from pydantic import BaseModel

if is_uvicorn_available():
    import uvicorn

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

logger = logging.getLogger(__name__)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    A vLLM **worker extension** that enables weight synchronization between a client and
    multiple server workers. This class should be passed to *worker_extension_cls* in
    the `LLM` constructor (vLLM ≥ 0.4).

    We still expose a legacy `WeightSyncWorker` alias for backward‑compatibility with
    code paths that expect a full custom worker class (pre‑0.4 style).
    """

    # These attributes are initialised by ``init_communicator``
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    # ---------------------------------------------------------------------
    # ♥  Communicator helpers
    # ---------------------------------------------------------------------
    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialised; call close_communicator() first.")
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1  # the external client is the last rank

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialised. Call init_communicator() first.")
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        # Hot‑swap the weights into the model
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None

# -------------------------------------------------------------------------
# Legacy compatibility shim –> provides the pre‑0.17 ``WeightSyncWorker``
# -------------------------------------------------------------------------
if is_vllm_available():
    try:
        from vllm.worker.worker import Worker as _BaseWorker
    except Exception:  # pragma: no cover
        _BaseWorker = object  # Fallback in exotic envs

    class WeightSyncWorker(_BaseWorker, WeightSyncWorkerExtension):  # type: ignore[misc]
        """*Alias* that combines the default vLLM Worker with our sync extension."""




@dataclass
class ScriptArguments:
    model: str
    revision: Optional[str] = None
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    kv_cache_dtype: str = "auto"
    log_level: str = "info"

def chunk_list(lst: list, n: int) -> list[list]:
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(n)]

def main(script_args: ScriptArguments):
    master_port = get_open_port()
    connections, processes = [], []
    for dp_rank in range(script_args.data_parallel_size):
        parent_conn, child_conn = Pipe()
        p = Process(target=llm_worker, args=(script_args, dp_rank, master_port, child_conn))
        p.start()
        connections.append(parent_conn)
        processes.append(p)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ready = set()
        while len(ready) < script_args.data_parallel_size:
            for conn in connections:
                msg = conn.recv()
                if msg.get("status") == "ready":
                    ready.add(conn)
        yield
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex) \
            if request.guided_decoding_regex else None
        sampling_params = SamplingParams(**request.dict(exclude={"prompts", "guided_decoding_regex"}),
                                         guided_decoding=guided_decoding)
        chunks = chunk_list(request.prompts, script_args.data_parallel_size)
        for conn, prompts in zip(connections, chunks):
            if not prompts:
                prompts = ["<placeholder>"]
            conn.send({"type": "call", "method": "generate", "kwargs": {"prompts": prompts, "sampling_params": sampling_params}})
        outputs = [conn.recv() for conn in connections if conn]
        outputs = list(chain.from_iterable(outputs))
        return {"completion_ids": [list(o.token_ids) for r in outputs for o in r.outputs]}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        for conn in connections:
            conn.send({"type": "fire_and_forget", "method": "collective_rpc",
                       "kwargs": {"method": "init_communicator",
                                  "args": (request.host, request.port, request.world_size)}})
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        dtype = getattr(torch, request.dtype.split(".")[-1])
        for conn in connections:
            conn.send({"type": "fire_and_forget", "method": "collective_rpc",
                       "kwargs": {"method": "update_named_param",
                                  "args": (request.name, dtype, tuple(request.shape))}})
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        for conn in connections:
            conn.send({"type": "call", "method": "reset_prefix_cache"})
        results = [conn.recv() for conn in connections]
        return {"message": f"Reset status: {all(results)}"}

    @app.post("/close_communicator/")
    async def close_communicator():
        for conn in connections:
            conn.send({"type": "fire_and_forget", "method": "collective_rpc",
                       "kwargs": {"method": "close_communicator"}})
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)

def llm_worker(script_args: ScriptArguments, dp_rank: int, master_port: int, conn: Connection):
    os.environ.update({
        "VLLM_DP_RANK": str(dp_rank),
        "VLLM_DP_RANK_LOCAL": str(dp_rank),
        "VLLM_DP_SIZE": str(script_args.data_parallel_size),
        "VLLM_DP_MASTER_PORT": str(master_port),
    })
    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
    )
    conn.send({"status": "ready"})
    while True:
        try:
            cmd = conn.recv()
        except KeyboardInterrupt:
            llm.collective_rpc("close_communicator")
            break
        if cmd["type"] == "shutdown":
            break
        method = getattr(llm, cmd["method"])
        result = method(*cmd.get("args", ()), **cmd.get("kwargs", {})) if cmd["type"] == "call" else method(*cmd.get("args", ()), **cmd.get("kwargs", {}))
        if cmd["type"] == "call":
            conn.send(result)

def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        return subparsers.add_parser("vllm-serve", help="Run vLLM serve script", dataclass_types=ScriptArguments)
    return TrlParser(ScriptArguments)

if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
