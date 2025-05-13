# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Patched version of vllm_client.py for trl==0.17.0
import atexit
import logging
import time
from typing import Optional

import torch
from torch import nn

from trl.import_utils import is_requests_available, is_vllm_ascend_available, is_vllm_available

if is_requests_available():
    import requests
    from requests import ConnectionError

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

logger = logging.getLogger(__name__)

class VLLMClient:
    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                if time.time() - start_time >= total_timeout:
                    raise ConnectionError(
                        f"Cannot reach vLLM server at {self.host}:{self.server_port} after {total_timeout} seconds."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return
            logger.info(f"Waiting for server... retrying in {retry_interval} seconds.")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[int]]:
        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        raise Exception(f"Generate request failed: {response.status_code}, {response.text}")

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        stop: Optional[list[str]] = None,
        include_stop_str_in_output: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> dict[str, list]:
        url = f"http://{self.host}:{self.server_port}/chat/"
        response = self.session.post(
            url,
            json={
                "messages": messages,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "stop": stop,
                "include_stop_str_in_output": include_stop_str_in_output,
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
            },
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Chat request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        url = f"http://{self.host}:{self.server_port}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"World size request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1
        self.rank = vllm_world_size

        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Init communicator failed: {response.status_code}, {response.text}")

        time.sleep(0.1)
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Update named param failed: {response.status_code}, {response.text}")

        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Reset prefix cache failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        url = f"http://{self.host}:{self.server_port}/close_communicator/"
        try:
            response = self.session.post(url)
            if response.status_code != 200:
                raise Exception(f"Close communicator failed: {response.status_code}, {response.text}")
        except ConnectionError:
            pass  # Server likely already down

# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()
    client.init_communicator()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)