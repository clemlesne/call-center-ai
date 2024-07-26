import hashlib
import random
import string
import xml.etree.ElementTree as ET
from textwrap import dedent
from typing import Any, Callable, Optional, Union

import pytest
import yaml
from _pytest.mark.structures import MarkDecorator
from azure.communication.callautomation import FileSource, SsmlSource, TextSource
from azure.communication.callautomation._models import TransferCallResult
from azure.communication.callautomation.aio import (
    CallAutomationClient,
    CallConnectionClient,
)
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from deepeval.models.gpt_model import GPTModel
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, ValidationError

from function_app import _str_to_contexts
from helpers.call_utils import ContextEnum as CallContextEnum
from helpers.config import CONFIG
from helpers.logging import logger
from models.call import CallInitiateModel, CallStateModel


class CallConnectionClientMock(CallConnectionClient):
    _hang_up_callback: Callable[[], None]
    _play_media_callback: Callable[[str], None]
    _transfer_callback: Callable[[], None]

    last_contexts: set[CallContextEnum] = set()

    def __init__(
        self,
        hang_up_callback: Callable[[], None],
        play_media_callback: Callable[[str], None],
        transfer_callback: Callable[[], None],
    ) -> None:
        self._hang_up_callback = hang_up_callback
        self._play_media_callback = play_media_callback
        self._transfer_callback = transfer_callback

    async def start_recognizing_media(
        self,
        play_prompt: Union[FileSource, TextSource, SsmlSource],
        *args,
        operation_context: Optional[str] = None,
        **kwargs,
    ) -> None:
        contexts = _str_to_contexts(operation_context)
        for context in contexts or []:
            self.last_contexts.add(context)
        self._log_media(play_prompt)

    async def play_media(
        self,
        play_source: Union[FileSource, TextSource, SsmlSource],
        *args,
        operation_context: Optional[str] = None,
        **kwargs,
    ) -> None:
        contexts = _str_to_contexts(operation_context)
        for context in contexts or []:
            self.last_contexts.add(context)
        self._log_media(play_source)

    async def transfer_call_to_participant(
        self,
        *args,
        **kwargs,
    ) -> TransferCallResult:
        self._transfer_callback()
        return TransferCallResult()

    async def hang_up(
        self,
        *args,
        **kwargs,
    ) -> None:
        self._hang_up_callback()

    async def cancel_all_media_operations(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    def _log_media(
        self, play_source: Union[FileSource, TextSource, SsmlSource]
    ) -> None:
        if isinstance(play_source, TextSource):
            self._play_media_callback(play_source.text.strip())
        elif isinstance(play_source, SsmlSource):
            # deepcode ignore InsecureXmlParser/test: SSML is internally generated
            for text in ET.fromstring(play_source.ssml_text).itertext():
                if text.strip():
                    self._play_media_callback(text.strip())


class CallAutomationClientMock(CallAutomationClient):
    _call_client: CallConnectionClientMock

    def __init__(
        self,
        hang_up_callback: Callable[[], None],
        play_media_callback: Callable[[str], None],
        transfer_callback: Callable[[], None],
    ) -> None:
        self._call_client = CallConnectionClientMock(
            hang_up_callback=hang_up_callback,
            play_media_callback=play_media_callback,
            transfer_callback=transfer_callback,
        )

    def get_call_connection(
        self,
        *args,
        **kwargs,
    ) -> CallConnectionClientMock:
        return self._call_client


class DeepEvalAzureOpenAI(GPTModel):
    _cache: pytest.Cache
    _langchain_kwargs: dict[str, Any]
    _model: AzureChatOpenAI

    def __init__(
        self,
        cache: pytest.Cache,
        **kwargs,
    ):
        platform = CONFIG.llm.fast.azure_openai
        assert platform

        _langchain_kwargs = {
            # Repeatability
            "model_kwargs": {
                "seed": 42,
            },
            "temperature": 0,
            # Deployment
            "api_version": "2023-12-01-preview",
            "azure_deployment": platform.deployment,
            "azure_endpoint": platform.endpoint,
            "model": platform.model,
            # Authentication, either RBAC or API
            "api_key": (
                platform.api_key.get_secret_value() if platform.api_key else None
            ),  # pyright: ignore
            "azure_ad_token_provider": (
                get_bearer_token_provider(
                    ManagedIdentityCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                if not platform.api_key
                else None
            ),
            # DeepEval
            **kwargs,
        }
        self._cache = cache
        self._model = AzureChatOpenAI(**_langchain_kwargs)

    def generate(self, prompt: str) -> tuple[str, float]:
        prompt = dedent(prompt).strip()
        cache_key = self._cache_key(prompt)
        # Try cache
        content: tuple[str, float] = self._cache.get(cache_key, None)
        if content:
            return content
        # Try live
        res = super().generate(prompt)
        # Update cache
        self._cache.set(cache_key, res)
        return res

    async def a_generate(self, prompt: str) -> tuple[str, float]:
        prompt = dedent(prompt).strip()
        cache_key = self._cache_key(prompt)
        # Try cache
        content: tuple[str, float] = self._cache.get(cache_key, None)
        if content:
            return content
        # Try live
        res = await super().a_generate(prompt)
        # Update cache
        self._cache.set(cache_key, res)
        return res

    def get_model_name(self) -> str:
        return "Azure OpenAI"

    def load_model(self) -> AzureChatOpenAI:
        return self._model

    def should_use_azure_openai(self) -> bool:
        return True

    def _cache_key(self, prompt: str) -> str:
        llm_string = self._model._get_llm_string(
            input=prompt
        )  # pylint: disable=protected-access
        llm_hash = hashlib.sha256(
            llm_string.encode(), usedforsecurity=False
        ).hexdigest()
        return f"call-center-ai/{llm_hash}"


class Conversation(BaseModel):
    claim_tests_excl: list[str] = []
    expected_output: str
    id: str
    speeches: list[str]
    lang: str


def with_conversations(fn=None) -> MarkDecorator:
    with open(
        encoding="utf-8",
        file="tests/conversations.yaml",
        mode="r",
    ) as f:
        file: dict = yaml.safe_load(f)
    conversations: list[Conversation] = []
    for conv in file.get("conversations", []):
        try:
            conversations.append(Conversation.model_validate(conv))
        except ValidationError:
            logger.error("Failed to parse conversation", exc_info=True)
    print(f"Loaded {len(conversations)} conversations")
    keys = sorted(Conversation.model_fields.keys() - {"id"})
    values = [
        pytest.param(
            *[conversation.model_dump()[key] for key in keys],
            id=conversation.id,
        )
        for conversation in conversations
    ]
    decorator = pytest.mark.parametrize(keys, values)
    if fn:
        return decorator(fn)
    return decorator


@pytest.fixture
def random_text() -> str:
    text = "".join(random.choice(string.printable) for _ in range(100))
    return text


@pytest.fixture
def call() -> CallStateModel:
    call = CallStateModel(
        initiate=CallInitiateModel(
            **CONFIG.conversation.initiate.model_dump(),
            phone_number="+33612345678",  # pyright: ignore
        ),
        voice_id="dummy",
    )
    return call


@pytest.fixture
def deepeval_model(cache: pytest.Cache) -> GPTModel:
    return DeepEvalAzureOpenAI(cache)
