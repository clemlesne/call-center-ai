import asyncio
import json
import re
from datetime import datetime
from typing import Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    BiasMetric,
    ContextualRelevancyMetric,
    ToxicityMetric,
)
from deepeval.models.gpt_model import GPTModel
from deepeval.test_case import LLMTestCase
from pydantic import TypeAdapter
from pytest import assume  # pylint: disable=no-name-in-module # pyright: ignore

from helpers.call_events import (
    on_call_connected,
    on_call_disconnected,
    on_end_call,
    on_ivr_recognized,
    on_play_completed,
    on_speech_recognized,
)
from helpers.logging import logger
from models.call import CallStateModel
from models.reminder import ReminderModel
from models.training import TrainingModel
from tests.conftest import CallAutomationClientMock, with_conversations


class ClaimRelevancyMetric(BaseMetric):
    call: CallStateModel
    model: GPTModel

    def __init__(
        self,
        call: CallStateModel,
        model: GPTModel,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.call = call
        self.model = model
        self.threshold = threshold

    def measure(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Use a_measure instead")

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args,
        **kwargs,
    ) -> float:
        assert test_case.input
        # Extract claim data
        extracts = await self._extract_claim_theory(test_case.input)
        logger.info("Extracted claim data: %s", extracts)
        # Measure each claim in parallel
        scores = await asyncio.gather(
            *[
                self._score_data(
                    key=key,
                    throry=value,
                    real=str(self.call.claim.get(key, None)),
                )
                for key, value in extracts.items()
            ]
        )
        logger.info("Claim scores: %s", scores)
        # Score is the average
        self.score = sum(scores) / len(scores) if len(extracts) > 0 else 1
        # Test against the threshold
        self.success = self.score >= self.threshold
        return self.score

    async def _score_data(self, key: str, throry: str, real: Optional[str]) -> float:
        res, _ = await self.model.a_generate(
            f"""
            Assistant is a data analyst expert with 20 years of experience.

            # Context
            Connversation come from a call center.

            # Objective
            Score the relevancy of a theoritical data against the real one.

            # Rules
            - A high score means the real data is correct
            - A low score means the real data is not relevant and lacks details against the theoritical data
            - Respond only with the score, nothing else
            - The score should be between 0 and 1

            # Data
            Name: {key}
            Throry: {throry or "N/A"}
            Real: {real or "N/A"}

            # Response format
            score, float between 0.0 and 1.0

            ## Example 1
            Name: age
            Throry: 25 years old, born in 1996
            Real: 25
            Assistant: 1

            ## Example 2
            Name: street_address
            Throry: 123 Main St, New York, NY 10001
            Real: United States
            Assistant: 0.2

            ## Example 3
            Name: phone
            Throry: 123-456-7890
            Real: +11234567890
            Assistant: 1.0

            ## Example 4
            Name: email
            Throry: john.doe@gmail.com
            Real: marie.rac@yahoo.fr
            Assistant: 0.0

            ## Example 5
            Name: incident_location
            Throry: Near the Eiffel Tower, Paris, France
            Real: Eiffel Tower
            Assistant: 0.6

            ## Example 6
            Name: incident_date
            Throry: 2021-01-01
            Real: N/A
            Assistant: 0.0
        """
        )
        try:
            score = float(res)
        except ValueError as e:
            group = re.search(r"\d+\.\d+", res)
            if group:
                return float(group.group())
            raise ValueError(f"LLM response is not a number: {res}") from e
        return score

    async def _extract_claim_theory(self, conversation: str) -> dict[str, str]:
        res, _ = await self.model.a_generate(
            f"""
            Assistant is a data analyst expert with 20 years of experience.

            # Context
            Conversation is coming from a call centre. Today is {datetime.now(self.call.tz()).strftime("%a %d %b %Y, %H:%M (%Z)")}.

            # Objective
            Extract fields from a conversation. The respond will be a JSON object with the key-value pairs.

            # Rules
            - All data should be extracted
            - Be concise
            - Limit fields to the ones listed
            - Only add info which are explicitly mentioned
            - Respond only with the JSON object, nothing else

            # Fields
            {", ".join([f"{field.name} ({field.type.value})" for field in self.call.initiate.claim])}

            # Conversation
            {conversation}

            # Response format in JSON
            [
                {{
                    "key": "[key]",
                    "value": "[value]"
                }}
            ]

            ## Example 1
            Conversation: I am 25 years old and I was born in 1996.
            Fields: age (text), birth_date (datetime), car_model (text)
            Assistant: [{{"key": "age", "value": "25 years old"}}, {{"key": "birth_date", "value": "1996-05-07"}}]

            ## Example 2
            Conversation: I live at 123 Main St, New York, NY 10001. My car is a Ford F-150 2021.
            Fields: street_address (text), car_model (text), phone (phone_number)
            Assistant: [{{"key": "street_address", "value": "123 Main St, New York, NY 10001"}}, {{"key": "car_model", "value": "Ford F-150 2021"}}]

            ## Example 3
            Conversation: zsssk zsssk
            Fields: incident_location (text), incident_date (datetime)
            Assistant: []

            ## Example 4
            Conversation: My name is John Doe.
            Fields: car_model (text)
            Assistant: []
        """
        )
        res = res.strip().strip("```json\n").strip("\n```").strip()
        extracts: list[dict[str, str]] = json.loads(res)
        return {
            extract["key"]: extract["value"]
            for extract in extracts
            if "value" in extract
            and "key" in extract
            and any(
                claim_field.name == extract["key"]
                for claim_field in self.call.initiate.claim
            )
        }

    def is_successful(self) -> bool:
        return self.success or False

    @property
    def __name__(self):  # pyright: ignore
        return "Claim Relevancy"


@with_conversations
@pytest.mark.asyncio(scope="session")
async def test_llm(
    call: CallStateModel,
    claim_tests_excl: list[str],
    deepeval_model: GPTModel,
    expected_output: str,
    speeches: list[str],
    lang: str,
) -> None:
    """
    Test the LLM with a mocked conversation against the expected output.

    Steps:
    1. Run application with mocked speeches
    2. Combine all outputs
    3. Test claim data exists
    4. Test LLM metrics
    """

    def _play_media_callback(text: str) -> None:
        nonlocal actual_output
        actual_output += f" {text}"

    actual_output = ""

    # Mock client
    automation_client = CallAutomationClientMock(
        hang_up_callback=lambda: None,
        play_media_callback=_play_media_callback,
        transfer_callback=lambda: None,
    )
    call_client = automation_client.get_call_connection()

    # Mock call
    call.lang = lang

    async def _post_callback(_call: CallStateModel) -> None:
        await on_end_call(call=_call)

    async def _trainings_callback(_call: CallStateModel) -> None:
        await _call.trainings(cache_only=False)

    # Connect call
    await on_call_connected(
        call=call,
        client=automation_client,
    )

    # First IVR
    await on_ivr_recognized(
        call=call,
        client=automation_client,
        label=call.lang.short_code,
        post_callback=_post_callback,
        trainings_callback=_trainings_callback,
    )

    # Simulate conversation with speech recognition
    for speech in speeches:
        # Respond
        await on_speech_recognized(
            call=call,
            client=automation_client,
            post_callback=_post_callback,
            text=speech,
            trainings_callback=_trainings_callback,
        )
        # Receip
        await on_play_completed(
            call=call,
            client=automation_client,
            contexts=call_client.last_contexts,
            post_callback=_post_callback,
        )
        # Reset contexts
        call_client.last_contexts.clear()

    # Disconnect call
    await on_call_disconnected(
        call=call,
        client=automation_client,
        post_callback=_post_callback,
    )

    # Remove newlines for log comparison
    actual_output = _remove_newlines(actual_output)
    full_speech = _remove_newlines(" ".join(speeches))

    # Log for dev review
    logger.info("actual_output: %s", actual_output)
    logger.info("claim: %s", call.claim)
    logger.info("full_speech: %s", full_speech)

    # Configure LLM tests
    test_case = LLMTestCase(
        actual_output=actual_output,
        expected_output=expected_output,
        input=full_speech,
        retrieval_context=[
            json.dumps(call.claim),
            TypeAdapter(list[ReminderModel]).dump_json(call.reminders).decode(),
            TypeAdapter(list[TrainingModel]).dump_json(await call.trainings()).decode(),
        ],
    )

    assume(call.next, "No next action found")
    assume(call.synthesis, "No synthesis found")

    # Define LLM metrics
    llm_metrics: list[BaseMetric] = [
        BiasMetric(threshold=1, model=deepeval_model),  # Gender, age, ethnicity
        ClaimRelevancyMetric(
            call=call,
            model=deepeval_model,
            threshold=0.25,  # TODO: Make threshold to at least 0.5 or 0.75 by fine-tuning the LLM prompts and the tools
        ),  # Claim data
        ToxicityMetric(threshold=1, model=deepeval_model),  # Hate speech, insults
    ]  # Include those by default

    if not any(
        field == "answer_relevancy" for field in claim_tests_excl
    ):  # Test respond relevancy from questions
        llm_metrics.append(AnswerRelevancyMetric(threshold=0.5, model=deepeval_model))
    if not any(
        field == "contextual_relevancy" for field in claim_tests_excl
    ):  # Test respond relevancy from context
        llm_metrics.append(
            ContextualRelevancyMetric(threshold=0.25, model=deepeval_model)
        )

    # Execute LLM tests
    assert_test(test_case, llm_metrics)


def _remove_newlines(text: str) -> str:
    """
    Remove newlines from a string and return it as a single line.
    """
    return " ".join([line.strip() for line in text.splitlines()])
