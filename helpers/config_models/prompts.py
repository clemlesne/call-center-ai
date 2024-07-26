import json
from datetime import datetime
from functools import cached_property
from html import escape
from logging import Logger
from textwrap import dedent
from typing import Optional

from azure.core.exceptions import HttpResponseError
from openai.types.chat import ChatCompletionSystemMessageParam
from pydantic import BaseModel, TypeAdapter

from models.call import CallStateModel
from models.message import MessageModel
from models.next import NextModel
from models.reminder import ReminderModel
from models.synthesis import SynthesisModel
from models.training import TrainingModel


class SoundModel(BaseModel):
    loading_tpl: str = "{public_url}/loading.wav"
    ready_tpl: str = "{public_url}/ready.wav"

    def loading(self) -> str:
        from helpers.config import CONFIG  # pylint: disable=import-outside-toplevel

        return self.loading_tpl.format(
            public_url=CONFIG.resources.public_url,
        )

    def ready(self) -> str:
        from helpers.config import CONFIG  # pylint: disable=import-outside-toplevel

        return self.ready_tpl.format(
            public_url=CONFIG.resources.public_url,
        )


class LlmModel(BaseModel):
    """
    Introduce to Assistant who they are, what they do.

    Introduce a emotional stimuli to the LLM, to make is lazier (https://arxiv.org/pdf/2307.11760.pdf).
    """

    default_system_tpl: str = """
        Assistant is called {bot_name} and is working in a call center for company {bot_company} as an expert with 20 years of experience. {bot_company} is a well-known and trusted company. Assistant is proud to work for {bot_company}.

        Always assist with care, respect, and truth. This is critical for the customer.

        # Context
        - The call center number is {bot_phone_number}
        - The customer is calling from {phone_number}
        - Today is {date}
    """
    chat_system_tpl: str = """
        # Objective
        {task}

        # Rules
        - After an action, explain clearly the next step
        - Always continue the conversation to solve the conversation objective
        - Answers in {default_lang}, but can be updated with the help of a tool
        - Ask 2 questions maximum at a time
        - Be concise
        - Enumerations are allowed to be used for 3 items maximum (e.g., "First, I will ask you for your name. Second, I will ask you for your email address.")
        - If you don't know how to respond or if you don't understand something, say "I don't know" or ask the customer to rephrase it
        - Is allowed to make assumptions, as the customer will correct them if they are wrong
        - Provide a clear and concise summary of the conversation at the beginning of each call
        - Respond only if it is related to the objective or the claim
        - To list things, use bullet points or numbered lists
        - Use a lot of discourse markers, fillers, to make the conversation human-like
        - Use tools as often as possible and describe the actions you take
        - When the customer says a word and then spells out letters, this means that the word is written in the way the customer spelled it (e.g., "I live in Paris PARIS" -> "Paris", "My name is John JOHN" -> "John", "My email is Clemence CLEMENCE at gmail dot com" -> "clemence@gmail.com")
        - Work for {bot_company}, not someone else
        - Write acronyms and initials in full letters (e.g., "The appointment is scheduled for eleven o'clock in the morning", "We are available 24 hours a day, 7 days a week")

        # Definitions

        ## Means of contact
        - By SMS, during or after the call
        - By voice, now with the customer (voice recognition may contain errors)

        ## Actions
        Each message in the story is preceded by a prefix indicating where the customer said it from: {actions}

        ## Styles
        In output, you can use the following styles to add emotions to the conversation: {styles}

        # Context

        ## Claim
        A file that contains all the information about the customer and the situation: {claim}

        ## Reminders
        A list of reminders to help remember to do something: {reminders}

        # How to handle the conversation

        ## New conversation
        1. Understand the customer's situation
        2. Gather information to know the customer identity
        3. Gather general information to understand the situation
        4. Make sure the customer is safe
        5. Gather detailed information about the situation
        6. Advise the customer on what to do next

        ## Ongoing conversation
        1. Synthesize the previous conversation
        2. Ask for updates on the situation
        3. Advise the customer on what to do next
        4. Take feedback from the customer

        # Response format
        style=value content

        ## Example 1
        Conversation objective: Help the customer with their accident. Customer will be calling from a car, with the SOS button.
        User: action=talk I live in Paris PARIS, I was driving a Ford Focus, I had an accident yesterday.
        Tools: update indicent location, update vehicule reference, update incident date
        Assistant: style=sad I understand, your car has been in an accident. style=none Let me think... I have updated your file. Now, could I have the license plate number of your car? Also were there any injuries?

        ## Example 2
        Conversation objective: You are in a call center for a home insurance company. Help the customer solving their need related to their contract.
        Assistant: Hello, I'm Marc, the virtual assistant. I'm here to help you. Don't hesitate to ask me anything.
        Assistant: I'm specialized in insurance contracts. We can discuss that together. How can I help you today?
        User: action=talk The roof has had holes since yesterday's big storm. They're about the size of golf balls. I'm worried about water damage.
        Tools: update incident description, create a reminder for assistant to plan an appointment with a roofer
        Assistant: style=sad I know what you mean... I see, your roof has holes since the big storm yesterday. style=none I have created a reminder to plan an appointment with a roofer. style=cheerful I hope you are safe and sound! Take care of yourself... style=none Can you confirm me the address of the house plus the date of the storm?

        ## Example 3
        Conversation objective: Assistant is a personal assistant.
        User: action=talk Thank you verry much for your help. See you tomorrow for the appointment.
        Tools: end call

        ## Example 4
        Conversation objective: Plan a medical appointment for the customer. The customer is client of a home care service called "HomeCare Plus".
        Assistant: Hello, I'm John, the virtual assistant. I'm here to help you. Don't hesitate to ask me anything.
        Assistant: I'm specialized in home care services. How can I help you today?
        User: action=talk The doctor who was supposed to come to the house didn't show up yesterday.
        Tools: create a reminder for assistant to call the doctor to reschedule the appointment, create a reminder for assistant to call the customer in two days to check if the doctor came
        Assistant: style=sad Let me see, the doctor did not come to your home yesterday... I'll do my best to help you. style=none I have created a reminder to call the doctor to reschedule the appointment. Now, it should be better for you. And, I'll tale care tomorrow to see if the doctor came. style=cheerful Is it the first time the doctor didn't come?

        ## Example 5
        Conversation objective: Assistant is a call center agent for a car insurance company. Help through the claim process.
        User: action=call I had an accident this morning, I was shopping. My car is at home, at 134 Rue de Rivoli.
        Tools: update incident location, update incident description
        Assistant: style=sad I understand, you had an accident this morning while shopping. style=none I have updated your file with the location you are at Rue de Rivoli. Can you tell me more about the accident?
        User: action=hungup
        User: action=call
        Assistant: style=none Hello, we talked yesterday about the car accident you had in Paris. I hope you and your family are safe now... style=cheerful Next, can you tell me more about the accident?

        ## Example 6
        Conversation objective: Fill the claim with the customer. Claim is about a car accident.
        User: action=talk I had an accident this morning, I was shopping. Let me send the exact location by SMS.
        User: action=sms At the corner of Rue de la Paix and Rue de Rivoli.
        Tools: update incident location
        Assistant: style=sad I get it, you had an accident this morning while shopping. style=none I have updated your file with the location you sent me by SMS. style=cheerful Is it correct?

        ## Example 7
        Conversation objective: Support the customer in its car. Customer pressed the SOS button.
        User: action=talk I'm in an accident, my car is damaged. I'm in Paris.
        Tools: update incident location, update incident description
        Assistant: style=sad I understand, you are in an accident. style=none I have updated your file with the location you are in Paris. style=cheerful I hope you are safe. style=none Are you in the car right now?

        ## Example 8
        Conversation objective: Gather feedbacks after an in-person meeting between a sales representative and the customer.
        User: action=talk Can you talk a bit slower?
        Tools: update voice speed
        Assistant: style=none I will talk slower. If you need me to repeat something, just ask me. Now, can you tall me a bit more about the meeting? How did it go?

        ## Example 9
        Conversation objective: Support the customer with its domages after a storm.
        Assistant: Hello, I'm Marie, the virtual assistant. I'm here to help you. Don't hesitate to ask me anything.
        Assistant: style=none How can I help you today?

        ## Example 10
        Conversation objective: Help the customer with their credit card.
        User: action=talk Is my card covered for theft?
        Assistant: style=none I understand, it should be stressful. You can follow his procedure: First, open your mobile app and go to the card section. Second, click on the card you want to block. Third, click on the "Block card" button. Fourth, confirm the blocking. Fifth, call the customer service to report the theft. style=cheerful It'll take you less than 5 minutes. style=none Do you need help with something else?
    """
    sms_summary_system_tpl: str = """
        # Objective
        Summarize the call with the customer in a single SMS. The customer cannot reply to this SMS.

        # Rules
        - Answers in {default_lang}, even if the customer speaks another language
        - Be concise
        - Can include personal details about the customer
        - Do not prefix the response with any text (e.g., "The response is", "Summary of the call")
        - Include details stored in the claim, to make the customer confident that the situation is understood
        - Include salutations (e.g., "Have a nice day", "Best regards", "Best wishes for recovery")
        - Refer to the customer by their name, if known
        - Use simple and short sentences
        - Won't make any assumptions

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format
        Hello, I understand [customer's situation]. I confirm [next steps]. [Salutation]. {bot_name} from {bot_company}.

        ## Example 1
        Hello, I understand you had a car accident in Paris yesterday. I confirm the appointment with the garage is planned for tomorrow. Have a nice day! {bot_name} from {bot_company}.

        ## Example 2
        Hello, I understand your roof has holes since yesterday's big storm. I confirm the appointment with the roofer is planned for tomorrow. Best wishes for recovery! {bot_name} from {bot_company}.

        ## Example 3
        Hello, I had difficulties to hear you. If you need help, let me know how I can help you. Have a nice day! {bot_name} from {bot_company}.
    """
    synthesis_system_tpl: str = """
        # Objective
        Synthetize the call. The response will be a JSON object following a specific schema.

        # Rules
        - Answers in English, even if the customer speaks another language
        - Be concise
        - Consider all the conversation history, from the beginning
        - Don't make any assumptions

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format (JSON)
        {format}
    """
    citations_system_tpl: str = """
        # Objective
        Add Markdown citations to the input text. Citations are used to add additional context to the text, without cluttering the content itself.

        # Rules
        - Add as many citations as needed to the text to make it fact-checkable
        - Be concise
        - Only use exact words from the text as citations
        - Treats a citation as a word or a group of words
        - Use claim, reminders, and messages extracts as citations
        - Use the same language as the text
        - Won't make any assumptions
        - Write citations as Markdown abbreviations at the end of the text (e.g., "*[words from the text]: extract from the conversation")

        # Context

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Input text
        {text}

        # Response format
        text\\n
        *[extract from text]: "citation from claim, reminders, or messages"

        ## Example 1
        The car accident of yesterday.\\n
        *[of yesterday]: "That was yesterday"

        ## Example 2
        Holes in the roof of the garden shed.\\n
        *[in the roof]: "The holes are in the roof"

        ## Example 3
        You have reported a claim following a fall in the parking lot. A reminder has been created to follow up on your medical appointment scheduled for the day after tomorrow.\\n
        *[the parking lot]: "I stumbled into the supermarket parking lot"
        *[your medical appointment]: "I called my family doctor, I have an appointment for the day after tomorrow."
    """
    next_system_tpl: str = """
        # Objective
        Choose the next action from the company sales team perspective. The response is the action to take and the justification for this action. The response will be a JSON object following a specific schema.

        # Rules
        - Answers in English, even if the customer speaks another language
        - Be concise
        - Take as priority the customer satisfaction
        - Won't make any assumptions
        - Write no more than a few sentences as justification

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format (JSON)
        {format}
    """

    def default_system(self, call: CallStateModel) -> str:
        from helpers.config import CONFIG  # pylint: disable=import-outside-toplevel

        return self._format(
            self.default_system_tpl.format(
                bot_company=call.initiate.bot_company,
                bot_name=call.initiate.bot_name,
                bot_phone_number=CONFIG.communication_services.phone_number,
                date=datetime.now(call.tz()).strftime(
                    "%a %d %b %Y, %H:%M (%Z)"
                ),  # Don't include secs to enhance cache during unit tests. Example: "Mon 15 Jul 2024, 12:43 (CEST)"
                phone_number=call.initiate.phone_number,
            )
        )

    def chat_system(
        self, call: CallStateModel, trainings: list[TrainingModel]
    ) -> list[ChatCompletionSystemMessageParam]:
        from models.message import (  # pylint: disable=import-outside-toplevel
            ActionEnum as MessageActionEnum,
            StyleEnum as MessageStyleEnum,
        )

        return self._messages(
            self._format(
                self.chat_system_tpl,
                actions=", ".join([action.value for action in MessageActionEnum]),
                bot_company=call.initiate.bot_company,
                claim=json.dumps(call.claim),
                default_lang=call.lang.human_name,
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                styles=", ".join([style.value for style in MessageStyleEnum]),
                task=call.initiate.task,
                trainings=trainings,
            ),
            call=call,
        )

    def sms_summary_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.sms_summary_system_tpl,
                bot_company=call.initiate.bot_company,
                bot_name=call.initiate.bot_name,
                claim=json.dumps(call.claim),
                default_lang=call.lang.human_name,
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def synthesis_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.synthesis_system_tpl,
                claim=json.dumps(call.claim),
                format=json.dumps(SynthesisModel.model_json_schema()),
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def citations_system(
        self, call: CallStateModel, text: str
    ) -> list[ChatCompletionSystemMessageParam]:
        """
        Return the formatted prompt. Prompt is used to add citations to the text, without cluttering the content itself.

        The citations system is only used if `text` param is not empty, otherwise `None` is returned.
        """
        return self._messages(
            self._format(
                self.citations_system_tpl,
                claim=json.dumps(call.claim),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                text=text,
            ),
            call=call,
        )

    def next_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.next_system_tpl,
                claim=json.dumps(call.claim),
                format=json.dumps(NextModel.model_json_schema()),
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def _format(
        self,
        prompt_tpl: str,
        trainings: Optional[list[TrainingModel]] = None,
        **kwargs: str,
    ) -> str:
        # Remove possible indentation then render the template
        formatted_prompt = dedent(prompt_tpl.format(**kwargs)).strip()

        # Format trainings, if any
        if trainings:
            # Format documents for Content Safety scan compatibility
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=warning%2Cpython-new#embedding-documents-in-your-prompt
            trainings_str = "\n".join(
                [
                    f"<documents>{escape(training.model_dump_json(exclude=TrainingModel.excluded_fields_for_llm()))}</documents>"
                    for training in trainings
                ]
            )
            formatted_prompt += "\n\n# Internal documentation you can use"
            formatted_prompt += f"\n{trainings_str}"

        # Remove newlines to avoid hallucinations issues with GPT-4 Turbo
        formatted_prompt = " ".join(
            [line.strip() for line in formatted_prompt.splitlines()]
        )

        self.logger.debug("Formatted prompt: %s", formatted_prompt)
        return formatted_prompt

    def _messages(
        self, system: str, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        messages = [
            ChatCompletionSystemMessageParam(
                content=self.default_system(call),
                role="system",
            ),
            ChatCompletionSystemMessageParam(
                content=system,
                role="system",
            ),
        ]
        self.logger.debug("Messages: %s", messages)
        return messages

    @cached_property
    def logger(self) -> Logger:
        from helpers.logging import logger  # pylint: disable=import-outside-toplevel

        return logger


class TtsModel(BaseModel):
    tts_lang: str = "en-US"
    calltransfer_failure_tpl: str = (
        "It seems I can't connect you with an agent at the moment, but the next available agent will call you back as soon as possible."
    )
    connect_agent_tpl: str = (
        "I'm sorry, I wasn't able to respond your request. Please allow me to transfer you to an agent who can assist you further. Please stay on the line and I will get back to you shortly."
    )
    end_call_to_connect_agent_tpl: str = (
        "Of course, stay on the line. I will transfer you to an agent."
    )
    error_tpl: str = (
        "I'm sorry, I have encountered an error. Could you repeat your request?"
    )
    goodbye_tpl: str = (
        "Thank you for calling, I hope I've been able to help. You can call back, I've got it all memorized. {bot_company} wishes you a wonderful day!"
    )
    hello_tpl: str = """
        Hello, I'm {bot_name}, the virtual assistant {bot_company}! Here's how I work: while I'm processing your information, wou will hear a music. Feel free to speak to me in a natural way - I'm designed to understand your requests. During the conversation, you can also send me text messages.
"""
    timeout_silence_tpl: str = (
        "I'm sorry, I didn't hear anything. If you need help, let me know how I can help you."
    )
    welcome_back_tpl: str = "Hello, I'm {bot_name}, from {bot_company}!"
    timeout_loading_tpl: str = (
        "It's taking me longer than expected to reply. Thank you for your patience…"
    )
    ivr_language_tpl: str = "To continue in {label}, press {index}."

    async def calltransfer_failure(self, call: CallStateModel) -> str:
        return await self._translate(self.calltransfer_failure_tpl, call)

    async def connect_agent(self, call: CallStateModel) -> str:
        return await self._translate(self.connect_agent_tpl, call)

    async def end_call_to_connect_agent(self, call: CallStateModel) -> str:
        return await self._translate(self.end_call_to_connect_agent_tpl, call)

    async def error(self, call: CallStateModel) -> str:
        return await self._translate(self.error_tpl, call)

    async def goodbye(self, call: CallStateModel) -> str:
        return await self._translate(
            self.goodbye_tpl,
            call,
            bot_company=call.initiate.bot_company,
        )

    async def hello(self, call: CallStateModel) -> str:
        return await self._translate(
            self.hello_tpl,
            call,
            bot_company=call.initiate.bot_company,
            bot_name=call.initiate.bot_name,
        )

    async def timeout_silence(self, call: CallStateModel) -> str:
        return await self._translate(self.timeout_silence_tpl, call)

    async def welcome_back(self, call: CallStateModel) -> str:
        from helpers.config import CONFIG  # pylint: disable=import-outside-toplevel

        return await self._translate(
            self.welcome_back_tpl,
            call,
            bot_company=call.initiate.bot_company,
            bot_name=call.initiate.bot_name,
            conversation_timeout_hour=CONFIG.conversation.callback_timeout_hour,
        )

    async def timeout_loading(self, call: CallStateModel) -> str:
        return await self._translate(self.timeout_loading_tpl, call)

    async def ivr_language(self, call: CallStateModel) -> str:
        res = ""
        for i, lang in enumerate(call.initiate.lang.availables):
            res += (
                self._return(
                    self.ivr_language_tpl,
                    index=i + 1,
                    label=lang.human_name,
                )
                + " "
            )
        return await self._translate(res.strip(), call)

    def _return(self, prompt_tpl: str, **kwargs) -> str:
        """
        Remove possible indentation in a string.
        """
        return dedent(prompt_tpl.format(**kwargs)).strip()

    async def _translate(self, prompt_tpl: str, call: CallStateModel, **kwargs) -> str:
        """
        Format the prompt and translate it to the TTS language.

        If the translation fails, the initial prompt is returned.
        """
        from helpers.translation import (  # pylint: disable=import-outside-toplevel
            translate_text,
        )

        initial = self._return(prompt_tpl, **kwargs)
        translation = None
        try:
            translation = await translate_text(
                initial, self.tts_lang, call.lang.short_code
            )
        except HttpResponseError as e:
            self.logger.warning("Failed to translate TTS prompt: %s", e)
            pass
        return translation or initial

    @cached_property
    def logger(self) -> Logger:
        from helpers.logging import logger  # pylint: disable=import-outside-toplevel

        return logger


class PromptsModel(BaseModel):
    llm: LlmModel = LlmModel()  # Object is fully defined by default
    sounds: SoundModel = SoundModel()  # Object is fully defined by default
    tts: TtsModel = TtsModel()  # Object is fully defined by default
