#  PAT - Toolkit to analyze podcast audio and topics talked about in the podcast. For example Books
#  Copyright (c) 2024.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
import os
from enum import Enum, auto as enum_auto
from logging import getLogger
from typing import Optional, Tuple

_SYSTEM_MESSAGE = ("You are a helpful assistant that helps to create precise and complete summaries. "
                   "Always keep your answers precise and only use what you either find in a prompt or in the source "
                   "material that is to be summarized. You may use contextual information you know on the source "
                   "material.\n"
                   "Only return the requested summary without any introduction of yourself or other system messages.\n"
                   "Dont include personal opinions on the content.\n"
                   "Always start the summary with a heading, an introduction of the text and it being a summary.\n"
                   "If the prompt contains an already existing summary of the text extend the existing introduction "
                   "and extend it with the new information. Then return the combined summary of before and the new "
                   "information you can add to it.\n"
                   "Always return the summary in valid Markdown format.")


class Templates(Enum):
    Basic = enum_auto()
    Podcast = enum_auto()
    LANGCHAIN_DEFAULT = enum_auto()

    def get_template(self) -> Tuple[str, str]:
        if self == Templates.Basic:
            return (
                "Write a complete in depth summary of the following text\n{text}",
                "Your task is to create an in depth summary. "
                "There already is a summary you created of the previous part of the text up to this point:\n\n"
                "{existing_answer}"
                "\n\nPlease extend this summary with the following information\n"
                "{text}"
            )
        if self == Templates.Podcast:
            return (
                "Write a complete in depth summary of the core points of the following transcript of a podcast "
                "as a bullet point list\n"
                "{text}",
                "Your task is to create a complete and in depth summary of the core points of a transcribed podcast "
                "as a bullet point list.\n"
                "There already is a summary you created of the previous part of the podcast up to this point:\n\n"
                "{existing_answer}"
                "\n\nPlease extend this summary with the following information\n"
                "{text}"
            )
        if self == Templates.LANGCHAIN_DEFAULT:
            from langchain.chains.summarize.refine_prompts import PROMPT, REFINE_PROMPT
            return PROMPT.template, REFINE_PROMPT.template
        raise ValueError(f"Template for {self} not defined")


class LLM(Enum):
    OpenAI = enum_auto()
    Groq = enum_auto()

    def get_llm(self, model: str):

        if self == self.OpenAI:
            from langchain_openai import ChatOpenAI as OpenAI
            if "OPENAI_API_BASE" in os.environ:
                getLogger("Summarization").info(f"Using openai api located at {os.environ['OPENAI_API_BASE']}")
            return OpenAI(model_name=model, max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", 4096)))
        if self == self.Groq:
            from langchain_groq import ChatGroq as Groq
            return Groq(model_name=model, streaming=True, max_tokens=int(os.environ.get("GROQ_MAX_TOKENS", 4096)), )

        raise ValueError(f"LLM of type {self} not defined")


def summarize(
        text: str, model: str = "gpt-4o", lang: str = None, template: Templates = Templates.Basic, llm: LLM = LLM.OpenAI
) -> Optional[str]:
    logger = getLogger("Summarization")

    try:
        from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
        from langchain.prompts import PromptTemplate
        from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains.mapreduce import MapReduceChain
        from langchain.docstore.document import Document
        from langchain.chains.summarize import load_summarize_chain

        chunk_size = int(os.environ.get("SUMMARIZE_CHUNK_SIZE", 15000))

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")

            def _len_fun(_txt: str) -> int:
                return len(enc.encode(_txt, ))
        except ImportError:
            logger.error("Could not import tiktoken. Will use python length for text length estimation")

            def _len_fun(_txt: str) -> int:
                return len(_txt)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n",
                        "\n",
                        ".",
                        ",",
                        " ",
                        "\u200b",
                        "\uff0c",
                        "\u3001",
                        "\uff0e",
                        "\u3002",
                        "",
                        ],
            chunk_size=chunk_size,
            length_function=_len_fun,
        )

        llm = llm.get_llm(model=model)
        logger.info(f"Loaded {llm.name} as summarization LLM with model {model}")
        doc = Document(text)
        split_doc = splitter.split_documents([doc])

        logger.info(
            f"Split text into {len(split_doc)} documents. "
            f"Original text length: {_len_fun(doc.page_content)}. "
            f"Chunk size: {chunk_size}"
        )

        prompt_template, refine_template = template.get_template()
        if lang is not None:
            if os.environ.get("TRANSLATE_PROMPTS", "False").lower() in ("true", "1", "t", "y", "yes"):
                logger.info(f"Language {lang} provided. Will try to translate prompts to preserve language in result")
                try:
                    from deep_translator import GoogleTranslator
                    translator = GoogleTranslator(target=lang)
                    prompt_template = translator.translate(prompt_template)
                    refine_template = translator.translate(refine_template)
                    del GoogleTranslator
                except ImportError:
                    translate_prompt = f"Please translate to following prompt to %s. Preserve text in brackets: %s"
                    prompt_template = llm.invoke(translate_prompt % (lang, prompt_template)).content
                    refine_template = llm.invoke(translate_prompt % (lang, refine_template)).content
                logger.info(f"Translated summarization prompts to {lang}")
            else:
                from iso639 import languages
                language = None
                for k in ("alpha2", "bibliographic", "terminology"):
                    try:
                        language = languages.get(**{k: lang}).name
                        break
                    except KeyError:
                        pass
                if language is None:
                    language = lang
                prompt_template = f"{prompt_template}\nProvide your answers in {language} always"
                refine_template = f"{refine_template}\nProvide your answers in {language} always"
                logger.info(f"Extended prompts to return answers in {language} (from {lang})")
            logger.debug(f"Translated prompt template: {prompt_template}")
            logger.debug(f"Translated refine template: {refine_template}")

        question = ConditionalPromptSelector(
            default_prompt=PromptTemplate.from_template(f"{_SYSTEM_MESSAGE}\n\n{prompt_template}"),
            conditionals=[(is_chat_model, ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(_SYSTEM_MESSAGE),
                    HumanMessagePromptTemplate.from_template(prompt_template),
                ]
            ))]
        )

        refine = ConditionalPromptSelector(
            default_prompt=PromptTemplate.from_template(f"{_SYSTEM_MESSAGE}\n\n{refine_template}"),
            conditionals=[(is_chat_model, ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(_SYSTEM_MESSAGE),
                    HumanMessagePromptTemplate.from_template(refine_template),
                ]
            ))]
        )

        # TODO: Maybe use conditional to utilize chat models and create a chat conversation rather than a chain

        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=PromptTemplate.from_template(f"{_SYSTEM_MESSAGE}\n\n{prompt_template}"),
            refine_prompt=PromptTemplate.from_template(f"{_SYSTEM_MESSAGE}\n\n{refine_template}"),
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

        summary = chain.invoke({"input_documents": split_doc}, config={"max_concurrency": 1})
        logger.info("Created summary of text")

        return summary["output_text"]
    except ImportError as e:
        logger.exception(
            f"Could not perform summarization. Please check requirements for summarization. Especially for "
            f"selected {llm.name}-LLM",
            exc_info=e)
        return None
