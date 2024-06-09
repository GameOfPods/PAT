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


class Templates(Enum):
    Basic = enum_auto()
    Podcast = enum_auto()
    LANGCHAIN_DEFAULT = enum_auto()

    def get_template(self) -> Tuple[str, str]:
        if self == Templates.Basic:
            return (
                "Write a complete in depth summary of the following text\n{text}",
                "Your task is to create an in depth summary. "
                "There already is a summary you created of the previous part of the text up to this point:\n"
                "{existing_answer}"
                "\nPlease append this summary with the following information\n"
                "{text}"
            )
        if self == Templates.Podcast:
            return (
                "Write a complete in depth summary of the core points of the following transcript of a podcast\n{text}",
                "Your task is to create an in depth summary of the core points of a transcribed podcast. "
                "There already is a summary you created of the previous part of the podcast up to this point:\n"
                "{existing_answer}"
                "\nPlease append this summary with the following information\n"
                "{text}"
            )
        if self == Templates.LANGCHAIN_DEFAULT:
            from langchain.chains.summarize.refine_prompts import PROMPT, REFINE_PROMPT
            return PROMPT.template, REFINE_PROMPT.template
        raise ValueError(f"Template for {self} not defined")


def summarize(
        text: str, openai_model: str = "gpt-4o", language: str = None, template: Templates = Templates.Basic
) -> Optional[str]:
    logger = getLogger("Summarization")

    try:
        from langchain_openai import ChatOpenAI as OpenAI
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.chains.mapreduce import MapReduceChain
        from langchain.prompts import PromptTemplate
        from langchain.docstore.document import Document
        from langchain.chains.summarize import load_summarize_chain

        llm = OpenAI(model_name=openai_model)
        if "OPENAI_API_BASE" in os.environ:
            logger.info(f"Using openai api located at {os.environ['OPENAI_API_BASE']}")
        logger.info(f"Loaded openai model {openai_model}")
        doc = Document(text)
        prompt_template, refine_template = template.get_template()
        if language is not None:
            logger.info(f"Language {language} provided. Will try to translate prompts to preserve language in result")
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(target=language, model=openai_model)
                prompt_template = translator.translate(prompt_template)
                refine_template = translator.translate(refine_template)
                del GoogleTranslator
            except ImportError:
                translate_prompt = f"Please translate to following prompt into %s. Preserve every text in brackets: %s"
                prompt_template = llm.invoke(translate_prompt % (language, prompt_template)).content
                refine_template = llm.invoke(translate_prompt % (language, refine_template)).content
            logger.debug(f"Translated prompt template: {prompt_template}")
            logger.debug(f"Translated refine template: {refine_template}")
            logger.info(f"Translated summarization prompts to {language}")

        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=PromptTemplate.from_template(prompt_template),
            refine_prompt=PromptTemplate.from_template(refine_template),
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

        summary = chain.invoke({"input_documents": [doc]}, config={"max_concurrency": 1})
        logger.info("Created summary of text")

        return summary["output_text"]
    except ImportError as e:
        logger.exception(f"Could not perform summarization. Please check requirements for summarization", exc_info=e)
        return None
