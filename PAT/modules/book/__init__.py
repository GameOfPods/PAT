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


import json
import logging
import os
import re
from collections import defaultdict, Counter
from typing import Tuple, Dict, Any, List, Union, Set, Optional

from PAT.modules import PATModule
from PAT.utils.summerize import LLM as LLMService


class BookModule(PATModule):
    _LOGGER = logging.getLogger("BookModule")
    _SPECIAL_CHAPTERS = {"prologue", "introduction", "epilogue", "prolog", "epilog"}
    _BOOK_VALID_CHAPTERS: Dict[str, Set[str]] = defaultdict(lambda: set())
    _CHAPTER_NUMBER_REGEX = {re.compile(r"chapter \d+$"), re.compile(r"kapitel \d+$")}

    _SPACY_MODELS = {"en": "en_core_web_trf", "de": "de_core_news_lg"}

    _SUMMARIZE_MODEL: Optional[str] = None
    _SUMMARIZE_SERVICE: Optional[LLMService] = None

    def _chapter_valid(self, chapter_name: str, chapter_counter: Dict[str, int]) -> bool:
        if len(self._BOOK_VALID_CHAPTERS[self._book.title]) > 0:
            return chapter_name in self._BOOK_VALID_CHAPTERS[self._book.title]
        if chapter_name.isnumeric():
            return True
        if chapter_counter.get(chapter_name, 0) > 0 or chapter_counter.get(chapter_name.lower(), 0) > 0:
            return True
        if chapter_name in self._SPECIAL_CHAPTERS or chapter_name.lower() in self._SPECIAL_CHAPTERS:
            return True
        if any(x.fullmatch(chapter_name.lower()) for x in self._CHAPTER_NUMBER_REGEX):
            return True
        return False

    @classmethod
    def config_keys(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "chapter_infos": {
                "type": str, "required": False, "default": None,
                "help": "json file that contains chapter information. Format should be {'book title': [list, of, chapters]}"
            },
            "book_summarize_model": {
                "type": str, "required": False, "default": None,
                "help": f"OpenAI model to use to summarize book content. "
                        f"If not provided no summarization will be performed. (Default: %(default)s)"
            },
            "book_summarize_service": {
                "type": str, "choices": [x.name for x in LLMService], "required": False,
                "default": LLMService.OpenAI.name,
                "help": f"Service to use for summarization tasks. Choices: %(choices)s. (Default: %(default)s)"
            }
        }

    @classmethod
    def load(cls, config: Dict[str, Any]):
        super().load(config)
        if "chapter_infos" in config and config["chapter_infos"] is not None:
            with open(config["chapter_infos"]) as f_in:
                chapter_info = json.load(f_in)
            for k, v in chapter_info.items():
                cls._BOOK_VALID_CHAPTERS[k] = set(v)
                cls._LOGGER.info(f"Set valid chapters for '{k}' to {cls._BOOK_VALID_CHAPTERS[k]} "
                                 f"from '{config['chapter_infos']}'")
        cls._SUMMARIZE_MODEL = config.get("book_summarize_model", None)
        if "book_summarize_service" in config:
            try:
                cls._SUMMARIZE_SERVICE = [x for x in LLMService if x.name == config["book_summarize_service"]][0]
            except IndexError:
                cls._LOGGER.error(
                    f"Could not find Summarization LLM service with name {config['book_summarize_service']}"
                )

    def __init__(self, file: str):
        super().__init__(file)

        from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION
        from langdetect import detect
        from bs4 import BeautifulSoup
        from spacy_download import load_spacy

        book = epub.read_epub(self.file, options={"ignore_ncx": True})
        self._book = book
        self._LOGGER.info(f"Read book \"{book.title}\"")
        nav = list(book.get_items_of_type(ITEM_NAVIGATION))[0].get_content().decode()
        items = sorted((x for x in book.get_items_of_type(ITEM_DOCUMENT) if x.get_name() in nav),
                       key=lambda x: nav.index(x.get_name()))
        self._chapters = []
        for item in items:
            content = item.get_body_content()
            soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
            story = [x.get_text() for x in soup.find_all("p")]
            headings = [x.get_text() for x in soup.find_all(re.compile(r"^h[1-6]$"))]
            if len(headings) != 1 or len(headings[0]) <= 0:
                continue
            self._chapters.append((headings[0], story))

        heading_c = Counter(x[0] for x in self._chapters)
        self._valid_chapters = set(
            k for k, v in heading_c.items() if v > 1 or k.isnumeric() or k.lower() in self._SPECIAL_CHAPTERS)
        self._invalid_chapters = set(heading_c.keys()) - self._valid_chapters
        self._LOGGER.info(f"Valid chapters: {', '.join(self._valid_chapters)}; "
                          f"Invalid chapters: {', '.join(self._invalid_chapters)}")
        self._chapters = [(x, y) for x, y in self._chapters if x in self._valid_chapters]

        full_text = "\n".join("\n".join(x) for _, x in self._chapters)
        self._lang = detect(full_text)
        self._LOGGER.info(f"Found language {self._lang}")

        spacy_module_kwargs = {}
        self._nlp = load_spacy(self._SPACY_MODELS.get(self._lang, self._SPACY_MODELS["en"]), **spacy_module_kwargs)

    @classmethod
    def supports_file(cls, file: str) -> bool:
        try:
            from ebooklib import epub
            import bs4
            import spacy
            from langdetect import detect
            from spacy_download import load_spacy
            try:
                epub.read_epub(file, options={"ignore_ncx": True})
                return True
            except epub.EpubException:
                raise ValueError()
        except (ImportError, ValueError):
            return False

    @classmethod
    def description(cls) -> str:
        return f"Module that analyzes book data from provided epub files"

    def process(self) -> Union[Tuple[Dict[str, Any], List[Tuple[str, bytes]]], Dict[str, Any]]:

        try:
            from roman import toRoman
        except ImportError:
            def toRoman(n):
                return n

        from tqdm import tqdm

        _ = self.get_ner_pipeline()

        chapter_infos = []

        _name_counter = defaultdict(lambda: 0)
        _over_all_counter = Counter(x for x, *_ in self._chapters)

        with tqdm(enumerate(self._chapters), total=len(self._chapters), desc="Processing chapters", leave=False,
                  unit="c") as pb:
            for i, (chapter_title, chapter_content) in pb:
                pb.set_description(f"Processing {chapter_title}")
                _name_counter[chapter_title] += 1
                _title_extension = toRoman(_name_counter[chapter_title]) if _over_all_counter[chapter_title] > 1 else ''
                chapter_info = {
                    "chapter_idx": i,
                    "chapter_title": chapter_title,
                    "chapter_title_ext": f"{chapter_title}{f' {_title_extension}' if len(_title_extension) else ''}",
                    "chapter_content": chapter_content
                }
                pb.set_description(f"Processing {chapter_info['chapter_title_ext']}")
                text = "\n".join(chapter_content).strip()
                doc = self._nlp(text)

                chapter_info["words_per_sentence"] = [
                    Counter(x.text for x in sent if not any([x.is_space, x.is_punct])) for sent in doc.sents
                ]
                chapter_info["words"] = Counter([x.text for x in doc if not any([x.is_space, x.is_punct])])

                chapter_info["words_per_sentence_no_stop"] = [
                    Counter(x.text for x in sent if not any([x.is_space, x.is_punct, x.is_stop])) for sent in doc.sents
                ]
                chapter_info["words_no_stop"] = Counter(
                    [x.text for x in doc if not any([x.is_space, x.is_punct, x.is_stop])])
                chapter_info["entities"] = defaultdict(Counter)

                if self._SUMMARIZE_SERVICE is not None and self._SUMMARIZE_MODEL is not None:
                    try:
                        from PAT.utils.summerize import summarize, Templates
                        chapter_info["summary"] = summarize(
                            text=text,
                            model=self._SUMMARIZE_MODEL,
                            lang=self._lang,
                            template=Templates.LANGCHAIN_DEFAULT,
                            llm=self._SUMMARIZE_SERVICE,
                        )
                    except Exception as ignore:
                        pass

                for ent in doc.ents:
                    lbl = ent.label_.lower()
                    lbl = {"per": "person"}.get(lbl, lbl)
                    ent_identifier = " ".join(
                        x.capitalize() for x in ent.text.split(" ")) if lbl == "person" else ent.lemma_
                    chapter_info["entities"][lbl][ent_identifier] += 1

                chapter_infos.append(chapter_info)

        ret = {
            "book": self._book.title,
            "language": self._lang,
            "chapters": chapter_infos
        }

        if self._SUMMARIZE_SERVICE is not None and self._SUMMARIZE_MODEL is not None:
            try:
                from PAT.utils.summerize import summarize, Templates
                self._LOGGER.info("Creating summary of whole book")
                text_f = "\n\n".join(
                    f"{x['chapter_title']}\n\n{os.linesep.join(x['chapter_content'])}" for x in chapter_infos
                )
                ret["summary"] = summarize(
                    text=text_f,
                    model=self._SUMMARIZE_MODEL,
                    lang=self._lang,
                    template=Templates.LANGCHAIN_DEFAULT,
                    llm=self._SUMMARIZE_SERVICE,
                )
            except Exception as ignore:
                pass

        return ret

    def get_ner_pipeline(self):
        if "ner" in self._nlp.pipe_names:
            return self._nlp.get_pipe("ner")
        config = {}
        r = self._nlp.add_pipe("ner", name="ner", config=config)
        return r
