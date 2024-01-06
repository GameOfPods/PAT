#  PAT - Toolkit to analyze podcast audio and topics talked about in the podcast. For example Books
#  Copyright (c) 2024.  RedRem95
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

import logging
import re
from collections import defaultdict, Counter

from .. import Module


class BookModule(Module):

    _LOGGER = logging.getLogger("BookModule")
    _SPECIAL_CHAPTERS = {"prologue", "introduction", "epilogue", "prolog", "epilog"}

    _SPACY_MODELS = {"en": "en_core_web_trf", "de": "de_core_news_lg"}

    def __init__(self, file: str):
        super().__init__(file)

        from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION
        from langdetect import detect
        from bs4 import BeautifulSoup
        from spacy_download import load_spacy

        book = epub.read_epub(self.file, options={"ignore_ncx": True})
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
        self._valid_chapters = set(k for k, v in heading_c.items() if v > 1 or k.isnumeric() or k.lower() in self._SPECIAL_CHAPTERS)
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
    def description(cls):
        return f"Module that analyzes book data from provided epub files"

    def process(self):

        try:
            from roman import toRoman
        except ImportError:
            def toRoman(n):
                return n

        from tqdm import tqdm

        _ = self.get_ner_pipeline()

        infos = []

        _name_counter = defaultdict(lambda: 0)
        _over_all_counter = Counter(x for x, *_ in self._chapters)

        with tqdm(self._chapters, desc="Processing chapters", leave=False, unit="c") as pb:
            for chapter_title, chapter_content in pb:
                pb.set_description(f"Processing {chapter_title}")
                _name_counter[chapter_title] += 1
                _title_extension = toRoman(_name_counter[chapter_title]) if _over_all_counter[chapter_title] > 1 else ''
                chapter_info = {
                    "chapter_title": chapter_title,
                    "chapter_title_ext": f"{chapter_title}{f' {_title_extension}' if len(_title_extension) else ''}",
                    "chapter_content": chapter_content
                }
                text = "\n".join(chapter_content).strip()
                doc = self._nlp(text)

                chapter_info["sentence_count"] = len(list(doc.sents))
                chapter_info["words"] = [token.text for token in doc if not token.is_stop and not token.is_punct]
                chapter_info["word_count"] = Counter(chapter_info["words"])
                chapter_info["entities"] = defaultdict(Counter)

                for ent in doc.ents:
                    lbl = ent.label_.lower()
                    lbl = {"per": "person"}.get(lbl, lbl)
                    ent_identifier = " ".join(x.capitalize() for x in ent.text.split(" ")) if lbl == "person" else ent.lemma_
                    chapter_info["entities"][lbl][ent_identifier] += 1

                infos.append(chapter_info)

        print("Chapter hard facts:")
        for info in infos:
            print(f"  {info['chapter_title_ext']} -> Sentences: {info['sentence_count']}; Words: {len(info['words'])}; Most common words: {info['word_count'].most_common(5)}")

        print("Most words:         ", sorted(((x['chapter_title_ext'], len(x['words'])) for x in infos), key=lambda x: -x[1])[:5])
        print("Most sentences:     ", sorted(((x['chapter_title_ext'], x['sentence_count']) for x in infos), key=lambda x: -x[1])[:5])
        print("Most words/sentence:", sorted(((x['chapter_title_ext'], len(x['words'])/x['sentence_count']) for x in infos), key=lambda x: -x[1])[:5])

        pov_words, pov_sentences, pov_words_per_sentence = Counter(), Counter(), defaultdict(list)
        for info in infos:
            pov_words[info['chapter_title']] += len(info['words'])
            pov_sentences[info['chapter_title']] += info['sentence_count']
            pov_words_per_sentence[info['chapter_title']].append(len(info['words'])/info['sentence_count'])

        print("POV verbosity:")
        for pov, pov_count in sorted(_name_counter.items(), key=lambda x: x[::-1], reverse=True):
            print(f"  {pov:{max(len(x) for x in _name_counter.keys())}}: {pov_count} chapters; Words: {pov_words[pov]/pov_count:.2f}/{pov_words[pov]}; Sentences: {pov_sentences[pov]/pov_count:.2f}/{pov_sentences[pov]}; Words per sentence: {sum(pov_words_per_sentence[pov])/len(pov_words_per_sentence[pov]):.2f}")

        person_collection = Counter()
        chapter_person_collection = defaultdict(Counter)
        for info in infos:
            person_collection.update(info["entities"]["person"])
            chapter_person_collection[info["chapter_title"]].update(info["entities"]["person"])

        print("Most common persons:", person_collection.most_common(10))

        print("Most common persons per POV:")
        for pov, counter in chapter_person_collection.items():
            print(f"  {pov}: {counter.most_common(10)}")

    def get_ner_pipeline(self):
        if "ner" in self._nlp.pipe_names:
            return self._nlp.get_pipe("ner")
        config = {}
        r = self._nlp.add_pipe("ner", name="ner", config=config)
        return r

