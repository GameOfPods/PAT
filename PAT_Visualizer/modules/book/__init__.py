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

import typing as t
from collections import Counter, defaultdict

from PAT import PATModule
from PAT.utils import create_table, super_format, counter_union
from PAT_Visualizer.modules import VisualizerModule


class BookVisualizer(VisualizerModule):
    _MOST_COMMON_N = 10

    _POV_COUNTER = defaultdict(lambda: 0)
    _WORD_COUNTER = {"words_no_stop": Counter(), "words": Counter()}
    _ENTITY_COUNTER = defaultdict(lambda: Counter())
    _ADDITIONAL_COUNTERS = defaultdict(lambda: 0)

    @classmethod
    def supported_modules(cls) -> t.List[t.Type[PATModule]]:
        from PAT.modules.book import BookModule
        return [BookModule, ]

    @classmethod
    def config_keys(cls) -> t.Dict[str, t.Tuple[t.Callable[[str], t.Any], bool, str, t.Union[None, str, int]]]:
        return {"most_common_n": (int, False, "Amount of most commons to be printed", None)}

    @classmethod
    def load(cls, config: t.Dict[str, t.Any]):
        super().load(config)
        if config.get("most_common_n", None) is not None:
            cls._MOST_COMMON_N = int(config["most_common_n"])

        cls._POV_COUNTER = defaultdict(lambda: 0)
        cls._ENTITY_COUNTER.clear()
        cls._ADDITIONAL_COUNTERS.clear()
        for k in cls._WORD_COUNTER.keys():
            cls._WORD_COUNTER[k] = Counter()

    @classmethod
    def unload(cls):
        cls._POV_COUNTER.clear()
        cls._ENTITY_COUNTER.clear()
        cls._ADDITIONAL_COUNTERS.clear()
        for v in cls._WORD_COUNTER.values():
            v.clear()

    @classmethod
    def _chapter_str(cls, _c) -> str:
        return f"{_c['chapter_title_ext']}{'' if _c['chapter_title_ext'] == _c['chapter_title'] else ' (' + str(_c['chapter_title'] + ')')}"

    @classmethod
    def _print_most_cms(cls, _c: t.Iterable[t.Tuple[t.Any, t.Union[int, float]]]) -> t.List[str]:
        _fmt = lambda x: super_format(x, format_strs={int: "d", float: ".2f"})
        return [f"{_k}@{_fmt(_v)}" for _k, _v in
                sorted(_c, key=lambda x: x[::-1], reverse=True)[:cls._MOST_COMMON_N]]

    def visualize(self) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:
        import xlsxwriter
        from io import BytesIO

        tbl_file = BytesIO()
        workbook = xlsxwriter.Workbook(tbl_file, options={"strings_to_numbers": True})

        data = self.data
        file_path = data["path"]
        infos = data["infos"]
        chapters = infos["chapters"]
        chapter_names = [x["chapter_title"] for x in chapters]
        chapter_name_counter = Counter(chapter_names)
        sentence_count_book = sum(sum(1 if len(k) > 0 else 0 for k in x["words_per_sentence"]) for x in chapters)
        chapter_counters = {
            "words": {c["chapter_title_ext"]: sum(c["words"].values()) for c in chapters},
            "sentences": {c["chapter_title_ext"]: sum(1 if len(k) > 0 else 0 for k in c["words_per_sentence"]) for c in
                          chapters},
            "w/s": {c["chapter_title_ext"]: sum(c["words"].values()) / sum(
                1 if len(k) > 0 else 0 for k in c["words_per_sentence"]) for c in chapters},
        }
        word_counter = {"words": Counter(), "words_no_stop": Counter()}
        word_counter_chapter = defaultdict(lambda: {"words": Counter(), "words_no_stop": Counter()})
        lbl_counter = defaultdict(lambda: Counter())
        lbl_counter_chapter = defaultdict(lambda: defaultdict(lambda: Counter()))
        for chapter in chapters:
            for k, v in word_counter.items():
                v.update(chapter[k])
                word_counter_chapter[self._chapter_str(chapter)][k].update(chapter[k])
            for lbl, ent_counter in chapter["entities"].items():
                lbl_counter[lbl].update(ent_counter)
                lbl_counter_chapter[self._chapter_str(chapter)][lbl].update(ent_counter)
                self._ENTITY_COUNTER[lbl].update(ent_counter)

        self._ADDITIONAL_COUNTERS["Chapters"] += len(chapters)
        self._ADDITIONAL_COUNTERS["Sentences"] += sentence_count_book
        self._ADDITIONAL_COUNTERS["Words"] += word_counter["words"].total()
        self._ADDITIONAL_COUNTERS["Words (No Stop)"] += word_counter["words_no_stop"].total()
        self._WORD_COUNTER["words"].update(word_counter["words"])
        self._WORD_COUNTER["words_no_stop"].update(word_counter["words_no_stop"])

        _ = {
            "Title": {"": infos["book"]},
            "Language": {"": infos["language"]},
            "Chapters": {"": len(chapters)},
            "Distinct Chapter Names": {"": len(chapter_name_counter)},
            "Sentences": {"": sentence_count_book},
            "Unique Words": {"": len(word_counter["words"])},
            "Overall Words": {"": word_counter["words"].total()},
            "Words/Sentence": {"": f'{word_counter["words"].total() / sentence_count_book:.2f}'},
            "Unique Words (No Stop)": {"": len(word_counter["words_no_stop"])},
            "Overall Words (No Stop)": {"": word_counter["words_no_stop"].total()},
            "Words (No Stop)/Sentence": {"": f'{word_counter["words_no_stop"].total() / sentence_count_book:.2f}'},
            "Most verbose Chapter (Words)": {"": ", ".join(self._print_most_cms(chapter_counters["words"].items()))},
            "Most verbose Chapter (Sentences)": {
                "": ", ".join(self._print_most_cms(chapter_counters["sentences"].items()))},
            "Most verbose Chapter (W/S)": {"": ", ".join(self._print_most_cms(chapter_counters["w/s"].items()))},
            "Unique Entity-Types": {"": len(lbl_counter)}
        }
        for ent_type, ent_count in lbl_counter.items():
            _[f"  {ent_type}"] = {"": len(ent_count)}
            _[f"  {ent_type} - Most common ({self._MOST_COMMON_N})"] = {
                "": ", ".join(self._print_most_cms(ent_count.items()))}

        _["Most common Words"] = {"": ", ".join(self._print_most_cms(word_counter["words"].items()))}
        _["Most common Words (No Stop)"] = {"": ", ".join(self._print_most_cms(word_counter["words_no_stop"].items()))}

        for l in create_table(data=_, print_heading=False, worksheet=workbook.add_worksheet("Book")):
            yield l

        yield ""
        yield "Per Chapter Information"
        yield ""

        per_chapter_tbl = {}

        for chapter in sorted(chapters, key=lambda x: x["chapter_idx"]):
            sentence_cnt = sum(1 if len(k) > 0 else 0 for k in chapter["words_per_sentence"])
            chap_key = self._chapter_str(_c=chapter)
            per_chapter_tbl[self._chapter_str(_c=chapter)] = {
                "Name": chapter["chapter_title"],
                "Sentences": sentence_cnt,
                "Unique Words": len(word_counter_chapter[chap_key]["words"]),
                "Overall Words": word_counter_chapter[chap_key]["words"].total(),
                "Words/Sentence": f'{word_counter_chapter[chap_key]["words"].total() / sentence_cnt:.2f}',
                "Unique Words (No Stop)": len(word_counter_chapter[chap_key]["words_no_stop"]),
                "Overall Words (No Stop)": word_counter_chapter[chap_key]["words_no_stop"].total(),
                "Words (No Stop)/Sentence": f'{word_counter_chapter[chap_key]["words_no_stop"].total() / sentence_cnt:.2f}',
            }

        for l in create_table(print_heading=True, data=per_chapter_tbl, worksheet=workbook.add_worksheet("Chapter")):
            yield l

        yield ""

        if len(chapters) > len(chapter_name_counter):
            yield " "
            yield "Entering POV-Mode"
            yield f"POVs: {', '.join(chapter_name_counter.keys())}"
            yield ""

            pov_tbl = {}
            pov_entities = {}

            for pov in chapter_name_counter.keys():
                valid_chapters = [x for x in chapters if x["chapter_title"] == pov]
                sentence_cnt = sum(sum(1 if len(y) > 0 else 0 for y in x["words_per_sentence"]) for x in valid_chapters)
                ent_counter = {
                    k: counter_union(*[Counter(x["entities"].get(k, {})) for x in valid_chapters]) for k in
                    lbl_counter.keys()
                }

                self._POV_COUNTER[pov] += chapter_name_counter[pov]

                pov_tbl[pov] = {
                    "Chapters": len(valid_chapters),
                    "Sentences": sentence_cnt,
                    "Sentences/Chapter": f"{sentence_cnt / len(valid_chapters):.2f}",
                    "Unique Words": sum(
                        len(word_counter_chapter[self._chapter_str(_c=x)]["words"]) for x in valid_chapters),
                    "Overall Words": sum(
                        word_counter_chapter[self._chapter_str(_c=x)]["words"].total() for x in valid_chapters),
                    "Words/Sentences": f'{sum(word_counter_chapter[self._chapter_str(_c=x)]["words"].total() / sentence_cnt for x in valid_chapters):.2f}',
                    "Words/Chapter": f'{sum(word_counter_chapter[self._chapter_str(_c=x)]["words"].total() / len(valid_chapters) for x in valid_chapters):.2f}',
                    "Unique Words (No Stop)": sum(
                        len(word_counter_chapter[self._chapter_str(_c=x)]["words_no_stop"]) for x in valid_chapters),
                    "Overall Words (No Stop)": sum(
                        word_counter_chapter[self._chapter_str(_c=x)]["words_no_stop"].total() for x in valid_chapters),
                    "Words/Sentences (No Stop)": f'{sum(word_counter_chapter[self._chapter_str(_c=x)]["words_no_stop"].total() / sentence_cnt for x in valid_chapters):.2f}',
                    "Words/Chapter (No Stop)": f'{sum(word_counter_chapter[self._chapter_str(_c=x)]["words_no_stop"].total() / len(valid_chapters) for x in valid_chapters):.2f}',
                }

                for lbl, c in ent_counter.items():
                    pov_entities[f"{pov}: {lbl}"] = {
                        str(i): k for i, k in enumerate(self._print_most_cms(c.items()), 1)
                    }

            for l in create_table(print_heading=True, data=pov_tbl, worksheet=workbook.add_worksheet(f"POV")):
                yield l

            for l in create_table(print_heading=True, data=pov_entities, worksheet=workbook.add_worksheet(f"Entities")):
                yield l

        workbook.close()

        return {"tables.xlsx": tbl_file.getbuffer().tobytes()}

    @classmethod
    def visualize_final(cls) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:

        tbl: t.Dict[str, t.Dict[str, t.Any]] = {k: {"": v} for k, v in cls._ADDITIONAL_COUNTERS.items()}
        if len(cls._POV_COUNTER) > 0:
            tbl[f"POV"] = {"": len(cls._POV_COUNTER)}
            tbl[f"POV - Most common ({cls._MOST_COMMON_N})"] = {
                "": ", ".join(cls._print_most_cms(cls._POV_COUNTER.items()))}
        if len(cls._ENTITY_COUNTER) > 0:
            tbl[f"Unique Entity-Types"] = {"": len(cls._ENTITY_COUNTER)}
            for lbl, c in cls._ENTITY_COUNTER.items():
                tbl[f"  {lbl}"] = {"": c.total()}
                tbl[f"  {lbl} - Most common ({cls._MOST_COMMON_N})"] = {"": ", ".join(cls._print_most_cms(c.items()))}

        if len(tbl) > 0:
            import xlsxwriter
            from io import BytesIO

            tbl_file = BytesIO()
            workbook = xlsxwriter.Workbook(tbl_file, options={"strings_to_numbers": True})

            for l in create_table(print_heading=False, data=tbl, worksheet=workbook.add_worksheet("DATA")):
                yield l
            workbook.close()
            return {"tables.xlsx": tbl_file.getbuffer().tobytes()}

        return
