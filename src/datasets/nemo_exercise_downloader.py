"""
Extracts exercise descriptions from web pages, cleans and unifies the text, filters out irrelevant content, and removes duplicates.

Much of this code is adapted from the NVIDIA NeMo Curator tutorial at:
 https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup

from nemo_curator import Sequential, ScoreFilter
from nemo_curator.download.doc_builder import DocumentDownloader, DocumentIterator, DocumentExtractor
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.modify import Modify
from nemo_curator.modifiers import DocumentModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules.add_id import AddId
from nemo_curator.filters import DocumentFilter, WordCountFilter
from nemo_curator.modules import ExactDuplicates

## Downloading a dataste of exercises from the web


class ExerciseDownloader(DocumentDownloader):
    """Exercise downloader that saves downloaded files to a specified directory"""

    def __init__(self, download_dir: str):
        super().__init__()
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)
        self._download_dir = download_dir

    def download(self, url: str) -> str:
        filename = os.path.basename(url) or f"url_{len(os.listdir(self._download_dir))}.html"
        output_file = os.path.join(self._download_dir, filename)
        if os.path.exists(output_file):
            print(f"File '{output_file}' already exists, skipping download.")
            return output_file
        print(f"Downloading dataset from '{url}'...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            file.write(response.content)
        return output_file


class ExerciseIterator(DocumentIterator):
    """Exercise iterator that reads exercises from a file and yields them as examples"""

    def __init__(self):
        super().__init__()
        self._counter = -1

    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            example = []
            for line in file:
                if line.strip() == "":
                    if example:
                        yield self.split_meta(example, file_name)
                        example = []
                else:
                    example.append(line.strip())
            if example:
                yield self.split_meta(example, file_name)

    def split_meta(self, example, file_name):
        """split the example into metadata and content"""
        self._counter += 1
        content = " ".join(example)
        meta = {"filename": file_name, "id": f"{file_name}-{self._counter}"}
        return meta, content


class ExerciseExtractor(DocumentExtractor):
    """Extracts text content from HTML files"""

    def extract(self, content: str):
        soup = BeautifulSoup(content, "html.parser")
        for element in soup(["style", "script", "head", "title", "meta", "[document]"]):
            element.decompose()
        text_content = set()  # Use a set to avoid duplicates
        for tag in soup.find_all(True):
            text = tag.get_text()
            text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
            if text:
                text_content.add(text)
        return {"text": " ".join(text_content)}


def write_jsonl(input_filename: str, output_dir: str, extractor):
    """Adapted from https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/"""
    iterator = ExerciseIterator()
    to_dump = []
    dump_ctr = 0

    def dump_to_file(to_dump, dump_ctr):
        """Helper function to facilitate dumping to file."""
        output_filename = f"{os.path.basename(input_filename)}-{dump_ctr}.jsonl"
        with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as output_file:
            output_file.writelines(to_dump)
        return [], dump_ctr + 1

    for item in iterator.iterate(input_filename):
        record_meta, content = item
        extracted = extractor.extract(content)

        if extracted is None:
            continue

        line = {"text": extracted["text"], **record_meta}
        json_out = json.dumps(line, ensure_ascii=False)
        to_dump.append(json_out + "\n")

        if len(to_dump) == 10000:
            to_dump, dump_ctr = dump_to_file(to_dump, dump_ctr)

    if to_dump:
        dump_to_file(to_dump, dump_ctr)


## Text cleaning and unification


class UnicodeCleaner(DocumentModifier):
    """Unicode cleaner that removes non-ASCII characters from text"""

    def modify_document(self, text):
        cleaned_text = text.encode("ascii", "ignore").decode("ascii")
        return cleaned_text


def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    """Clean and unify the text in the dataset."""
    cleaners = Sequential(
        [
            AddId(id_field="id"),
            Modify(UnicodeReformatter(), text_field="text"),
            Modify(UnicodeCleaner(), text_field="text"),
        ]
    )
    return cleaners(dataset)


## Dataset filtering


class KeywordFilter(DocumentFilter):
    """Filter that removes documents containing specific keywords"""

    def __init__(self):
        super().__init__()
        self.keywords = [
            "menu",
            ":",
            ".",
            "@",
            "view details",
            "?",
            "how to",
            "about",
            "exercises",
            "save",
            "workout planner",
            "previous",
            "next",
            "calculators home",
            "my profile",
            "subscribe",
        ]

    def score_document(self, text: str) -> bool:
        return not any(keyword in text.lower() for keyword in self.keywords)

    def keep_document(self, scores) -> bool:
        return scores


def filter_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """Filter the dataset to remove irrelevant documents."""
    filters = Sequential(
        [
            ScoreFilter(WordCountFilter(min_words=1), text_field="text", score_field="word_count"),
            ScoreFilter(KeywordFilter(), text_field="text"),
        ]
    )
    return filters(dataset)


## Deduplication


def dedupe(dataset: DocumentDataset) -> DocumentDataset:
    """Copied from https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/"""
    deduplicator = ExactDuplicates(id_field="id", text_field="text", hash_method="md5")
    # Find the duplicates
    duplicates = deduplicator(dataset)
    docs_to_remove = duplicates.df.map_partitions(lambda x: x[x._hashes.duplicated(keep="first")])  # pylint: disable=protected-access
    # Remove the duplicates using their IDs.
    duplicate_ids = list(docs_to_remove.compute().id)
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    return DocumentDataset(deduped)


def main():
    """Main entry point for the script"""

    download_dir = "./exercise_download/"
    output_dir = "./exercise_output/"
    urls = [
        "https://www.strengthlog.com/exercise-directory/",
        "https://fitnessprogramer.com/exercises",
    ] + [f"https://fitnessprogramer.com/exercises/page/{i}/" for i in range(2, 82)]

    downloader = ExerciseDownloader(download_dir)
    extractor = ExerciseExtractor()

    # downloader fetches HTML content
    html_file_paths = []
    for url in urls:
        html_file_path = downloader.download(url)
        html_file_paths.append(html_file_path)

        # extract downloaded HTML data and write to JSONL files
        write_jsonl(html_file_path, output_dir, extractor)

    jsonl_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jsonl")]
    dataset = DocumentDataset.read_json(jsonl_files, add_filename=True)

    curation_steps = Sequential(
        [
            clean_and_unify,
            filter_dataset,
            dedupe,
        ]
    )

    dataset = curation_steps(dataset)

    print("Executing the pipeline........")
    dataset = dataset.persist()

    dataset.to_json(output_dir, write_to_filename=True)

    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
