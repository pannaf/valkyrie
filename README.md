# V : AI Personal Trainer

TODO system diagram

## Tech Used
- [x] <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> NeMo Curator - build a dataset of exercises 💪
- [x] LangGraph - V as an agent
- [x] LangSmith - prompt evaluation
- [ ] <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> NeMo Guardrails - ensure V doesn't venture into a medical domain space

## <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="25"/> [NeMo Curator] Building an Exercise Dataset
To construct meaningful workouts, V needed to be able to draw from a solid exercise list.

### <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="20"/> [NeMo Curator] Generating an Exercise List
I used NeMo Curator to generate an exercise list through a data curation pipeline that gathers, cleans, and processes data scraped from various web sources.

#### Pipeline Overview
> Refer to [src/datasets/nemo_exercise_downloader.py](src/datasets/nemo_exercise_downloader.py) for the full code implementation

Following the NeMo Curator tutorial [here](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/), my pipeline includes the following steps:

1. Download and Extract Data:
- `ExerciseDownloader` (custom): Downloads HTML from specified URLs that contain exercise lists.
- `ExerciseIterator` (custom): Splits the HTML content into individual records with metadata.
- `ExerciseExtractor` (custom): Extracts and cleans text content from HTML, removing unnecessary tags.

2. Clean and Unify Text:
- `UnicodeCleaner` (custom): Removes non-ASCII characters.
- `UnicodeReformatter` (direct NeMo): Standardizes Unicode formatting.

3. Filter the Dataset:
- `KeywordFilter` (custom): Removes irrelevant documents based on specific keywords.
- `WordCountFilter` (direct NeMo): Ensures documents meet a minimum word count, at least 1 word.

4. Deduplicate Records:
- `ExactDuplicates` (direct NeMo): Identifies and removes exact duplicate records.

5. Output the Results:
- Saves the final dataset in JSONL format.

#### A few notes
##### Installation
Installation on a Mac was painful 😅. I did see in the GitHub Issues [here](https://github.com/NVIDIA/NeMo-Curator/issues/76#issuecomment-2135907968) that it's really meant for Linux machines. But.. this didn't stop me from trying to install on my Mac laptop anyway 🙃. After some trial and error, I did land on something that ultimately worked. A key ingredient was using `conda` with a Python 3.10.X version. Later versions of Python, such as 3.11 and 3.12, didn't work for me. I'm not normally a fan of how bloated `conda` can be, but in this case it was recommended in the README of the NeMo text processing repo [here](https://github.com/NVIDIA/NeMo-text-processing) for the `pyini` install that `nemo_text_processing` needs. 

In case others find this helpful, my `~/.zsh_history` shows these steps just prior to getting things working:
```zsh
➜ conda install -c conda-forge pynini=2.1.5
➜ pip install nemo_text_processing
➜ pip install 'nemo-toolkit[all]'
➜ cd NeMo-Curator # note I cloned the NeMo-Curator repo for this install
NeMo-Curator git:(main) ➜ pip install .
NeMo-Curator git:(main) ➜ brew install opencc
NeMo-Curator git:(main) ➜ export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/opencc/1.1.7/lib:$DYLD_LIBRARY_PATH
```
##### NeMo Magic Sauce
I didn't get a chance to take full advantage of what I think probably makes the magic sauce for NeMo Curator. Namely... Next time!
##### Experiments that didn't make the cut
I experimented with some other aspects of NeMo Curator that I ultimately didn't use in my final system:
- Wikipedia pull
- Common data crawler

