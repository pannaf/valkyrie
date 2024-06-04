# V : AI Personal Trainer

Meet V! Your new virtual personal trainer! 🙃

This repo has the code for my entry for the [Generative AI Agents Developer Contest by NVIDIA and LangChain](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/).

TODO system diagram

## Tech Used
- [x] <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> NeMo Curator - build a dataset of exercises that V can draw from when planning workouts 💪
- [x] <img src="https://github.com/pannaf/artemis/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="35"/> LangGraph - V as an agent
- [x] <img src="https://github.com/pannaf/artemis/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="35"/> LangSmith - prompt evaluation
- [ ] <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> NeMo Guardrails - ensure V doesn't venture into a medical domain space

## <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="25"/> [NeMo Curator] Building an Exercise Dataset
To construct meaningful workouts, V needed to draw from a solid exercise list with a diverse set of movements. While this list could have been generated by prompting an LLM, doing so runs the risk of hallucination and lack of comprehensiveness. On the other hand, scraping credible fitness websites ensures accurate, relevant, and consistent information from domain experts.

### <img src="https://github.com/pannaf/artemis/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="20"/> [NeMo Curator] Generating an Exercise List
I used NeMo Curator to generate an exercise list through a pipeline that gathers, cleans, and processes web-scraped data.

#### Pipeline Overview

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

#### Code
Refer to [src/datasets/nemo_exercise_downloader.py](src/datasets/nemo_exercise_downloader.py) for the full implementation.

#### A few notes
##### Installation
Installation on my Mac laptop was painful 😅. I did see in the GitHub Issues [here](https://github.com/NVIDIA/NeMo-Curator/issues/76#issuecomment-2135907968) that it's really meant for Linux machines. But.. this didn't stop me from trying to install on my Mac anyway 🙃. After some trial and error, I landed on something that ultimately worked. `conda` with a Python 3.10.X version was clutch. Later versions of Python (3.11 & 3.12) didn't work for me. I'm not normally a fan of the `conda` bloat, but in this case the README of the NeMo text processing repo [here](https://github.com/NVIDIA/NeMo-text-processing) recommended it for the `pyini` install that `nemo_text_processing` needs. 

In case others find this helpful, I've got the following in my `~/.zsh_history` as the steps just prior to getting things working:
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
I didn't get a chance to take full advantage of what I think is a key ingredient in the magic sauce for NeMo Curator. Namely the GPU acceleration that makes it possible to more efficiently process massive amounts of data. My exercise dataset doesn't really fall into that "massive amounts" category, so it wasn't super necessary to use in my case. But.. I think it would have been super fun to explore their multi-node, multi-GPU classifier inference for distributed data classification (example [here](https://github.com/NVIDIA/NeMo-Curator/blob/main/docs/user-guide/DistributedDataClassification.rst)). Next time!

##### Experiments that didn't make the cut
I explored a couple aspects of NeMo Curator that I ultimately didn't get to use in my final system:
- Wikipedia data pull - one thing I found here is that I needed to set `dump_date=None` in `download_wikipedia()` in order to get [this example](https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/download_wikipedia.py) working
- Common data crawler - no insights to report.. [this example](https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/download_common_crawl.py) worked pretty well for me, just requires you to specify a reasonable directory

### Cleaning the `.jsonl` format a little
Normally I wouldn't include this type of detail, but I thought this was pretty neat and worth a quick share! My `.jsonl` files output from my NeMo Curator pipeline end up with this type of format:

```json
{"filename":"exercises-0.jsonl","id":"doc_id-04900","text":"Seated Barbell Shoulder Press","word_count":4}
{"filename":"exercises-0.jsonl","id":"doc_id-06200","text":"Dumbbell Goblet Squat","word_count":3}
{"filename":"exercises-0.jsonl","id":"doc_id-07500","text":"Curtsy Lunge","word_count":2}
```

I wanted to transform this into a simple text file that has just the text fields extracted. There's lotsa ways I could have done this in Python, but there's also this pretty sweet command line JSON processor `jq` for these types of things. Head over [here](https://formulae.brew.sh/formula/jq) if you haven't heard of `jq` before. Here's the command:

```zsh
for file in exercises-*.jsonl; do
    jq -r '.text' "$file" > "${file%.jsonl}-text.jsonl" # pulls out the "text" field from each JSON line
done
```

Giving the output:

```text
Seated Barbell Shoulder Press
Dumbbell Goblet Squat
Curtsy Lunge
```

Exactly what I wanted 🙃

### Annotating Exercises with Attributes
To get the final list of exercises that V could use in workout planning, I took a few additional Human-in-the-loop steps:

1. Collaborated with ChatGPT to fill out an initial pass at annotating each of the ~1400 exercises with the following attributes: `"Exercise Name","Exercise Type","Target Muscle Groups","Movement Pattern","Exercise Difficulty/Intensity","Equipment Required","Exercise Form and Technique","Duration and Repetitions","Exercise Modifications and Variations","Safety Considerations","Primary Goals","Exercise Dynamics","Rest Periods","Exercise Sequence"`
2. Collaborated with a NASM-certified exercise domain expert to correct any mistakes from ChatGPT's initial attributes.
   
The attributes are needed to provide the sufficient context needed for V to select appropriate exercises for a user, based on their fitness level and preferences.
