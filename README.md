# V : AI Personal Trainer

TODO system diagram

## Tech Used
- [x] NeMo Curator - build a dataset of exercises 💪
- [x] LangGraph - V as an agent
- [x] LangSmith - prompt evaluation
- [ ] NeMo Guardrails - ensure V doesn't venture into a medical domain space

## [NeMo Curator] Building a Dataset of Exercises
To construct meaningful workouts, V needed to be able to draw from an exercise list. To generate this list, I used [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator). Refer to TODO link for the full code.

Following the NeMo Curator tutorial [here](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/), I built a pipeline that includes the following high-level steps:

1. Define a custom document builder to download the dataset from the web and convert to JSONL format.
2. Define custom modifiers to clean and unify the text data.
3. Filter the dataset using predefined and custom heuristics.
4. Deduplicate the dataset to remove identical records.
5. Output the results to JSONL format.

### A few notes
#### Installation
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
#### NeMo Magic Sauce
I didn't get a chance to take full advantage of what I think probably makes the magic sauce for NeMo Curator. Namely... Next time!
#### Experiments that didn't make the cut
I experimented with some other aspects of NeMo Curator that I ultimately didn't use in my final system:
- Wikipedia pull
- Common data crawler

