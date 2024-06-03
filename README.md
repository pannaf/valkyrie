# V : AI Personal Trainer

TODO system diagram

## Tech Used
- [x] NeMo Curator - build a dataset of exercises 💪
- [x] LangGraph - V as an agent
- [x] LangSmith - prompt evaluation
- [ ] NeMo Guardrails - ensure V doesn't venture into a medical domain space

## [NeMo Curator] Building a Dataset of Exercises
To construct meaningful workouts, V needed to be able to draw from an exercise list. I used [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) to generate said exercise list, pulled from scraping a couple exercise database sites. This made it easy to build a pipeline where I could pull from these sites and auto-filter down to an exercise list through a sequence of `modify_document` calls via `Sequential([Modify(...), Modify(...),])`. Refer to TODO link for the full code.

### A few notes
#### Installation
Installation on a Mac was painful 😅. I did see in the GitHub Issues [here](https://github.com/NVIDIA/NeMo-Curator/issues/76#issuecomment-2135907968) that it's really meant for Linux machines. But.. this didn't stop me from trying to install on my Mac laptop anyway 🙃. After some trial and error, I did land on something that ultimately worked. A key ingredient was using `conda`.I'm not normally a fan of how bloated `conda` can be, but in this case it was recommended in the README of the NeMo text processing repo [here](https://github.com/NVIDIA/NeMo-text-processing) for the `pyini` install that `nemo_text_processing` needs. 

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

