# V : AI Personal Trainer

## Tech Used
[x] NeMo Curator
[x] LangGraph
[x] LangSmith

## Building a Dataset of Exercises
To construct meaningful workouts, V needed to be able to draw from an exercise list. I used [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) to generate an exercise list, pulled from scraping a couple exercise database sites. This made it easy to build a pipeline where I could pull and auto-filter the exercise list through a sequence of `modify_document` calls via `Sequential([Modify(...), Modify(...),])`. Refer to TODO link for the full code.

### A couple notes
#### Installation
Installation on a Mac was painful. I did see in the GitHub Issues that it's only meant for Linux machines. But.. this didn't stop me from trying to install on my Mac laptop anyway 🙃. After some trial and error, I did land on something that ultimately worked. I'm not even convinced I could retrace the steps for what ultimately worked. In case others find this helpful, looking through my `~/.zsh_history` shows these steps just prior to getting things working: 
- use `conda` .. I'm not normally a fan of how bloated `conda` environments can be, so I tend to `venv` and `pip install`, but in this case.. it was needed for the `pyini` install that `nemo_text_processing` relies on
- `conda install -c conda-forge pynini=2.1.5`.
- `pip install nemo_text_processing`
- `pip install 'nemo-toolkit[all]'`
- `cd NeMo-Curator; pip install .` # note I cloned the NeMo-Curator repo for this install
- `brew install opencc`
- `export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/opencc/1.1.7/lib:$DYLD_LIBRARY_PATH`
#### NeMo Magic Sauce
I didn't get a chance to take full advantage of what I think probably makes the magic sauce for NeMo Curator. Namely...

