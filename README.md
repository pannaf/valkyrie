# V : AI Personal Trainer

Meet V! Your new virtual personal trainer! 🙃

This repo has the code for my entry in the [Generative AI Agents Developer Contest by NVIDIA and LangChain](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/).

## Main Tech
- [x] <img src="https://github.com/pannaf/valkyrie/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> [NeMo Curator](#-nemo-curator-building-an-exercise-dataset) - build a dataset of exercises that V can draw from when planning workouts 💪
- [x] <img src="https://github.com/pannaf/valkyrie/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="35"/> [LangGraph](#-langgraph-v-as-an-agent) - V as an agent
- [x] <img src="https://github.com/pannaf/valkyrie/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="35"/> [LangSmith](#-langsmith-langgraph-tracing) - LangGraph tracing
- [x] <img src="https://github.com/pannaf/valkyrie/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="15"/> [NeMo Guardrails](#-nemo-guardrails-ensuring-v-stays-out-of-the-medical-domain) - ensure V doesn't venture into a medical domain space

## Setup

### Requirements
- Python 3.12.3, for the main project
- Python 3.10.14, for NeMo Curator (see [notes](#a-few-notes) below)

### Installation
Clone the repo and `cd` into the code directory
```bash
➜  git clone https://github.com/pannaf/valkyrie.git
➜  cd valkyrie
```
#### MacOS Setup with venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nemoguardrails==0.9.0
pip install -r requirements.txt
```
There's a small dependency conflict with the LangChain version for `nemoguardrails` that I did a small workaround for by installing `nemoguardrails` first. I still wind up with this warning:
```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nemoguardrails 0.9.0 requires langchain!=0.1.9,<0.2.0,>=0.1.0, but you have langchain 0.2.3 which is incompatible.
nemoguardrails 0.9.0 requires langchain-community<0.1.0,>=0.0.16, but you have langchain-community 0.2.4 which is incompatible.
```
But.. things ran fine for me with this setup, so I didn't spend time looking into resolving this further.  

#### Environment Variables
To avoid seeing the following warnings, set the `TOKENIZERS_PARALLELISM` environment variable to `false`:
```text
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
dialog_state=[]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```
Use this command to set the environment variable:
```bash
(.venv) ➜  valkyrie git:(main) ✗ export TOKENIZERS_PARALLELISM=False
```

#### Postgres Install with brew
```bash
(.venv) ➜  valkyrie git:(main) ✗ brew install postgresql
(.venv) ➜  valkyrie git:(main) ✗ brew services start postgresql
```
Verify PostgreSQL is running via `brew services list`. On my machine, I see the following:
```bash
(.venv) ➜  valkyrie git:(main) ✗ brew services list
Name          Status  User  File
postgresql@14 started panna ~/Library/LaunchAgents/homebrew.mxcl.postgresql@14.plist
```

### Demo
1. Create `.env` file in your code root directory
```bash
# POSTGRES
DATABASE_URL=...

# Anthropic
ANTHROPIC_API_KEY=...

# OPENAI
OPENAI_API_KEY=...

# Langsmith
LANGSMITH_API_KEY=...
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT=...

# NVIDIA API KEY
NVIDIA_API_KEY=...
```   
2. Run `bootstrap.py` script to setup needed tables and create an initial user.
```bash
(.venv) ➜  valkyrie git:(main) ✗ python bootstrap.py
```
This should look something like:
```text
(.venv) ➜  valkyrie git:(main) ✗ python bootstrap.py
DATABASE_URL='postgresql://<username>:<password>@localhost:5432/<db name>'
Tables created successfully.
Enter your first name: <your name>
Enter your last name: <your name>
Enter your email: <your email>
Enter your date of birth (YYYY-MM-DD): <your date of birth>
Enter your height in centimeters: <your height>
User created with ID: <uuid that was created>
Updated configs/agent.yaml with user_id: <uuid that was created>
```
3. Run V
```bash
➜  valkyrie git:(main) ✗ python -m src.assistant_system
``` 
You should see something along the lines of:
```bash
(.venv) ➜  valkyrie git:(main) ✗ python -m src.assistant_system 
2024-06-17 10:57:36.160 | INFO     | __main__:main:76 - Starting V | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.273 | INFO     | __main__:__init__:35 - Building graph | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.274 | DEBUG    | __main__:__init__:38 - Entering 'get_graph' (args=(<src.state_graph.graph_builder.GraphBuilder object at 0x14e2e57f0>,), kwargs={}) | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.274 | DEBUG    | src.state_graph.graph_builder:get_graph:234 - Entering 'build' (args=(<src.state_graph.graph_builder.GraphBuilder object at 0x14e2e57f0>,), kwargs={}) | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.274 | DEBUG    | src.state_graph.graph_builder:get_graph:234 - Function 'build' executed in 0.0004767079371958971s | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.274 | DEBUG    | src.state_graph.graph_builder:get_graph:234 - Exiting 'build' (result=<langgraph.graph.state.StateGraph object at 0x14e2e5550>) | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.274 | DEBUG    | src.state_graph.graph_builder:get_graph:235 - Entering 'compile' (args=(<src.state_graph.graph_builder.GraphBuilder object at 0x14e2e57f0>, <langgraph.graph.state.StateGraph object at 0x14e2e5550>), kwargs={}) | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.276 | DEBUG    | src.state_graph.graph_builder:get_graph:235 - Function 'compile' executed in 0.0014505410799756646s | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.276 | SUCCESS  | src.state_graph.graph_builder:get_graph:236 - Graph built and compiled. | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.277 | INFO     | src.state_graph.graph_builder:get_graph:237 - Graph structure:
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__[__start__]:::startclass;
        __end__[__end__]:::endclass;
        fetch_user_info([fetch_user_info]):::otherclass;
        guardrails_input_handler([guardrails_input_handler]):::otherclass;
        enter_onboarding_wizard([enter_onboarding_wizard]):::otherclass;
        onboarding_wizard([onboarding_wizard]):::otherclass;
        onboarding_wizard_safe_tools([onboarding_wizard_safe_tools]):::otherclass;
        onboarding_wizard_sensitive_tools([onboarding_wizard_sensitive_tools]):::otherclass;
        enter_goal_wizard([enter_goal_wizard]):::otherclass;
        goal_wizard([goal_wizard]):::otherclass;
        goal_wizard_safe_tools([goal_wizard_safe_tools]):::otherclass;
        goal_wizard_sensitive_tools([goal_wizard_sensitive_tools]):::otherclass;
        enter_programming_wizard([enter_programming_wizard]):::otherclass;
        programming_wizard([programming_wizard]):::otherclass;
        programming_wizard_safe_tools([programming_wizard_safe_tools]):::otherclass;
        programming_wizard_sensitive_tools([programming_wizard_sensitive_tools]):::otherclass;
        enter_v_wizard([enter_v_wizard]):::otherclass;
        v_wizard([v_wizard]):::otherclass;
        v_wizard_safe_tools([v_wizard_safe_tools]):::otherclass;
        v_wizard_sensitive_tools([v_wizard_sensitive_tools]):::otherclass;
        leave_skill([leave_skill]):::otherclass;
        primary_assistant([primary_assistant]):::otherclass;
        primary_assistant_tools([primary_assistant_tools]):::otherclass;
        __start__ --> fetch_user_info;
        enter_goal_wizard --> goal_wizard;
        enter_onboarding_wizard --> onboarding_wizard;
        enter_programming_wizard --> programming_wizard;
        enter_v_wizard --> v_wizard;
        fetch_user_info --> guardrails_input_handler;
        goal_wizard_safe_tools --> goal_wizard;
        goal_wizard_sensitive_tools --> goal_wizard;
        leave_skill --> primary_assistant;
        onboarding_wizard_safe_tools --> onboarding_wizard;
        onboarding_wizard_sensitive_tools --> onboarding_wizard;
        primary_assistant_tools --> primary_assistant;
        programming_wizard_safe_tools --> programming_wizard;
        programming_wizard_sensitive_tools --> programming_wizard;
        v_wizard_safe_tools --> v_wizard;
        v_wizard_sensitive_tools --> v_wizard;
        onboarding_wizard -.-> onboarding_wizard_safe_tools;
        onboarding_wizard -.-> onboarding_wizard_sensitive_tools;
        onboarding_wizard -.-> leave_skill;
        onboarding_wizard -.-> __end__;
        goal_wizard -.-> goal_wizard_safe_tools;
        goal_wizard -.-> goal_wizard_sensitive_tools;
        goal_wizard -.-> leave_skill;
        goal_wizard -.-> __end__;
        programming_wizard -.-> programming_wizard_safe_tools;
        programming_wizard -.-> programming_wizard_sensitive_tools;
        programming_wizard -.-> leave_skill;
        programming_wizard -.-> __end__;
        v_wizard -.-> v_wizard_safe_tools;
        v_wizard -.-> v_wizard_sensitive_tools;
        v_wizard -.-> leave_skill;
        v_wizard -.-> __end__;
        primary_assistant -.-> enter_onboarding_wizard;
        primary_assistant -.-> enter_goal_wizard;
        primary_assistant -.-> enter_programming_wizard;
        primary_assistant -.-> enter_v_wizard;
        primary_assistant -.-> primary_assistant_tools;
        primary_assistant -.-> __end__;
        guardrails_input_handler -.-> onboarding_wizard;
        guardrails_input_handler -.-> goal_wizard;
        guardrails_input_handler -.-> programming_wizard;
        guardrails_input_handler -.-> primary_assistant;
        guardrails_input_handler -.-> v_wizard;
        guardrails_input_handler -.-> __end__;
        classDef startclass fill:#ffdfba;
        classDef endclass fill:#baffc9;
        classDef otherclass fill:#fad7de;
 | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.278 | DEBUG    | __main__:__init__:38 - Function 'get_graph' executed in 0.0039283750811591744s | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.278 | INFO     | __main__:main:81 - Saving graph visualization to v_graph.png | f7877a93-30e4-43ff-9969-1ec6b1e03b9b
2024-06-17 10:57:36.474 | SUCCESS  | src.state_graph.graph_builder:visualize_graph:245 - Graph saved to v_graph.png | f7877a93-30e4-43ff-9969-1ec6b1e03b9b

---------------- User Message ----------------
User: 
```
From there, you can start chatting with V. Notice that the uuid attached to each log message should match the uuid that was created during bootstrap for the user you created.

## <img src="https://github.com/pannaf/valkyrie/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="25"/> [NeMo Curator] Building an Exercise Dataset
To construct meaningful workouts, V needed to draw from a solid exercise list with a diverse set of movements. While this list could have been generated by prompting an LLM, doing so runs the risk of hallucination and lack of comprehensiveness. On the other hand, scraping credible fitness websites ensures accurate, relevant, and consistent information from domain experts.

### <img src="https://github.com/pannaf/valkyrie/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="20"/> [NeMo Curator] Generating an Exercise List
I used [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) to generate an exercise list with a pipeline that gathers, cleans, and processes web-scraped data.

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

1. Collaborated with ChatGPT to fill out an initial pass at annotating each of the ~1400 exercises with the following attributes: `Exercise Name,Exercise Type,Target Muscle Groups,Movement Pattern,Exercise Difficulty/Intensity,Equipment Required,Exercise Form and Technique,Exercise Modifications and Variations,Safety Considerations,Primary Goals,Exercise Dynamics,Exercise Sequence,Exercise Focus,Agonist Muscles,Synergist Muscles,Antagonist Muscles,Stabilizer Muscles`
2. Scanned through the ChatGPT annotations to correct any obvious mistakes.
   
The attributes provide the sufficient context needed for V to select appropriate exercises for a user, based on their fitness level and preferences.

[back to top](#tech-used)

## <img src="https://github.com/pannaf/valkyrie/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="55"/> [LangGraph] V as an Agent
To create V, I followed the LangGraph Customer Support Bot tutorial [here](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/). Finding the ipynb with the code [here](https://github.com/langchain-ai/langgraph/blob/main/examples/customer-support/customer-support.ipynb) was clutch. As in the tutorial, I separated the sensitive tools (ones that update the DB) from the safe tools. Refer to [Section 2: Add Confirmation](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#part-2-add-confirmation) in the tutorial. But, I didn't like the user experience where an AI message didn't follow the human-in-the-loop approval when invoking a sensitive tool because it felt like the user needed to do more to drive the conversation than what I had in mind with V. For example:
```bash
Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changed.

 y
================================ Human Message =================================

<requires human message after approval>
```
While I do see error modes with V where the database doesn't get updated in the ways that I had in mind, it wasn't sufficiently prohibitive to warrant exceeding my time box on investigating alternatives with including the user confirmation on sensitive tools. I do plan to circle back to this eventually though because it seems important for robustifying V. 

### V's LangGraph Structure
![v_graph](https://github.com/pannaf/valkyrie/assets/18562964/b9d18dc5-0fee-4a6f-889b-77fda728023d)

I refer to each of V's assistants, except for the primary assistant as a "wizard" that V can invoke. 

### Things V can do
- [Onboard a New User](#onboard-a-new-user): gather basic info about a user
- [Goal Setting](#goal-setting): help a user set goals and update their goals
- [Workout Programming](#workout-programming): given the user profile and their goals, plan a 1 week workout program
- [Answer Questions about V](#answer-questions-about-v): share a little personality and answer some basic design questions

#### Onboard a New User
`onboarding_wizard`

Two primary objectives:
1. Getting to know a new user
2. Updating the user's profile information

Tools available:
- `fetch_user_profile_info` : Used for retrieving the user's profile information, so that V can know what info is filled and what info still needs to be filled.
- `set_user_profile_info` : Used to update the user's profile information, as V learns more about the user.  

This is used to update the `user_profiles` table, which has the following keys:
- `user_profile_id` (uuid) - this is just the uuid primary key
- `user_id` (uuid) - this is a foreign key to match the primary key of the `users` table
- `last_updated` (date)
- `activity_preferences` (text) - I'd like to add more structure to this. Currently the LLM has a lot of wiggle room to decide what to drop in here. For example, if I have a single activity then this may just be a string like `"swimming"` but if I like many things, then this could be a list like `['swimming', 'weightlifting', 'water polo']`.
- `workout_location` (text) - This has a similar issue to the one above, except it's a bit more nuanced. For multi-activity things, sometimes I see a list like `['YMCA for swimming', '24 Hour Fitness for weightlifting', 'Community college pool for water polo']` and other times I see a dict like `{'swimming': 'YMCA', 'weightlifting': '24 Hour Fitness', 'water polo': 'Community college pool']`
- `workout_frequency` (text) - Similar to the two above, this can end up being a dict like `{'weightlifting': 4, 'swimming': 5, 'water polo': 1}`
- `workout_duration` (text) - Like the one above, this can end up being a dict like `{'weightlifting': 60, 'swimming': 60, 'water polo': 90}`
- `fitness_level` (text) - This one is all over the place and could also benefit from more guidance on what structure I'd like to see 🙃 I've seen it offer `"beginner", "intermediate", or "advanced"` and I've seen it ask for fitness level on a scale of 1-5.
- `weight` (double precision) - The user's current weight. I opted for this to go in the `user_profiles` table instead of `users` table because it's a bit more dynamic. Many folks who workout have goals around their weight.. losing weight, bulking, adding muscle, etc.
- `goal_weight` (double precision) - Given how common it is for folks to have weight-related goals, it seemed more convenient in the user experience flow to chat about the goal weight here, instead of during goal setting.

#### Goal Setting
TODO

#### Workout Programming
TODO

#### Answer Questions about V
I wanted to mix a little personality and fun into V with this little easter egg. Asking about V should invoke the "V Wizard" workflow that has a prompt that can provide some basic info about V 🙃

[back to top](#tech-used)

## <img src="https://github.com/pannaf/valkyrie/assets/18562964/c579f82c-7fe8-4709-8b4c-379573843545" alt="image" width="55"/> [LangSmith] LangGraph Tracing
Initially, I found it extremely helpful to look at the traces in LangSmith to verify that V was actually persistently staying in the correct wizard workflow. I used it to help identify a bug in my state where I wasn't correctly passing around the `dialog_state`.

Example from when I had the bug:

<img width="720" alt="langgraph-debug-trace" src="https://github.com/pannaf/valkyrie/assets/18562964/11cdd753-7210-4356-a6bd-906f10011295">

Notice that in the trace, V doesn't correctly leave the primary assistant to enter the Goal Wizard.

Correct version:

<img width="720" alt="langgraph-correct-trace" src="https://github.com/pannaf/valkyrie/assets/18562964/b638db69-c980-4ddf-89a2-39deb0047761">

[back to top](#tech-used)

## <img src="https://github.com/pannaf/valkyrie/assets/18562964/3ec5b89a-8634-492f-8077-b636466de285" alt="image" width="25"/> [NeMo Guardrails] Ensuring V Stays out of the Medical Domain
I used NeMo Guardrails to apply checks on the user input message, as a way of ensuring V doesn't engage meaningfully with a user on topics that land in the medical domain where only a licensed medical professional has the requisite expertise.

### LangChain Integration
> Didn't work for me with LangGraph.

I attempted to follow [this NVIDIA NeMo Guardrails tutorial](https://docs.nvidia.com/nemo/guardrails/user_guides/langchain/langchain-integration.html) to integrate with my LangGraph agent with the `RunnableRails` class as:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# ... initialize `some_chain`

config = RailsConfig.from_path("path/to/config")

# Using LCEL, you first create a RunnableRails instance, and "apply" it using the "|" operator
guardrails = RunnableRails(config)
chain_with_guardrails = guardrails | some_chain

# Alternatively, you can specify the Runnable to wrap
# when creating the RunnableRails instance.
chain_with_guardrails = RunnableRails(config, runnable=some_chain)
```

With my LangGraph chain including a model with `.bind_tools()` I wasn't able to get this working in my time box. Instead, I added the guardrails in a LangGraph node, as described below.

### Standard `rails.generate()` in a LangGraph Node

What ended up working for me was to add a node at the top of my graph that does a check on the user messages, following the [Input Rails guide](https://docs.nvidia.com/nemo/guardrails/getting_started/4_input_rails/README.html) and the [Topical Rails guide](https://docs.nvidia.com/nemo/guardrails/getting_started/6_topical_rails/README.html). The node simply updates a `valid_input` field in the state, which is checked when determining which workflow to route to. When `valid_input` is `False`, V outputs the guardrails message and jumps to `END` to allow for user input.

[back to top](#tech-used)
