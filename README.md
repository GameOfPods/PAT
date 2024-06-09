# PAT (PodcastProject Analytics Toolkit)

Toolkit to analyze several aspects of either podcasts themselves or books talked about in the podcast

This project aims to create a collection of tools and workflows surrounding all kinds of data that might (or might not)
happen to be in the reach of a podcast. Since this project started with a podcast talking about books the first
modules all focus on analyzing the podcast audio files and the books.

## Install

```shell
pip install git+https://github.com/GameOfPods/PAT.git
```

## Running the application

See

```bash
PAT -h
```

for all available options

See

```bash
PAT -ls
```

for a list of all available Modules

## Modules

PAT automatically discovers all modules implementing `PATModule`. Each of these modules can introduce a set of command
line options that get automatically added to the cli of PAT with a prefix.

When loading a file every Module is asked if the module can and wants to handle the file. If yes all combinations of
files and modules are queued and the ordered by module. Like this a module that has a long start and stop time
(loading models, etc.) only has to go through them once.

The results are stored in a dedicated folder named `A_date_time`.

The results consist of a basic json file and potentially some additional files produced by the modules that can have
different shapes.

Currently, a speaker and a book module are implemented. More to come.

### SpeakerModule

This module aims to produce insights into the recorded podcast and the podcasters. It uses different methods to
diarize the speakers, transcribe the audio and then summarize it. For all these steps all data is recorded.

The diarization is done with either
[Nvidia NEMO](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html)
if present and all requirements are installed or using
[pyannote](https://huggingface.co/pyannote/speaker-diarization)
if its requirements are met.
If both are present the user can select which to use.
I found Nvidia NEMO to be more reliable but that might be very dependant on the data present in the specific usecase.

The transcription is done using
[openAI whisper](https://github.com/openai/whisper)
with
[whisperX](https://github.com/m-bain/whisperX)
to better align the transcription to the single words.

If both are present, diarization and transcription, they are mapped to each other to create a transcription per person.
Finally, for each part the punctuation is restored using
[deepmultilingualpunctuation](https://github.com/oliverguhr/deepmultilingualpunctuation.git).

If OpenAI is configured to run a summary is created over the transcription.
For OpenAI either the official web api can be used if a valid token is given or a local OpenAI compatible installation
(eg.
[localai](https://localai.io/features/openai-functions/)
) can be used by using the `OPENAI_API_BASE` environment variable. Also the model that should be
used has to be given in cli.

### BookModule

This module is used to do several nlp analysis on books. Currently only epub files are supported but this will be
extended in the future. For all extracted content from the book
[spacy](https://spacy.io/)
is used to create the data. Especially NER is implemented at the time to see which persons, locations are most important
and present in each chapter. This can be used for some fun analysis on books and stories.
More nlp stuff will be implemented in the future.

If OpenAI is configured to run a summary is created over the chapters and then the whole book.
For OpenAI either the official web api can be used if a valid token is given or a local OpenAI compatible installation
(eg.
[localai](https://localai.io/features/openai-functions/)
) can be used by using the `OPENAI_API_BASE` environment variable. Also the model that should be
used has to be given in cli.

## Vizualization

The focus of PAT is on creating raw data that can then be used by different UI (Websites, etc) of further tools.
Mainly
[podcastproject](https://github.com/GameOfPods/podcastproject)
will get native support for several modules to show the data directly on a website that also shows the podcast.

But PAT also has a sister module `PAT_Visualizer` that aims to create some basic insights into some of the data.
This should really be just a glimps of what is possible with the data `PAT` generates.

For docu on `PAT_Visualizer` see

```bash
PAT-Visualizer -h
```

## TODOs

- Create full documentation (readthedocs or similar)
- Improve mapping of words to speaker. Sometimes very buggy
- Seperate requirements by modules. Utilize extras requires from setuptools?
- Do speaker mapping using voice markers for each speaker by providing samples
   - Mel-frequency cepstral coefficients
- Look into summarization task and improve prompts
