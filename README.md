# 🚀 BARK INFINITY with Evaluating AI-Generated Audio Quality 🎶
### Evaluating AI-Generated Audio Quality with Python

This code is an augmentation of the original script BARK INFINITY script, an AI-based text-to-voice model that generates an audio file. The goal of this script is to evaluate the output of the model, reject unsatisfactory samples, and render the final result. The function generate_audio_with_zcr_check attempts to create an audio file that meets specific quality criteria by employing multiple evaluation metrics, such as Zero Crossing Rate (ZCR), Spectral Contrast, and Bass Energy.

The script continuously executes the loop while attempts < max_attempts: until the number of attempts reaches the maximum specified by the max_attempts parameter. During each iteration, the function tries to generate the audio and verifies if it surpasses the specified thresholds.

Within the loop, the conditional if attempts > 0 and base is not None: checks if this is not the first attempt (attempts > 0) and if a valid base model token is supplied (base is not None). If both conditions hold true, it resets the base model token by setting base = None. This action allows subsequent calls to generate_audio to employ a new base model token, potentially generating a different audio sample that could pass the thresholds. If the base token remains unchanged, the function might continuously generate the same audio sample, repeatedly failing the thresholds.

ZCR is a feature employed in audio signal processing and speech recognition, representing the rate at which the audio signal changes its sign or crosses the zero-amplitude level. In speech signals, a higher ZCR typically signifies unvoiced or fricative sounds (such as 's' or 'f'), while a lower ZCR is linked to voiced sounds (like vowels). ZCR helps differentiate various types of sounds and evaluate synthesized speech quality.

Spectral Contrast measures the difference between peaks (high energy) and valleys (low energy) in an audio signal's frequency spectrum. It captures the relative prominence of distinct frequency components, offering information about the spectral content's uniqueness. In speech and audio processing, spectral contrast aids in assessing the intelligibility and quality of synthesized speech by examining the energy distribution across the frequency spectrum.

Bass Energy denotes the average energy in an audio signal's lower frequency range, typically associated with bass sounds. In speech signals, bass energy captures low-frequency components, such as a speaker's voice's fundamental frequency or the energy of voiced sounds. Analyzing bass energy proves useful in evaluating synthesized speech quality and naturalness, as it contributes to the audio signal's overall perception.

To install the additional packages required

```python
pip install librosa bark scipy soundfile
```
### Some limitations and potential flaws:

The function might be more suitable for a male voice, and it may not produce satisfactory results for female voices or voices with different characteristics. This limitation could be attributed to the model used for generating the audio samples.

The volume levels of the generated audio might be inconsistent. This could affect the overall listening experience and make it harder to understand the speech.

The voice may change in tenor and type during the audio generation process. This can lead to an inconsistent and less natural-sounding output, which is also a flaw of the model being used.

To improve the performance and make the function more suitable for various voice types, the model would need to be updated to a larger and more sophisticated one. A better model could potentially deliver more consistent results in terms of volume levels, tenor, and voice type while maintaining intelligibility and complying with the specified thresholds.
April 22: The main bark repo just implemented small model support, so if just want that you don't need use this fork. If I can find time this weekend I'll unhack the way I did and use their parameters.

April 25: I will update this with the main Bark repo Tuesday evening (which had a speed increase over the weekend). There's probably going to be two files, a light bark wrapper which you can just pop into the regular Bark repo as a pure wrapper, so it never gets out of date. And another file that requires forking this whole repo and installing it, with more options and power for how to build longer clips especially.

**Now it's BARK INFINITY!** 🎉


### 🌟 Main Features 🌟

### 1. INFINITY VOICES 🔊🌈
Discover cool new voices and reuse them. Performers, musicians, sound effects, two party dialog scenes. Save and share them. Every audio clip saves a speaker.npz file with the voice. To reuse a voice, move the generated speaker.npz file (named the same as the .wav file) to the "prompts" directory inside "bark" where all the other .npz files are.

🔊 With random celebrity appearances!

(I accidently left a bunch of voices in the repo, some of them are pretty good. Use --history_prompt 'en_fiery' for the same voice as the audio sample right after this sentence.)

https://user-images.githubusercontent.com/163408/233747981-173b5f03-654e-4a0e-b71b-5d220601fcc7.mp4



### 2. INFINITY LENGTH 🎵🔄
Any length prompt and audio clips. Sometimes the final result is seamless, sometimes it's stable (but usually not both!).

🎵 Now with Slowly Morphing Rick Rolls! Can you even spot the seams in the most earnest Rick Rolls you've ever heard in your life?

https://user-images.githubusercontent.com/163408/233747400-b18411f8-afcb-437d-9288-c54cc2c95e62.mp4

### 🕺 Confused Travolta Mode 🕺
Confused Travolta GIF
![confused_travolta](https://user-images.githubusercontent.com/163408/233747428-c6bf03e2-b3ce-4ce3-a29d-836bf73a4ec2.gif)

Can your text-to-speech model stammer and stall like a student answering a question about a book they didn't read? Bark can. That's the human touch. The *semantic* touch. You can almost feel the awkward silence through the screen.

## 💡 But Wait, There's More: Travolta Mode Isn't Just A Joke 💡

Are you tired of telling your TTS model what to say? Why not take a break and let your TTS model do the work for you. With enough patience and Confused Travolta Mode, Bark can finish your jokes for you. 

https://user-images.githubusercontent.com/163408/233746957-f3bbe25f-c8f0-4570-97b1-1005e1b40cbe.mp4

Truly we live in the future. It might take 50 tries to get a joke and it's probabably an accident, but all 49 failures are also *very* amusing so it's a win/win. (That's right, I set a single function flag to False in a Bark and raved about the amazing new feature. Everything here is small potatoes really.)

https://user-images.githubusercontent.com/163408/233746872-cac78447-8e87-49e7-b79b-28ec51264019.mp4



_**BARK INFINITY** is possible because Bark is such an amazingly simple and powerful model that even I could poke around easily._

_For music, I recommend using the --split_by_lines and making sure you use a multiline string as input. You'll generally get better results if you manually split your text, which I neglected to provide an easy way to do because I stayed too late listening to 100 different Bark versions of a scene an Andor and failed Why was 6 afraid of 7 jokes._

## 📝 Command Line Options 📝

```bash
--text_prompt                Text prompt. If not provided, a set of default prompts will be used defined in this file.
--history_prompt             Optional. Choose a speaker from the list of languages. Use --list_speakers to see all available options.
--text_temp                  Text temperature. Default is 0.7.
--waveform_temp              Waveform temperature. Default is 0.7.
--filename                   Output filename. If not provided, a unique filename will be generated based on the text prompt and other parameters.
--output_dir                 Output directory. Default is 'bark_samples'.
--list_speakers              List all preset speaker options instead of generating audio.
--use_smaller_models         Use for GPUs with less than 10GB of memory, or for more speed.
--iterations                 Number of iterations. Default is 1.
--split_by_words             Breaks text_prompt into <14 second audio clips every x words.
--split_by_lines             Breaks text_prompt into <14 second audio clips every x lines.
--stable_mode                Choppier and not as natural sounding, but much more stable for very long audio files.
--confused_travolta_mode     Just for fun. Try it, and you'll understand. 🤷

--prompt_file                Optional. The path to a file containing the text prompt. Overrides the --text_prompt option if provided.
--prompt_file_separator      Optional. The separator used to split the content of the prompt_file into multiple text prompts.
```
## 🎉 Get Started 🎉

1. Clone the Bark repository: `git clone https://github.com/JonathanFly/bark.git`
2. Install the required package: `pip install soundfile`
3. Run the example command:

```
python bark_perform.py --text_prompt "It is a mistake to think you can solve any major problems just with potatoes... or can you? (and the next page, and the next page...)" --split_by_words 35
```

_If you can't get Bark installed, you might try this one-click installer: [https://github.com/Fictiverse/bark/releases](https://github.com/Fictiverse/bark/releases) - but you'll still need to clone or copy all the files in this specific bark repo into the bark directory because I don't know what I'm doing._

I haven't posted much lately I dipped my toes back into a bit [twitter.com/jonathanfly](https://twitter.com/jonathanfly)


Original Bark README:

# 🐶 Bark

<a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/OnusFM.svg?style=social&label=@OnusFM)](https://twitter.com/OnusFM)
[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?compact=true&style=flat&)](https://discord.gg/J2B2vsjKuE)


[Examples](https://suno-ai.notion.site/Bark-Examples-5edae8b02a604b54a42244ba45ebc2e2) | [Model Card](./model-card.md) | [Playground Waitlist](https://3os84zs17th.typeform.com/suno-studio)

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints ready for inference.

<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/230698495-cbb1ced9-c911-4c9a-941d-a1a4a1286ac6.png" width="500"></img>
</p>

## 🔊 Demos

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/suno/bark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

## 🤖 Usage

```python
from bark import SAMPLE_RATE, generate_audio
from IPython.display import Audio

text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza.webm](https://user-images.githubusercontent.com/5068315/230490503-417e688d-5115-4eee-9550-b46a2b465ee3.webm)


To save `audio_array` as a WAV file:

```python
from scipy.io.wavfile import write as write_wav

write_wav("/path/to/audio.wav", SAMPLE_RATE, audio_array)
```

### 🌎 Foreign Language

Bark supports various languages out-of-the-box and automatically determines language from input text. When prompted with code-switched text, Bark will even attempt to employ the native accent for the respective languages in the same voice.

```python
text_prompt = """
    Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. 
    But I suppose your english isn't terrible.
"""
audio_array = generate_audio(text_prompt)
```

[miguel.webm](https://user-images.githubusercontent.com/5068315/230684752-10baadfe-1e7c-46a2-8323-43282aef2c8c.webm)

### 🎶 Music

Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.

```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```

[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)

### 🎤 Voice/Audio Cloning

Bark has the capability to fully clone voices - including tone, pitch, emotion and prosody. The model also attempts to preserve music, ambient noise, etc. from input audio. However, to mitigate misuse of this technology, we limit the audio history prompts to a limited set of Suno-provided, fully synthetic options to choose from for each language. Specify following the pattern: `{lang_code}_speaker_{number}`.

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="en_speaker_1")
```


[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)

*Note: since Bark recognizes languages automatically from input text, it is possible to use for example a german history prompt with english text. This usually leads to english audio with a german accent.*

### 👥 Speaker Prompts

You can provide certain speaker prompts such as NARRATOR, MAN, WOMAN, etc. Please note that these are not always respected, especially if a conflicting audio history prompt is given.

```python
text_prompt = """
    WOMAN: I would like an oatmilk latte please.
    MAN: Wow, that's expensive!
"""
audio_array = generate_audio(text_prompt)
```

[latte.webm](https://user-images.githubusercontent.com/5068315/230684864-12d101a1-a726-471d-9d56-d18b108efcb8.webm)


## 💻 Installation

```
pip install git+https://github.com/suno-ai/bark.git
```

or

```
git clone https://github.com/suno-ai/bark
cd bark && pip install . 
```

## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).
Running Bark requires running >100M parameter transformer models.
On modern GPUs and PyTorch nightly, Bark can generate audio in roughly realtime. On older GPUs, default colab, or CPU, inference time might be 10-100x slower. 

If you don't have new hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://3os84zs17th.typeform.com/suno-studio).

## ⚙️ Details

Similar to [Vall-E](https://arxiv.org/abs/2301.02111) and some other amazing work in the field, Bark uses GPT-style 
models to generate audio from scratch. Different from Vall-E, the initial text prompt is embedded into high-level semantic tokens without the use of phonemes. It can therefore generalize to arbitrary instructions beyond speech that occur in the training data, such as music lyrics, sound effects or other non-speech sounds. A subsequent second model is used to convert the generated semantic tokens into audio codec tokens to generate the full waveform. To enable the community to use Bark via public code we used the fantastic 
[EnCodec codec](https://github.com/facebookresearch/encodec) from Facebook to act as an audio representation.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- capitalization for emphasis of a word
- `MAN/WOMAN:` for bias towards speaker

**Supported Languages**

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |
| Arabic  | Coming soon! |
| Bengali | Coming soon! |
| Telugu | Coming soon! |

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of GPT-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for very related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under a non-commercial license: CC-BY 4.0 NC. The Suno models themselves may be used commercially. However, this version of Bark uses `EnCodec` as a neural codec backend, which is licensed under a [non-commercial license](https://github.com/facebookresearch/encodec/blob/main/LICENSE).

Please contact us at `bark@suno.ai` if you need access to a larger version of the model and/or a version of the model you can use commercially.  

## 📱 Community

- [Twitter](https://twitter.com/OnusFM)
- [Discord](https://discord.gg/J2B2vsjKuE)

## 🎧 Suno Studio (Early Access)

We’re developing a playground for our models, including Bark. 

If you are interested, you can sign up for early access [here](https://3os84zs17th.typeform.com/suno-studio).
