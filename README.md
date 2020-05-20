![Flowtron](https://nv-adlr.github.io/images/flowtron_logo.png "Flowtron")

Based on "Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis" (Rafael Valle, Kevin Shih, Ryan Prenger and Bryan Catanzaro)


## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/flowtron.git`
2. CD into this repo: `cd flowtron`
3. Initialize submodule: `git submodule update --init; cd tacotron2; git submodule update --init`
4. Install [PyTorch]
5. Install python requirements or build docker image
    - Install python requirements: `pip install -r requirements.txt`
6. Download pretrained models and place them in `models/`. [Flowtron LJS](https://drive.google.com/open?id=1Cjd6dK_eFz6DE0PKXKgKxrzTUqzzUDW-), [WaveGlow v4](https://drive.google.com/file/d/1okuUstGoBe_qZ4qUEF8CcwEugHP7GM_b/view).

## My inference demo
1. Run `inference_realtime.py`, you'll get presented with a keyboard loop. Every sentence ending with full stop is picked up, modeled and played in a separate thread, then it waits for a new sentence. Right alt is exit.
2. Change n_frames, gate_threshold in `inference_realtime.py` according to your intuition. Larger n_frames should require more GPU memory but also allows to model longer sentences.

## Original inference demo
1. `python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4.pt -t "It is well know that deep generative models have a deep latent space!" -i 0` - dumps the result in a file.
