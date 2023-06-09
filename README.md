# MusicGAN 
## M.Sc. Thesis Project
A data science (Artificial Intelligence) master thesis about music generation with GAN models. 

## Code
Code for training and data visualisation is available in [/code](/code).
- [Training (with Emotion Constraint)](/code/emo_con_training.ipynb), the main training file of the project.
- [Training Loss Visualisation](/code/Tensorboard_viewer.ipynb), using tensorboard to visualise the training data.
- [Generator Inspector](/code/Generator_inspector.ipynb), a notebook for generating music and evaluating the model.
- [Training Parameters](/code/params.py), a file for controlling the model hyperparameters.
- [MelSpec](/code/melspec.py), a keras layer implementation for creating the mel-scaled spectrograms, original code is found here: [https://github.com/keras-team/keras-io/blob/master/examples/audio/melgan_spectrogram_inversion.py](https://github.com/keras-team/keras-io/blob/master/examples/audio/melgan_spectrogram_inversion.py).
- [No-longer–used code](/code/old_code), kept for the purpose of documentation.

## Models
The main models of the project, named A-D, are available in [/models/train](/models/train)
- [Model A (no PatchGAN)](/models/train/A_no_patchGAN)
- [Model B (no rhythm loss)](/models/train/B_no_rhythm)
- [Model C (no emotion labels)](/models/train/C_no_emo)
- [Model D (full model)](/models/train/D_full_model)


## Generated Examples
### Model B
![Spectrogram](/misc/examples/demo_B2.png)
[Audio](/misc/examples/demo_B2.mp3?raw=true)

### Model C
![Spectrogram](/misc/examples/demo_C.png)
[Audio](/misc/examples/demo_C.mp3?raw=true)

### Model D
![Spectrogram](/misc/examples/demo_D.png)
[Audio](/misc/examples/demo_D.mp3?raw=true)
