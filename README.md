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
