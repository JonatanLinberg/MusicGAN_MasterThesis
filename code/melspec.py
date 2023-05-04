import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
import params

class MelSpec(layers.Layer):
    def __init__(
        self,
        frame_length=params.STFT_WINDOW,
        frame_step=params.STFT_HOP,
        fft_length=None,
        sampling_rate=params.SAMPLE_RATE,
        num_mel_channels=params.MEL_BINS,
        freq_min=params.MEL_FREQ_MIN,
        freq_max=params.MEL_FREQ_MAX,
        min_db = -params.DB_MIN,   # tf wants the positive version
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_db = min_db
        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, audio, training=True):
        # We will only perform the transformation during training.
        if training:
            # Taking the Short Time Fourier Transform. Ensure that the audio is padded.
            # In the paper, the STFT output is padded using the 'REFLECT' strategy.
            stft = tf.signal.stft(
                tf.squeeze(audio, -1),
                self.frame_length,
                self.frame_step,
                self.fft_length,
                pad_end=True,
            )

            # Taking the magnitude of the STFT output
            magnitude = tf.abs(stft)

            # Multiplying the Mel-filterbank with the magnitude and scaling it using the db scale
            mel = tf.matmul(magnitude, self.mel_filterbank)
            log_mel_spec = tfio.audio.dbscale(mel, top_db=self.min_db)
            return log_mel_spec
        else:
            return audio

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_channels": self.num_mel_channels,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config

if __name__=='__main__':
    print(MelSpec().get_config())