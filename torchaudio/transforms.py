from __future__ import division, print_function
import torch
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None
try:
    import pysptk
except ImportError:
    pysptk = None


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".

    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth

    """

    def __init__(self, factor=2**31):
        self.factor = factor

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)

        """
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        return tensor / self.factor

class PadTrim(object):
    """Pad/Trim a 1d-Tensor (Signal or Labels)

    Args:
        tensor (Tensor): Tensor of audio of size (Samples x Channels)
        max_len (int): Length to which the tensor will be padded

    """

    def __init__(self, max_len, fill_value=0):
        self.max_len = max_len
        self.fill_value = fill_value

    def __call__(self, tensor):
        """

        Returns:
            Tensor: (max_len x Channels)

        """
        if self.max_len > tensor.size(0):
            pad = torch.ones((self.max_len-tensor.size(0),
                              tensor.size(1))) * self.fill_value
            pad = pad.type_as(tensor)
            tensor = torch.cat((tensor, pad), dim=0)
        elif self.max_len < tensor.size(0):
            tensor = tensor[:self.max_len, :]
        return tensor


class DownmixMono(object):
    """Downmix any stereo signals to mono

    Inputs:
        tensor (Tensor): Tensor of audio of size (Samples x Channels)

    Returns:
        tensor (Tensor) (Samples x 1):

    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        if tensor.size(1) > 1:
            tensor = torch.mean(tensor.float(), 1, True)
        return tensor

class LC2CL(object):
    """Permute a 2d tensor from samples (Length) x Channels to Channels x
       samples (Length)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.transpose(0, 1).contiguous()


class MEL(object):
    """Create MEL Spectrograms from a raw audio signal. Relatively pretty slow.

       Usage (see librosa.feature.melspectrogram docs):
           MEL(sr=16000, n_fft=1600, hop_length=800, n_mels=64)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (samples x channels)

        Returns:
            tensor (Tensor): n_mels x hops x channels (BxLxC), where n_mels is
                the number of mel bins, hops is the number of hops, and channels
                is unchanged.

        """
        if librosa is None:
            print("librosa not installed, cannot create spectrograms")
            return tensor
        L = []
        for i in range(tensor.size(1)):
            nparr = tensor[:, i].numpy() # (samples, )
            sgram = librosa.feature.melspectrogram(nparr, **self.kwargs) # (n_mels, hops)
            L.append(sgram)
        L = np.stack(L, 2) # (n_mels, hops, channels)
        tensor = torch.from_numpy(L).type_as(tensor)

        return tensor

class BLC2CBL(object):
    """Permute a 3d tensor from Bands x samples (Length) x Channels to Channels x
       Bands x samples (Length)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.permute(2, 0, 1).contiguous()

class MuLawEncoding(object):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int): Number of channels. default: 256

    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x):
        """

        Args:
            x (FloatTensor/LongTensor or ndarray)

        Returns:
            x_mu (LongTensor or ndarray)

        """
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.FloatTensor([mu])
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
        return x_mu

class MuLawExpanding(object):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels. default: 256

    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x_mu):
        """

        Args:
            x_mu (FloatTensor/LongTensor or ndarray)

        Returns:
            x (FloatTensor or ndarray)

        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

class MultiAcoFeats(object):
    """ Extract acoustic features and compose all outputs,
    with wavs, into a list of tensors
    """

    def __init__(self, sr=16000, n_fft=1024, hop_length=80,
                 win_length=320, window='hann'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.mel = MEL(sr=sr, n_fft=n_fft, hop_length=hop_length)

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (samples x 1)

        """
        t_npy = tensor.cpu().squeeze(1).numpy()
        # MelSpectrum and MFCCs
        mel = self.mel(tensor).transpose(0, 1).squeeze(2)
        ret = {'mel_spec':mel}
        mfcc = librosa.feature.mfcc(y=t_npy, sr=self.sr,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length)
        ret['mfcc'] = torch.FloatTensor(mfcc.T)
        # Spectrogram abs magnitude [dB]
        spec = librosa.stft(t_npy, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window)
        spec_db = librosa.amplitude_to_db(spec)
        ret['spec'] = torch.FloatTensor(spec_db.T)
        # ZCR, E and lF0
        if pysptk is None:
            print("pysptk not installed, cannot compute F0")
            return tensor
        else:
            from ahoproc_tools.interpolate import interpolation
            f0 = pysptk.swipe(t_npy.astype(np.float64), fs=self.sr,
                              hopsize=self.hop_length, min=60, max=240,
                              otype="f0")
            lf0 = np.log(f0 + 1e-10)
            lf0, uv = interpolation(lf0, -1)
            ret['lf0'] = torch.FloatTensor(lf0).view(-1, 1)
            ret['uv'] = torch.FloatTensor(uv.astype(np.float32)).view(-1, 1)
        egy = librosa.feature.rmse(y=t_npy, frame_length=self.win_length,
                                   hop_length=self.hop_length,
                                   pad_mode='constant')
        zcr = librosa.feature.zero_crossing_rate(y=t_npy,
                                                 frame_length=self.win_length,
                                                 hop_length=self.hop_length)
        ret['egy'] = torch.FloatTensor(egy).view(-1, 1)
        ret['zcr'] = torch.FloatTensor(zcr).view(-1, 1)
        ret['wav'] = tensor.view((-1, 1))
        return ret



