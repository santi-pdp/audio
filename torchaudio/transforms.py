from __future__ import division, print_function
import torch
import numpy as np
import glob
import os
import random
from scipy.signal import lfilter
from scipy.interpolate import interp1d
try:
    import librosa
except ImportError:
    librosa = None


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
                 win_length=320, window='hann', noise_dir=None, 
                 snr_levels=[0, 5, 10, 15]):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.mel = MEL(sr=sr, n_fft=n_fft, hop_length=hop_length)
        self.additive = Additive(noises_dir=noise_dir,
                                 snr_levels=snr_levels)
        self.clipping = Clipping()

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (samples x 1)

        """
        # pysptk and interpolate are a MUST in this transform
        import pysptk
        from ahoproc_tools.interpolate import interpolation
        t_npy = tensor.cpu().squeeze(1).numpy()
        # compute LF0 and UV
        f0 = pysptk.swipe(t_npy.astype(np.float64), fs=self.sr,
                          hopsize=self.hop_length, min=60, max=240,
                          otype="f0")
        lf0 = np.log(f0 + 1e-10)
        lf0, uv = interpolation(lf0, -1)
        ret = {'lf0':torch.FloatTensor(lf0).view(-1, 1),
               'uv':torch.FloatTensor(uv.astype(np.float32)).view(-1, 1)}
        tot_frames = f0.shape[0]
        # --- All librosa processes will be trimmed to same frames as in pysptk
        # MelSpectrum and MFCCs
        mel = self.mel(tensor).transpose(0, 1).squeeze(2)
        ret['mel_spec'] = mel[:tot_frames]
        mfcc = librosa.feature.mfcc(y=t_npy, sr=self.sr,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length).T
        mfcc = mfcc[:tot_frames]
        ret['mfcc'] = torch.FloatTensor(mfcc)
        # Spectrogram abs magnitude [dB]
        spec = librosa.stft(t_npy, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window)
        spec_db = librosa.amplitude_to_db(spec).T
        spec_ang = np.angle(spec).T
        spec_db = spec_db[:tot_frames]
        spec_ang = spec_ang[:tot_frames]
        ret['mag'] = torch.FloatTensor(spec_db)
        ret['pha'] = torch.FloatTensor(spec_ang)
        # ZCR, E and lF0
        egy = librosa.feature.rmse(y=t_npy, frame_length=self.win_length,
                                   hop_length=self.hop_length,
                                   pad_mode='constant').T
        egy = egy[:tot_frames]
        zcr = librosa.feature.zero_crossing_rate(y=t_npy,
                                                 frame_length=self.win_length,
                                                 hop_length=self.hop_length).T
        zcr = zcr[:tot_frames]
        ret['egy'] = torch.FloatTensor(egy)
        ret['zcr'] = torch.FloatTensor(zcr)
        if hasattr(self, 'additive'):
            do_add = random.random() > 0.5
            if do_add:
                tensor = self.additive(t_npy, self.sr)
        if hasattr(self, 'clipping'):
            do_clip = random.random() > 0.5
            if do_clip:
                tensor = self.clipping(tensor.numpy())
        ret['wav'] = tensor.view((-1, 1))
        return ret



class Additive(object):

    def __init__(self, noises_dir, snr_levels=[0, 5, 10, 15], do_IRS=False):
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.do_IRS = do_IRS
        # read noises in dir
        noises = glob.glob(os.path.join(noises_dir, '*.wav'))
        if len(noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(noises)))
            self.noises = []
            for n_i, npath in enumerate(noises, start=1):
                #nwav = wavfile.read(npath)[1]
                nwav = librosa.load(npath, sr=None)[0]
                self.noises.append({'file':npath, 
                                    'data':nwav.astype(np.float32)})
                log_noise_load = 'Loaded noise {:3d}/{:3d}: ' \
                                 '{}'.format(n_i, len(noises),
                                             npath)
                print(log_noise_load)
        self.eps = 1e-22

    def __call__(self, wav, srate, nbits=16):
        """ Add noise to clean wav """
        noise_idx = np.random.choice(list(range(len(self.noises))), 1)
        sel_noise = self.noises[np.asscalar(noise_idx)]
        noise = sel_noise['data']
        snr = np.random.choice(self.snr_levels, 1)
        # print('Applying SNR: {} dB'.format(snr[0]))
        noisy, noise_bound = self.addnoise_asl(wav, noise, srate, 
                                               nbits, snr, 
                                               do_IRS=self.do_IRS)
        # normalize to avoid clipping
        if np.max(noisy) >= 1 or np.min(noisy) < -1:
            small = 0.1
            while np.max(noisy) >= 1 or np.min(noisy) < -1:
                noisy = noisy / (1. + small)
                small = small + 0.1
        return torch.FloatTensor(noisy.astype(np.float32))


    def addnoise_asl(self, clean, noise, srate, nbits, snr, do_IRS=False):
        if do_IRS:
            # Apply IRS filter simulating telephone 
            # handset BW [300, 3200] Hz
            clean = self.apply_IRS(clean, srate, nbits)
        Px, asl, c0 = self.asl_P56(clean, srate, nbits)
        # Px is active speech level ms energy
        # asl is active factor
        # c0 is active speech level threshold
        x = clean
        x_len = x.shape[0]

        noise_len = noise.shape[0]
        if noise_len <= x_len:
            print('Noise length: ', noise_len)
            print('Speech length: ', x_len)
            raise ValueError('Noise length has to be greater than speech '
                             'length!')
        rand_start_limit = int(noise_len - x_len + 1)
        rand_start = int(np.round((rand_start_limit - 1) * np.random.rand(1) \
                                  + 1))
        noise_segment = noise[rand_start:rand_start + x_len]
        noise_bounds = (rand_start, rand_start + x_len)

        if do_IRS:
            noise_segment = self.apply_IRS(noise_segment, srate, nbits)

        Pn = np.dot(noise_segment.T, noise_segment) / x_len

        # we need to scale the noise segment samples to obtain the 
        # desired SNR = 10 * log10( Px / ((sf ** 2) * Pn))
        sf = np.sqrt(Px / Pn / (10 ** (snr / 10)))
        noise_segment = noise_segment * sf
    
        noisy = x + noise_segment

        return noisy, noise_bounds

    def apply_IRS(self, data, srate, nbits):
        """ Apply telephone handset BW [300, 3200] Hz """
        raise NotImplementedError('Under construction!')
        from pyfftw.interfaces import scipy_fftpack as fftw
        n = data.shape[0]
        # find next pow of 2 which is greater or eq to n
        pow_of_2 = 2 ** (np.ceil(np.log2(n)))

        align_filter_dB = np.array([[0, -200], [50, -40], [100, -20],
                           [125, -12], [160, -6], [200, 0],
                           [250, 4], [300, 6], [350, 8], [400, 10],
                           [500, 11], [600, 12], [700, 12], [800, 12],
                           [1000, 12], [1300, 12], [1600, 12], [2000, 12],
                           [2500, 12], [3000, 12], [3250, 12], [3500, 4],
                           [4000, -200], [5000, -200], [6300, -200], 
                           [8000, -200]]) 
        print('align filter dB shape: ', align_filter_dB.shape)
        num_of_points, trivial = align_filter_dB.shape
        overallGainFilter = interp1d(align_filter_dB[:, 0], align_filter[:, 1],
                                     1000)

        x = np.zeros((pow_of_2))
        x[:data.shape[0]] = data

        x_fft = fftw.fft(x, pow_of_2)

        freq_resolution = srate / pow_of_2

        factorDb = interp1d(align_filter_dB[:, 0],
                            align_filter_dB[:, 1],
                                           list(range(0, (pow_of_2 / 2) + 1) *\
                                                freq_resolution)) - \
                                           overallGainFilter
        factor = 10 ** (factorDb / 20)

        factor = [factor, np.fliplr(factor[1:(pow_of_2 / 2 + 1)])]
        x_fft = x_fft * factor

        y = fftw.ifft(x_fft, pow_of_2)

        data_filtered = y[:n]
        return data_filtered


    def asl_P56(self, x, srate, nbits):
        """ ITU P.56 method B. """
        T = 0.03 # time constant of smoothing in seconds
        H = 0.2 # hangover time in seconds
        M = 15.9

        # margin in dB of the diff b/w threshold and active speech level
        thres_no = nbits - 1 # num of thresholds, for 16 bits it's 15

        I = np.ceil(srate * H) # hangover in samples
        g = np.exp( -1 / (srate * T)) # smoothing factor in envelop detection
        c = 2. ** (np.array(list(range(-15, (thres_no + 1) - 16))))
        # array of thresholds from one quantizing level up to half the max
        # code, at a step of 2. In case of 16bit: from 2^-15 to 0.5
        a = np.zeros(c.shape[0]) # activity counter for each level thres
        hang = np.ones(c.shape[0]) * I # hangover counter for each level thres

        assert x.ndim == 1, x.shape
        sq = np.dot(x, x) # long term level square energy of x
        x_len = x.shape[0]

        # use 2nd order IIR filter to detect envelope q
        x_abs = np.abs(x)
        p = lfilter(np.ones(1) - g, np.array([1, -g]), x_abs)
        q = lfilter(np.ones(1) - g, np.array([1, -g]), p)

        for k in range(x_len):
            for j in range(thres_no):
                if q[k] >= c[j]:
                    a[j] = a[j] + 1
                    hang[j] = 0
                elif hang[j] < I:
                    a[j] = a[j] + 1
                    hang[j] = hang[j] + 1
                else:
                    break
        asl = 0
        asl_ms = 0
        c0 = None
        if a[0] == 0:
            return asl_ms, asl, c0
        else:
            den = a[0] + self.eps
            AdB1 = 10 * np.log10(sq / a[0] + self.eps)
        
        CdB1 = 20 * np.log10(c[0] + self.eps)
        if AdB1 - CdB1 < M:
            return asl_ms, asl, c0
        AdB = np.zeros(c.shape[0])
        CdB = np.zeros(c.shape[0])
        Delta = np.zeros(c.shape[0])
        AdB[0] = AdB1
        CdB[0] = CdB1
        Delta[0] = AdB1 - CdB1

        for j in range(1, AdB.shape[0]):
            AdB[j] = 10 * np.log10(sq / (a[j] + self.eps) + self.eps)
            CdB[j] = 20 * np.log10(c[j] + self.eps)

        for j in range(1, Delta.shape[0]):
            if a[j] != 0:
                Delta[j] = AdB[j] - CdB[j]
                if Delta[j] <= M:
                    # interpolate to find the asl
                    asl_ms_log, cl0 = self.bin_interp(AdB[j],
                                                      AdB[j - 1],
                                                      CdB[j],
                                                      CdB[j - 1],
                                                      M, 0.5)
                    asl_ms = 10 ** (asl_ms_log / 10)
                    asl = (sq / x_len ) / asl_ms
                    c0 = 10 ** (cl0 / 20)
                    break
        return asl_ms, asl, c0

    def bin_interp(self, upcount, lwcount, upthr, lwthr, Margin, tol):
        if tol < 0:
            tol = -tol

        # check if extreme counts are not already the true active value
        iterno = 1
        if np.abs(upcount - upthr - Margin) < tol:
            asl_ms_log = lwcount
            cc = lwthr
            return asl_ms_log, cc
        if np.abs(lwcount - lwthr - Margin) < tol:
            asl_ms_log = lwcount
            cc =lwthr
            return asl_ms_log, cc

        midcount = (upcount + lwcount) / 2
        midthr = (upthr + lwthr) / 2
        # repeats loop until diff falls inside tolerance (-tol <= diff <= tol)
        while True:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break
            # if tol is not met up to 20 iters, then relax tol by 10%
            iterno += 1
            if iterno > 20:
                tol *= 1.1

            if diff > tol:
                midcount = (upcount + midcount) / 2
                # upper and mid activities
                midthr = (upthr + midthr) / 2
                # ... and thresholds
            elif diff < -tol:
                # then new bounds are...
                midcount = (midcount - lwcount) / 2
                # middle and lower activities
                midthr = (midthr + lwthr) / 2
                # ... and thresholds
        # since tolerance has been satisfied, midcount is selected as
        # interpolated value with tol [dB] tolerance
        asl_ms_log = midcount
        cc = midthr
        return asl_ms_log, cc


class Clipping(object):

    def __init__(self, clip_factors = [0.3, 0.4, 0.5]):
        self.clip_factors = clip_factors

    def __call__(self, wav):
        cf = np.random.choice(self.clip_factors, 1)
        clip = np.maximum(wav, cf * np.min(wav))
        clip = np.minimum(wav, cf * np.max(wav))
        return torch.FloatTensor(clip)
