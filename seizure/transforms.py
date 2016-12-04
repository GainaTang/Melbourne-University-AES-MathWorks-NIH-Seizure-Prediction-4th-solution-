from __future__ import division
import numpy as np
from scipy import signal
from scipy.signal import resample, hann
from sklearn import preprocessing
from math import *
from scipy.stats import skew, kurtosis



# onpptional modules for trying out different transforms
try:
    import pywt
except ImportError, e:
    pass

try:
    from scikits.talkbox.features import mfcc
except ImportError, e:
    pass


# NOTE(mike): All transforms take in data of the shape (NUM_CHANNELS, NUM_FEATURES)
# Although some have been written work on the last axis and may work on any-dimension data.


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """
    def get_name(self):
        return "fft"

    def apply(self, data):
        axis = data.ndim - 1
        return np.fft.rfft(data, axis=axis)


class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d-%d" % (self.start, self.end)

    def apply(self, data):
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]


class LPF:
    """
    Low-pass filter using FIR window
    """
    def __init__(self, f):
        self.f = f

    def get_name(self):
        return 'lpf%d' % self.f

    def apply(self, data):
        nyq = self.f / 2.0
        cutoff = min(self.f, nyq-1)
        h = signal.firwin(numtaps=101, cutoff=cutoff, nyq=nyq)

        # data[i][ch][dim0]
        for i in range(len(data)):
            data_point = data[i]
            for j in range(len(data_point)):
                data_point[j] = signal.lfilter(h, 1.0, data_point[j])

        return data


class MFCC:
    """
    Mel-frequency cepstrum coefficients
    """
    def get_name(self):
        return "mfcc"

    def apply(self, data):
        all_ceps = []
        for ch in data:
            ceps, mspec, spec = mfcc(ch)
            all_ceps.append(ceps.ravel())

        return np.array(all_ceps)


class Magnitude:
    """
    Take magnitudes of Complex data
    """
    def get_name(self):
        return "mag"

    def apply(self, data):
        return np.absolute(data)


class MagnitudeAndPhase:
    """
    Take the magnitudes and phases of complex data and append them together.
    """
    def get_name(self):
        return "magphase"

    def apply(self, data):
        magnitudes = np.absolute(data)
        phases = np.angle(data)
        return np.concatenate((magnitudes, phases), axis=1)


class Log10:
    """
    Apply Log10
    """
    def get_name(self):
        return "log10"

    def apply(self, data):
        # 10.0 * log10(re * re + im * im)
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)


class Stats:
    """
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
    """
    def get_name(self):
        return "stats"

    def apply(self, data):
        # data[ch][dim]
        shape = data.shape
        out = np.empty((shape[0], 3))
        for i in range(len(data)):
            ch_data = data[i]
            ch_data = data[i] - np.mean(ch_data)
            outi = out[i]
            outi[0] = np.std(ch_data)
            outi[1] = np.min(ch_data)
            outi[2] = np.max(ch_data)

#         return out
        return np.concatenate(out, axis=0)

class Resample:
    """
    Resample time-series data.
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data


class ResampleHanning:
    """
    Resample time-series data using a Hanning window
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%dhanning" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        out = resample(data, self.f, axis=axis, window=hann(M=data.shape[axis]))
        return out


class DaubWaveletStats:
    """
    Daubechies wavelet coefficients. For each block of co-efficients
    take (mean, std, min, max)
    """
    def __init__(self, n):
        self.n = n

    def get_name(self):
        return "dwtdb%dstats" % self.n

    def apply(self, data):
        # data[ch][dim0]
        shape = data.shape
        out = np.empty((shape[0], 4 * (self.n * 2 + 1)), dtype=np.float64)

        def set_stats(outi, x, offset):
            outi[offset*4] = np.mean(x)
            outi[offset*4+1] = np.std(x)
            outi[offset*4+2] = np.min(x)
            outi[offset*4+3] = np.max(x)

        for i in range(len(data)):
            outi = out[i]
            new_data = pywt.wavedec(data[i], 'db%d' % self.n, level=self.n*2)
            for i, x in enumerate(new_data):
                set_stats(outi, x, i)

        return out


class UnitScale:
    """
    Scale across the last axis.
    """
    def get_name(self):
        return 'unit-scale'

    def apply(self, data):
        return preprocessing.scale(data, axis=data.ndim-1)


class UnitScaleFeat:
    """
    Scale across the first axis, i.e. scale each feature.
    """
    def get_name(self):
        return 'unit-scale-feat'

    def apply(self, data):
        return preprocessing.scale(data, axis=0)


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply(self, data):
        return np.corrcoef(data)


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigenvalues'

    def apply(self, data):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


class OverlappingFFTDeltas:
    """
    Calculate overlapping FFT windows. The time window will be split up into num_parts,
    and parts_per_window determines how many parts form an FFT segment.

    e.g. num_parts=4 and parts_per_windows=2 indicates 3 segments
    parts = [0, 1, 2, 3]
    segment0 = parts[0:1]
    segment1 = parts[1:2]
    segment2 = parts[2:3]

    Then the features used are (segment2-segment1, segment1-segment0)

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "overlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data):
        axis = data.ndim - 1

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            combined_parts = parts[i:i+self.parts_per_window]
            if self.parts_per_window > 1:
                d = np.concatenate(combined_parts, axis=axis)
            else:
                d = combined_parts
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        diffs = []
        for i in range(1, len(partials)):
            diffs.append(partials[i] - partials[i-1])

        return np.concatenate(diffs, axis=axis)


class FFTWithOverlappingFFTDeltas:
    """
    As above but appends the whole FFT to the overlapping data.

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "fftwithoverlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data):
        axis = data.ndim - 1

        full_fft = np.fft.rfft(data, axis=axis)
        full_fft = Magnitude().apply(full_fft)
        full_fft = Log10().apply(full_fft)

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            d = np.concatenate(parts[i:i+self.parts_per_window], axis=axis)
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        out = [full_fft]
        for i in range(1, len(partials)):
            out.append(partials[i] - partials[i-1])

        return np.concatenate(out, axis=axis)


class FreqCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, start, end, scale_option, resample_size, with_fft=False, with_corr=True, with_eigen=True):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        self.resample_size = resample_size
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'freq-correlation-%d-%d-%s-%s%s' % (self.start, self.end, 'withfft' if self.with_fft else 'nofft',
                                                   self.scale_option, selection_str)

    def apply(self, data):
        data1 = FFT().apply(data)
        # data2 = FFT().apply(data[:,:2500])
        # plt.plot(Magnitude().apply(data1).transpose())
        # fig = plt.figure()
        # plt.plot(Magnitude().apply(data2).transpose())
        if data1.shape[1]<(self.start+self.resample_size):
            data1 = Magnitude().apply(data1)
            data1 = Log10().apply(data1)
        else:

            data1 = Slice(self.start, self.end).apply(data1)
            data1 = Magnitude().apply(data1)
            # print data1.shape
            # fig = plt.figure()
            # plt.plot(data1.transpose())
            axis = data1.ndim - 1
            data1 = resample(data1, num = self.resample_size, axis = axis)     #resample to resample_size
            data1 = Log10().apply(data1)

        data2 = data1
        if self.scale_option == 'usf':
            data2 = UnitScaleFeat().apply(data2)
        elif self.scale_option == 'us':
            data2 = UnitScale().apply(data2)

        data2 = CorrelationMatrix().apply(data2)

        if self.with_eigen:
            w = Eigenvalues().apply(data2)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)

class TimeCorrelation:
    """
    Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.

    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, max_hz, scale_option, with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'time-correlation-r%d-%s%s' % (self.max_hz, self.scale_option, selection_str)

    def apply(self, data):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001

        data1 = data
        if self.scale_option == 'usf':
            data1 = UnitScaleFeat().apply(data1)
        elif self.scale_option == 'us':
            data1 = UnitScale().apply(data1)

        data1 = CorrelationMatrix().apply(data1)

        if self.with_eigen:
            w = Eigenvalues().apply(data1)

        out = []
        if self.with_corr:
            data1 = upper_right_triangle(data1)
            out.append(data1)
        if self.with_eigen:
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)

class GetFeature:
    def __init__(self, start, end, max_hz, resample_size, scale_option,
                 onlyfd_dfa,with_dfa,with_six,with_dy,with_mc,with_time_corr,
                 with_equal_freq,smooth,smooth_Hz,power_edge,with_square,with_log,
                 with_sqrt,splitsize,calibrate):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.resample_size = resample_size
        self.scale_option = scale_option
        self.onlyfd_dfa = onlyfd_dfa
        self.with_dfa = with_dfa
        self.with_six = with_six
        self.with_dy = with_dy
        self.with_mc = with_mc
        self.with_time_corr=with_time_corr
        self.with_equal_freq=with_equal_freq
        self.smooth = smooth
        self.smooth_Hz = smooth_Hz
        self.power_edge = power_edge
        self.with_square = with_square
        self.with_log = with_log
        self.with_sqrt = with_sqrt
        self.splitsize = splitsize
        self.calibrate = calibrate
    def get_name(self):
        selections = []
        selections.append('bin_size_%d' % self.resample_size)
        selections.append('splitsize_%d'%self.splitsize)

        selections.append('%d'%self.start)
        selections.append('%d'%self.end)
        if self.calibrate:
            selections.append('calibrated')
        if self.smooth:
            selections.append('resampled_%dHz'%(self.smooth_Hz))
        # if self.with_dfa:
        #     selections.append('DFA')
        if self.with_six:
            selections.append('six')
        if self.with_dy:
            selections.append('dy')
        if self.with_mc:
            selections.append('mobility_complexsity')
        if self.with_time_corr:
            selections.append('time_corr')
        if self.with_equal_freq:
            selections.append('equal_freq')
        if self.with_square:
            selections.append('square')
        if self.with_sqrt:
            selections.append('sqrt')
        if self.with_log:
            selections.append('log')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'feature-%s-power_edge-%d' % (selection_str,self.power_edge)

    def apply(self, data):
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        data = data.transpose()
        data = np.pad(data,((0,self.splitsize -data.shape[0]),(0,0)),'constant')
        nt, nc =data.shape
        fs = 400
        feat = []
        if self.smooth:
            data=resample(data,int(nt/fs*self.smooth_Hz))
            nt, nc = data.shape
            fs = self.smooth_Hz
            lvl = np.array([0.1, 4, 8, 12, 30, 70])
        if self.with_six or self.with_dy:
            D = np.absolute(np.fft.rfft(data, axis=0))
            D[0, :] = 0  # set the DC component to zero
            for i in range(nc):
                D[:,i] /= D[:,i].sum()  # Normalize each channel
            coorD = np.corrcoef(D.transpose())
            w = Eigenvalues().apply(coorD)
            tfreq = self.power_edge
            ppow = 0.5
            # top_freq = int(round(nt / sfreq * tfreq)) + 1
            top = int(round(nt / fs * tfreq))
            spedge = np.cumsum(D[:top, :], axis=0)
            spedge = np.argmin(np.abs(spedge - (spedge.max(axis=0) * ppow)), axis=0)
            spedge = spedge / top * tfreq
            feat.append(w)
            feat.append(spedge.ravel())
        if self.with_six:
            lseg = np.round(nt / fs * lvl).astype('int')
            sixspect = np.zeros((len(lvl) - 1, nc))
            for j in range(len(sixspect)):
                sixspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)
            spentropy = -1 * np.sum(np.multiply(sixspect, np.log(sixspect)), axis=0)
            feat.append(sixspect.ravel())
            feat.append(spentropy.ravel())
        if self.with_dy:
            ldat = int(floor(nt / 2.0))
            no_levels = int(floor(log(ldat, 2.0)))
            dspect = np.zeros((no_levels, nc))
            for j in range(no_levels - 1, -1, -1):
                dspect[j, :] = 2 * np.sum(D[int(floor(ldat / 2.0)):ldat, :], axis=0)
                ldat = int(floor(ldat / 2.0))
            spentropyDyd = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)
            feat.append(dspect.ravel())
            feat.append(spentropyDyd.ravel())
        if self.with_mc:
            mobility = np.divide(
                np.std(np.diff(data, axis=0)),
                np.std(data, axis=0))
            complexity = np.divide(np.divide(
                # std of second derivative for each channel
                np.std(np.diff(np.diff(data, axis=0), axis=0), axis=0),
                # std of second derivative for each channel
                np.std(np.diff(data, axis=0), axis=0))
                , mobility)
            feat.append(mobility)
            feat.append(complexity)
        if self.with_time_corr:
            data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data.transpose())
            feat.append(data1)
        if self.with_equal_freq:
            data2 = FreqCorrelation(self.start, self.end, self.scale_option, self.resample_size, with_fft=True,
                                    with_corr=True).apply(data.transpose())
            feat.append(data2)
        if self.onlyfd_dfa:
            fd = np.zeros((2, nc))
            for j in range(nc):
                fd[0, j] = pyeeg.pfd(data[:, j])
                fd[1, j] = pyeeg.hfd(data[:, j], 3)
            DFA = np.zeros(nc)
            for j in range(nc):
                DFA[j] = pyeeg.dfa(data[:, j])
                feat=np.concatenate((
                    fd.ravel(),
                    DFA.ravel(),
                    np.sqrt(DFA).ravel(),
                    np.square(DFA.ravel()),
                    np.sqrt(fd).ravel(),
                    np.square(fd).ravel(),
                ))
        if self.with_square or self.with_log or self.with_sqrt:
            tmp = np.concatenate(feat, axis=0)
            tmp = np.absolute(tmp)
            if self.with_square:
                feat.append(np.square(tmp))
            if self.with_log:
                feat.append(np.log(tmp))
            if self.with_sqrt:
                feat.append(np.sqrt(tmp))
        return np.concatenate(feat, axis=0)



class TimeFreqCorrelation:
    """
    Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert scale_option in ('us', 'usf', 'none')

    def get_name(self):
        return 'time-freq-correlation-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(self.start, self.end, self.scale_option).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim-1)


class FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz, resample_size, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.resample_size = resample_size
        self.scale_option = scale_option

    def get_name(self):
        return 'fft-with-time-freq-corr-%d-%d-r%d-s%d-%s' % (self.start, self.end, self.max_hz, self.resample_size, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(self.start, self.end, self.scale_option, self.resample_size, with_fft=True, with_corr=True).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim-1)

class only_FD_DFA:
    def __init__(self,onlyfd_dfa):
        self.onlyfd_dfa = onlyfd_dfa
    def get_name(self):
        return 'onyl_FD_DFA'
    def apply(self,data):
        axis = data.ndim - 1
        data0 = GetFeature(self.onlyfd_dfa).apply(data)
        return data0