# Created by Devin Gordon on 01/05/23 at 15:23
# Updated by Devin Gordon on 06/25/23 at 13:27
""" PRIMARY GOAL: allow this to run and update in real time """
""" SECONDARY GOAL: use performFFT() to distinguish between notes """
""" TERTIARY GOAL: display both stereo channels instead of averaging """

import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from colorsys import hsv_to_rgb
import wave
import sys
import pyaudio
from scipy.io import wavfile
import time

# returns FFT index associated with magnitude peaks in FFT
def locatePeaks(arr):
    """ Returns all peaks within 65% of max amplitude"""
    # may need to find percentage w.r.t highest and lowest value
    # instead of highest and zero
    norm = max(abs(max(arr)), abs(min(arr)))
    peaks = []
    for x in range(len(arr)):
        if abs(arr[x]) > 0.65*norm: peaks.append(x)
    return peaks


# Reads in audio sample of length RECORD_SECONDS into a .wav file
# Returns: sample rate and audio data as a tuple
def generateAudioSample():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == 'darwin' else 2
    RATE = 44100
    RECORD_SECONDS = 5

    with wave.open('output.wav', 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print('Recording...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK, exception_on_overflow=False))
        print('Done')

        stream.close()
        p.terminate()

        return wavfile.read("output.wav")


# Input: tuple containing sample rate and data collected from .wav file
# Output: tuple containing time values and frequency data of FFT of input
def performFFT(info):
    # SECONDARY GOAL
    """ Add FFT sizing functionality """
    sampleRate = info[0]
    data = info[1]
    data = data.astype("float64")
    t = np.linspace(0., data.shape[0]/sampleRate, data.shape[0])
    data *= np.hamming(data.shape[0]) # taper function off at ends to prevent unwanted sharp changes in FFT
    S = fftshift(fft(data)) # fftshift shifts function as to be centered at 0
    SMag = np.abs(S)
    SAngle = np.angle(S)

    f = np.linspace(sampleRate/(-2), sampleRate/2, data.shape[0])

    # peaks = [f[peak] for peak in locatePeaks(SMag)]
    # print("Peak Frequencies: "+str(peaks))

    plt.subplot(211)
    plt.plot(t, data)
    plt.title("Audio Sample")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(212)
    plt.stem(f, SMag)
    plt.title("FFT of Input Function")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()
    plt.close()

    return t, SMag


# Input: data from .wav audio file, read using SciPy
# Output: color-mapped spectrogram
# Works for both single-channel (mono) and double-channel (stereo) audio
def createSpectrogram(info, pictureTitle="", colormap='viridis'):
    start = time.time()
    fftSize = 2**12 # 4096
    sampleRate = info[0]
    data = info[1]
    data = data.astype("float64")

    # TERTIARY GOAL
    if len(data.shape) > 1:
        avgData = [(x[0]+x[1])/2 for x in data]
        data = np.array(avgData)

    num_rows = len(data)//fftSize
    spectrogram = np.zeros((num_rows, fftSize))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(fftshift(fft(data[i*fftSize:(i+1)*fftSize])))**2)
    # removes mirrored frequencies
    spectrogram = spectrogram[:,fftSize//2:]

    plt.imshow(spectrogram, aspect='auto', extent = [0, sampleRate/2/1000, 0, len(data)/sampleRate], cmap=colormap)
    # more color maps at 'https://matplotlib.org/stable/tutorials/colors/colormaps.html'

    if not pictureTitle:
        plt.title("Audio Sample Spectrogram")
        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Time (s)")
        plt.xticks(ticks=[x for x in range(int(sampleRate/2/1000)-1)])
    else:
        plt.title(pictureTitle)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    end = time.time()
    print("Execution Time: "+f'{end-start: .3f}'+" seconds")
    plt.show()
    plt.close()


np.seterr(divide='ignore')
# performFFT(generateAudioSample())

colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
          'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 
          'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
          'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
          'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 
          'ocean', 'gist_earth', 'terrain',
          'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
          'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
          'turbo', 'nipy_spectral', 'gist_ncar']
colorError = "Colormap does not exist. Call 'python spectrogram.py -h' or 'python spectrogram.py --help' for options."


# takes in command line inputs
if (len(sys.argv) == 1): createSpectrogram(generateAudioSample())
else:
    if len(sys.argv) == 3:
        if sys.argv[1] == "-f":
            createSpectrogram(wavfile.read(str(sys.argv[2])))
        elif sys.argv[1] == "-t":
            createSpectrogram(generateAudioSample(), pictureTitle=sys.argv[2])
        elif sys.argv[1] == "-c":
            createSpectrogram(generateAudioSample(), colormap=sys.argv[2])
    elif len(sys.argv) == 4:
        if sys.argv[1] == "-ft":
            createSpectrogram(wavfile.read(str(sys.argv[2])), str(sys.argv[3]))
        if sys.argv[1] == "-fc":
            if sys.argv[3] not in colors: raise Exception(colorError)
            createSpectrogram(wavfile.read(str(sys.argv[2])), '', str(sys.argv[3]))
        if sys.argv[1] == "-tc":
            if sys.argv[3] not in colors: raise Exception(colorError)
            createSpectrogram(generateAudioSample(), str(sys.argv[2]), str(sys.argv[3]))
    elif len(sys.argv) == 5 and sys.argv[1][0] == ("-") and 0 not in [x in sys.argv[1] for x in ["f","t","c"]]:
        if sys.argv[4] not in colors: raise Exception(colorError)
        createSpectrogram(wavfile.read(str(sys.argv[2])), str(sys.argv[3]), str(sys.argv[4]))
    elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("\nSyntax:"
            +"\n\tpython spectrogram.py [-ftc] [filename] [title] [colormap]")
        print("\nFlags:"
            +"\n\tf: specify filename to be read into program (reads from microphone if not used)"
            +"\n\tt: assign title to graph created on output"
            +"\n\tc: pick colormap for graph created on output (default: viridis)")
        print("\nColormaps:"
            +"\n"+str(colors)+"\n")
    else: raise Exception("Incorrect flag syntax. Call 'python spectrogram.py -h' or 'python spectrogram.py --help' for syntax.")
