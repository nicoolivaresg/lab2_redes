##### Importación de librería y funciones #####
import numpy as np
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, ifft
from scipy import signal
from os.path import isfile
import matplotlib.pyplot as plt
import matplotlib
import re


####### Constantes y variables globales #######
TITLE_SPEC = "Espectrograma de la señal"
TITLE_FFT = "Gráfico del audio en el dominio de la frecuencia"
TITLE_IFFT = "Gráfico del audio en el tiempo usando la transformada inversa"

XLABEL_SPEC = "Tiempo [s]"
XLABEL_FFT = "Frecuencia [Hz]"
XLABEL_IFFT = "Tiempo [s]"

YLABEL_SPEC = "Frecuencia [Hz]"
YLABEL_FFT = "Amplitud [dB]"
YLABEL_IFFT = "Amplitud"

BUTTERWORTH = " - filtrada por butterworth"
CHEBYSHEV = " - filtrada por chebyshev"
CHEBYSHEV_INVERTED = " - filtrada por chebyshev invertido"

HIGHPASS = " (highpass)"
LOWPASS = " (lowpass)"
BANDPASS = " (bandpass)"

FFT = "-fft"
IFFT = "-ifft"
SPEC = "-spectrogram"

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

CMAP = 'nipy_spectral'
NOVERLAP = 250
########## Definición de Funciones ############

# Construye un titulo de un grafico de acuerdo al tipo de filtro que se utiliza y el nombre
# del filtro utilizado.
# Entrada:
#	title 		- Es una string que posee el comienzo del titulo, se utilizan las constantes
#				  definidas que comienzan con el prefijo TITLE.
# 	filter_name	- Nombre del filtro que se utiliza en el grafico, se utilizan las constantes
#				  definidas que tiene el nombre del filtro (p.ej: BUTTERWORTH)
#	filter_type	- Define si el tipo de filtro fue high (pasaalto), low (pasabajo) o 
#				  band (pasabanda).
# Salida:
#	title_full	- El titulo completo del grafico que sera creado.
def build_graph_title(title, filter_name, filter_type):
	if filter_type == "high":
		return title + filter_name + HIGHPASS
	elif filter_type == "low":
		return title + filter_name + LOWPASS
	elif filter_type == "band":
		return title + filter_name + BANDPASS

# Funcion que grafica los datos en ydata y xdata, y escribe los nombres del eje x, eje y,
# y el titulo de una figura. Esta figura la guarda en un archivo con el nombre filename.
# Entrada:
#	filename	- Nombre del archivo en donde se guarda la figura.
#	title		- Titulo de la figura.
#	ylabel		- Etiqueta del eje y.
#	xlabel		- Etiqueta del eje x.
#	ydata		- Datos del eje y.
#	xdata		- Datos del eje X, por defecto es un arreglo vacío que luego se cambia por un
#				  arreglo desde 0 hasta largo de ydata - 1
#	color		- Color de la figura en el grafico, por defecto es azul (blue).
def graficar(filename, title, ylabel, xlabel, ydata, xdata=np.array([]), color='b'):
	if xdata.size == 0:
		xdata = np.arange(len(ydata))

	plt.figure(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT), dpi=DPI)
	plt.plot(xdata, ydata, color)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(GRAPH_DIR + filename + ".png", bbox_inches='tight')
	plt.clf()

# Crea un espectrograma de la funcion que se encuentra en 'signal' con frecuencia de muestreo
# 'frequency' con su barra de colores al lado, esta figura es guardada en un archivo de 
# extension .png
# Entrada:
#	filename	- Nombre del archivo en donde se guarda la figura.
#	title		- Titulo de la figura.
#	ylabel		- Etiqueta del eje y.
#	xlabel		- Etiqueta del eje x.
#	signal		- Datos de la señal de la que se quiere obtener el espectrograma.
#	frequency	- Frecuencia de muestreo de la señal.
def audio_spectrogram(filename, title, ylabel, xlabel, signal, frequency):
	(Pxx, freqs, bins, im) = plt.specgram(signal,Fs=frequency, cmap=CMAP, noverlap=NOVERLAP)
	plt.colorbar(im)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(GRAPH_DIR + filename + ".png", bbox_inches='tight')
	plt.clf()

# Funcion que usa el read de scipy.io.wavfile para abrir un archivo .wav y obtener la
# frecuencia y la informacion del sonido, esta funcion ademas obtiene un vector de tiempo
# dependiendo de la canidad de datos y la frecuencia del audio.
# Entrada:
# 	filename	- Nombre del archivo por abrir.
# Salida:
#	frecuencia	- Numero entero con la frecuencia de muestreo del audio en [Hz].
#	datos		- Arreglo numpy con los datos obtenidos por la lectura del archivo.
#	tiempos		- Arreglo de floats con el tiempo en segundos para cada dato de 'datos'.
def load_wav_audio(filename):
	frecuencia, datos = read(filename)
	n = len(datos)
	Ts = n / frecuencia; # Intervalo de tiempo
	tiempos = np.linspace(0, Ts, n) # Tiempo en segundos para cada dato de 'datos'
	return (frecuencia, datos, tiempos)

# Funcion que usa el write de scipy.io.wavfile para escribir un archivo .wav de acuerdo a 
# los parametros entregados a esta funcion.
# Entrada:
#	filename	- Nombre del archivo .wav a crear (Ejemplo "salida.wav").
#	frequency	- Frecuencia de muestreo del audio.
#	signal		- La señal en el dominio del tiempo.
def save_wav_audio(filename, frequency, fsignal):
	write(AUDIO_DIR + filename + ".wav", frequency, fsignal.astype('int16'))

# Funcion que hace uso de la tranformada de fourier, fft() y fftfreq(), para obtener la
# secuencia de valores de los datos obtenidos del audio y para obtener las frecuencias
# de muestreo (que depende de la frecuencia del audio y del largo del audio) respectivamente.
# Entrada:
#	data		- Datos obtenidos al leer el archivo de audio con scipy o con load_wav_audio().
#	frequency	- Numero entero con la frecuencia de muestreo del audio en [Hz].
# Salida:
#	fftValues	- Transformada de fourier normalizada para los valores en data.
#	fftSamples	- Frecuencias de muestreo que dependen del largo del arreglo data y la frequency.
def fourier_transform(data, frequency):
	n = len(data)
	Ts = n / frequency
	fftValues = fft(data) / n # Computacion y normalizacion
	fftSamples = np.fft.fftfreq(n, 1/frequency)

	return (fftValues, fftSamples)

# Funcion que retorna la transformada inversa de fourier de un arreglo de valores.
# Entrada:
#	fftValues	- Transformada de fourier normalizada.
# Salida:
#	fftValuesInverse - Transformada de fourier inversa denormalizada, se puede utilizar para escribir
#					   un audio utilizando la funcion write de scipy.io.wavfile
def fourier_inverse(fftValues):
	return ifft(fftValues)*len(fftValues)

# Transforma una frecuencia (targetFreq) a su equivalente en frecuencia Nyquist, de acuerdo 
# a la frecuencia de muestreo de la señal, esta frecuencia es por defecto 44100.
# Entrada:
#	targetFreq	- Frecuencia de la que se quiere obtener su equivalente en Nyquist frequency.
#	sampleRate	- Frecuencia de muestreo de la señal (por defecto es 44100).
def get_wn(targetFreq, sampleRate=44100):
	if(targetFreq > sampleRate):
		return 1
	else:
		return np.divide(targetFreq, sampleRate / 2.0)

# PARTE MAS IMPORTANTE, ACA SE MODIFICAN LOS NUMEROS DE LOS FILTROS
def get_filter_limits(filterord_func, filter_type):
	N = 0 
	Wn = 0
	if filter_type == 'low':
		N, Wn = filterord_func(0.2, 0.5, 3, 40)
	elif filter_type == 'high':
		N, Wn = filterord_func(0.5, 0.2, 3, 40)
	elif filter_type == 'band':
		N, Wn = filterord_func([0.2, 0.7], [0.1, 0.8], 3, 40)

	print(N, Wn)
	return (N, Wn)

def butterworth(filter_type):
	(N, Wn) = get_filter_limits(signal.buttord, filter_type)
	b, a = signal.butter(N, Wn, filter_type)
	return (b, a)

def chebyshev(filter_type):
	(N, Wn) = get_filter_limits(signal.cheb1ord, filter_type)
	b, a = signal.cheby1(N, 5, Wn, filter_type)
	return (b, a)

def chebyshev_inverted(filter_type):
	(N, Wn) = get_filter_limits(signal.cheb2ord, filter_type)
	b, a = signal.cheby2(N, 5, Wn, filter_type)
	return (b, a)

def filter(fsignal, filter_func, filter_type='low'):
	(b, a) = filter_func(filter_type)
	z = signal.filtfilt(b, a, fsignal)
	return z

def filter_pass(filename, data, frequency, times, filter_type):
	dataButter = filter(data, butterworth, filter_type)
	dataCheby = filter(data, chebyshev, filter_type)
	dataChebyInverted = filter(data, chebyshev_inverted, filter_type)
	fftButter, fftSamples = fourier_transform(dataButter, frequency)
	fftCheby, _ = fourier_transform(dataCheby, frequency)
	fftChebyInverted, _ = fourier_transform(dataChebyInverted, frequency)

	filenameButter = filename + "-butter-" + filter_type
	filenameCheby = filename + "-cheby-" + filter_type
	filenameChebyInverted = filename + "-cheby_inverted-" + filter_type

	# build graph titles
	titleSpecButter = build_graph_title(TITLE_SPEC, BUTTERWORTH, filter_type)
	titleSpecCheby = build_graph_title(TITLE_SPEC, CHEBYSHEV, filter_type)
	titleSpecChebyInv = build_graph_title(TITLE_SPEC, CHEBYSHEV_INVERTED, filter_type)
	titleFFTButter = build_graph_title(TITLE_FFT, BUTTERWORTH, filter_type)
	titleFFTCheby = build_graph_title(TITLE_FFT, CHEBYSHEV, filter_type)
	titleFFTChebyInv = build_graph_title(TITLE_FFT, CHEBYSHEV_INVERTED, filter_type)
	titleIFFTButter = build_graph_title(TITLE_IFFT, BUTTERWORTH, filter_type)
	titleIFFTCheby = build_graph_title(TITLE_IFFT, CHEBYSHEV, filter_type)
	titleIFFTChebyInv = build_graph_title(TITLE_IFFT, CHEBYSHEV_INVERTED, filter_type)

	# get spectrograms
	audio_spectrogram(filenameButter + SPEC, titleSpecButter, YLABEL_SPEC, XLABEL_SPEC, dataButter, frequency)
	audio_spectrogram(filenameCheby + SPEC, titleSpecCheby, YLABEL_SPEC, XLABEL_SPEC, dataCheby, frequency)
	audio_spectrogram(filenameChebyInverted + SPEC, titleSpecChebyInv, YLABEL_SPEC, XLABEL_SPEC, dataChebyInverted, frequency)

	# inverted fourier graphs (basically signals) plus fourier transform graphs
	graficar(filenameButter + FFT, titleFFTButter, YLABEL_FFT, XLABEL_FFT, abs(fftButter), fftSamples)
	graficar(filenameButter + IFFT, titleIFFTButter, YLABEL_IFFT, XLABEL_IFFT, fourier_inverse(fftButter), times)
	graficar(filenameCheby + FFT,titleFFTCheby, YLABEL_FFT, XLABEL_FFT, abs(fftCheby), fftSamples)
	graficar(filenameCheby + IFFT,titleIFFTCheby, YLABEL_IFFT, XLABEL_IFFT, fourier_inverse(fftCheby), times)
	graficar(filenameChebyInverted + FFT,titleFFTChebyInv, YLABEL_FFT, XLABEL_FFT, abs(fftChebyInverted), fftSamples)
	graficar(filenameChebyInverted + IFFT,titleIFFTChebyInv, YLABEL_IFFT, XLABEL_IFFT, fourier_inverse(fftChebyInverted), times)

	# save filtered signals back to audio
	save_wav_audio(filenameButter, frequency, fourier_inverse(fftButter))
	save_wav_audio(filenameCheby, frequency, fourier_inverse(fftCheby))
	save_wav_audio(filenameChebyInverted, frequency, fourier_inverse(fftChebyInverted))


# Abre un archivo .wav y ...
# 
# Entrada:
#	filename	- Nombre del archivo con extension '.wav' que se quiere procesar.
def process_file(filename):
	"""
	global figureCounter
	figureCounter += 1
	"""
	filenameNoExtension = filename[:len(filename)-4]
	frecuencia, datos, tiempos = load_wav_audio(filename)
	#fftNormalizada, fftSamples = fourier_transform(datos, frecuencia)

	audio_spectrogram(filenameNoExtension + SPEC, TITLE_SPEC, XLABEL_SPEC, YLABEL_SPEC, datos, frecuencia)
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'low')
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'high')
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'band')

################ Bloque Main ##################
#process_file("lab1-1.wav")
process_file("lab1-2.wav")