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
TITLE1 = "Gráfico de audio en el tiempo"
TITLE2 = "Gráfico del audio en el dominio de la frecuencia"
TITLE3 = "Gráfico del audio en el tiempo usando la transformada inversa"
TITLE4 = "Gráfico en el dominio de la frecuencia con fft truncada en un 15%"
TITLE5 = "Gráfico del audio en el tiempo usando la transformada truncada inversa"

XLABEL1 = "Tiempo [s]"
XLABEL2 = "Frecuencia [Hz]"
XLABEL3 = "Tiempo [s]"
XLABEL4 = "Frecuencia [Hz]"
XLABEL5 = "Tiempo [s]"

YLABEL1 = "Amplitud"
YLABEL2 = "Amplitud [dB]"
YLABEL3 = "Amplitud"
YLABEL4 = "Amplitud [dB]"
YLABEL5 = "Amplitud"

XLABEL = "Tiempo [s]"
YLABEL = "Frecuencia [Hz]"
TITLE = "Espectrograma"

FFT = "-fft"
IFFT = "-ifft"
SPEC = "-spectrogram"

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

figureCounter = 0

CMAP = 'nipy_spectral'
NOVERLAP = 250
########## Definición de Funciones ############
"""
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
	plt.savefig(filename, bbox_inches='tight')
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
def openWavFile(filename):
	frecuencia, datos = read(filename)
	n = len(datos)
	Ts = n / frecuencia; # Intervalo de tiempo
	tiempos = np.linspace(0, Ts, n) # Tiempo en segundos para cada dato de 'datos'
	return (frecuencia, datos, tiempos)


# Funcion que hace uso de la tranformada de fourier, fft() y fftfreq(), para obtener la
# secuencia de valores de los datos obtenidos del audio y para obtener las frecuencias
# de muestreo (que depende de la frecuencia del audio y del largo del audio) respectivamente,
# Entrada:
#	data		- Datos obtenidos al leer el archivo de audio con scipy.
#	frequency	- Numero entero con la frecuencia de muestreo del audio en [Hz].
# Salida:
#	fftValues	- Transformada de fourier normalizada para los valores en data.
#	fftSamples	- Frecuencias de muestreo que dependen del largo del arreglo data y la frequency.
def fourierTransform(data, frequency):
	n = len(data)
	Ts = n / frequency
	fftValues = fft(data) / n # Computacion y normalizacion
	fftSamples = np.fft.fftfreq(n, 1/frequency)

	return (fftValues, fftSamples)
# Funcion que trunca en 15% hacia la izquierda y derecha de la frecuencia con mayor amplitud en
# un arreglo de valores con la transformada de fourier de un audio.
# Entrada:
#	fftValues	- Transformada de fourier normalizada para los valores obtenido al leer el archivo
#				  de audio con scipy.
# Salida:
#	fftTruncada	- Transformada de fourier normalizada con los valores que se encuentran fuera del
#				  margen de 15% seteados en 0.
def truncateFft(fftValues):
	n = len(fftValues)
	maxFreqI = 0
	maxFreq = 0
	for i in range(n):
		if fftValues[i] > maxFreq:
			maxFreq = fftValues[i]
			maxFreqI = i


	leftI = int(maxFreqI - (n*0.15))
	rightI = int(maxFreqI + (n*0.15))

	fftTruncada = np.array([0]*n)

	if (leftI < 0):
		for i in range(rightI):
			fftTruncada[i] = fftValues[i]

		for j in range(n+leftI, n):
			fftTruncada[j] = fftValues[j]
	elif (rightI >= n):
		for i in range(rightI-n):
			fftTruncada[i] = fftValues[i]

		for j in range(leftI, n):
			fftTruncada[j] = fftValues[j]
	else:
		for i in range(leftI, rightI):
			fftTruncada[i] = fftValues[i]

	return fftTruncada

# Funcion que retorna la transformada inversa de fourier de un arreglo de valores, se utiliza para
# devolver la tranformada y la transformada truncada a una señal de audio.
# Entrada:
#	fftValues	- Transformada de fourier normalizada para los valores obtenido al leer el archivo
#				  de audio con scipy.
# Salida:
#	fftValuesInverse - Transformada de fourier inversa denormalizada, se puede utilizar para escribir
#					   un audio utilizando la funcion write de scipy.io.wavfile
def fourierInverse(fftValues):
	return ifft(fftValues)*len(fftValues)

# Abre un archivo .wav y guarda los siguientes graficos en archivos .png:
# 1- Gráfico de audio en el tiempo
# 2- Gráfico del audio en el dominio de la frecuencia
# 3- Gráfico del audio en el tiempo usando la transformada inversa
# 4- Gráfico en el dominio de la frecuencia con fft truncada en un 15%
# 5- Gráfico del audio en el tiempo usando la transformada truncada inversa
# 
# Tambien usa la inversa truncada para crear y guardar el archivo de audio resultante de
# esta señal.
# 
# Entrada:
#	filename	- Nombre del archivo con extension '.wav' que se quiere procesar.
def processFile(filename):
	global figureCounter
	figureCounter += 1
	frecuencia, datos, times = openWavFile(filename)
	fftNormalizada, fftSamples = fourierTransform(datos, frecuencia)
	fftNormalizadaInversa = fourierInverse(fftNormalizada)

	#f, t, sxx = spectrogram(datos,frecuencia,nperseg=256)
	#CHAO plt.pcolormesh(t,f,sxx, cmap=plt.cm.get_cmap('cubehelix'))

	(Pxx, freqs, bins, im) = plt.specgram(datos,Fs=frecuencia, cmap='nipy_spectral', noverlap=250)
	plt.colorbar(im)
	plt.show()
	#write(filename[:len(filename)-4] + "-inversed.wav", frecuencia, fftTruncadaInversa.astype(datos.dtype))
"""

################################################### NUEVAS FUNCIONES ###################################################

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

# ...
def audio_spectrogram(filename, title, xlabel, ylabel, signal, frequency):
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

def get_wn(targetFreq, sampleRate=44100):
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
	#zi = signal.lfilter_zi(b, a)
	#z, _ = signal.lfilter(b, a, fsignal, zi=zi*fsignal[0])
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

	# get spectrograms
	audio_spectrogram(filenameButter + SPEC, TITLE, XLABEL, YLABEL, dataButter, frequency)
	audio_spectrogram(filenameCheby + SPEC, TITLE, XLABEL, YLABEL, dataCheby, frequency)
	audio_spectrogram(filenameChebyInverted + SPEC, TITLE, XLABEL, YLABEL, dataChebyInverted, frequency)

	# inverted fourier graphs (basically signals) plus fourier transform graphs
	graficar(filenameButter + FFT, TITLE2, YLABEL2, XLABEL2, abs(fftButter), fftSamples)
	graficar(filenameButter + IFFT, TITLE3, YLABEL3, XLABEL3, fourier_inverse(fftButter), times)
	graficar(filenameCheby + FFT,TITLE2, YLABEL2, XLABEL2, abs(fftCheby), fftSamples)
	graficar(filenameCheby + IFFT,TITLE3, YLABEL3, XLABEL3, fourier_inverse(fftCheby), times)
	graficar(filenameChebyInverted + FFT,TITLE2, YLABEL2, XLABEL2, abs(fftChebyInverted), fftSamples)
	graficar(filenameChebyInverted + IFFT,TITLE3, YLABEL3, XLABEL3, fourier_inverse(fftChebyInverted), times)

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

	audio_spectrogram(filenameNoExtension + SPEC, TITLE, XLABEL, YLABEL, datos, frecuencia)
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'low')
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'high')
	filter_pass(filenameNoExtension, datos, frecuencia, tiempos, 'band')

	


################################################# FIN NUEVAS FUNCIONES #################################################


################ Bloque Main ##################
#process_file("lab1-1.wav")
process_file("lab1-2.wav")