- h5 file: blurry_gemini_dense_miou_0.9275.h5
- fecha: 04-09-2020 // DD-MM-AAAA
- description:
	- Corresponden a los pesos con mejor rendimiento al minuto.
	- input_size: 320x320x3 (height, width, channels)
	- output_size: 320x320x4 (height, width, classes)
	- classes (ultimo canal de mascaras predichas): 
		- 0: background / periocular zone
		- 1: iris
		- 2: pupil
		- 3: sclera # no se usa para el calculo de ningun radio ni otra metrica, por lo cual su uso puede ser omitido


