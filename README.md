# fast-wavenet
- implement fast-wavenet by pytorch
#wavenet for music generation
- the model train from the [4 hours piano solo](https://www.youtube.com/watch?v=EhO_MrRfftU)
- hyper-parameters are:
	- dilations = [2 ** i for i in range(11)] * 4
	- residual channel = 128
	- skip channel = 512
	- sample rate = 8000
	- sample size = 16000

- the [result](https://soundcloud.com/shicheng-chen-147753167/wavenet-generate-piano)

- hyper-parameters are:
	- dilations = [2 ** i for i in range(10)] * 4
	- residual channel = 128
	- skip channel = 512
	- sample rate = 8000
	- sample size = 16000

- the [result](https://soundcloud.com/shicheng-chen-147753167/wavenet-generate-piano-sound)