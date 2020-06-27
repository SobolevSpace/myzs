# how to run this code
0.  Prepare a dataset. You will find it at https://download.zerospeech.com/.
1.	Used libs and their editions are listed in file “/src/myzs/requirments.txt”.  
2.	Make sure your file structure is like that(it’s correct at origin):
```file structure
myzs
├─ cfg
|  ├─ speakers.json
|  └─ cfg.json
├─ checkpoint
|  └─ model.pt
├─ dataset
|  └─english
|     ├─ synthesis.txt
|     ├─ test
|     |   ├─ S002_xxxx.wav
|     |   └─ ...
|     └─ train
|         ├─ unit
|         |   ├─ S133_xxxx.wav
|         |   └─ ...
|         └─ voice
|             ├─ V001_xxxx.wav
|             └─ ...
├─ output
├─ preprocessed_file
|  ├─ test
|  └─ train
|     ├─ unit
|     └─ voice
├─ tensorboard
|  └─ writer
├─ validation
|  ├─ V001_xxx.wav
|  └─ ...
├─ convert.py
├─ dataset.py
├─ encode.py
├─ evaluate.py
├─ model.py
├─ preprocess.py
├─ train.py
└─ requirements.txt
```
3.	Make sure the format of your json file and “synthesis.txt” is right. (If you don’t modify them, they are correct initially).  
4.	Preprocess.  
```shell code
python preprocess.py
```
Preprocess will generate files over 20 times larger than dataset, make sure your disk is available.  
5.	Train  
```shell code
python train.py
```
A trained model is provided in our project. However, if you want to train it, use above instruction.  
For convenience, it will train by 1 epoch per time. In real training process, it will train by 10000 epochs. The new model will be saved at “./checkpoint” automatically.  
By the way, if you want to create a new model and train it from zero, you need to modify “train.py” line 137 to “train_model(False)”.  
6.	Generate synthesis wave  
Make sure a “synthesis.txt” and input wave and a model in correct directory are prepared. And then you could run instructions below:  
```shell code
python convert.py
```
The output will be placed at directory “src/myzs/output/”.  
If you want to use model from other path, please change the variable “checkpoint_path” correctly.  
7.	Evaluation
To calculate ABX score, we need the real audio and the synthesis audio. Real audio could be found at https://download.zerospeech.com/.  
You can also find them at my github: https://github.com/SobolevSpace/myzs  
To calculate the similarity of synthesis audio and real audio, use below instructions:
```shell code
python evaluate.py
```
