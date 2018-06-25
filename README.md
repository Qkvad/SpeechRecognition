# Speech Recognition

## ToDo

- Jesu li dva zapisa iste/razlicite rijeci jednake duljine?
- @ivceh - SVD
- @Qkvad & @sandrolovnicki - TensorFlow
- Android App (optional)
- Hidden Markov Models

## data

http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

---

## tf

In this directory, we implement a tensorflow approach to the problem with multiple different models, each susceptible to further tweaks. Source code will be in src/ directory and is the most important and richest part. Additionally, we provide logs/ folder in which training logs and checkpoints are being saved during each training, enabling to proceed training from arbitrary checkpoint step. In models/ directory we provide models thrat we created and could be used to predict certain speech commands.

### example usage

First off, we need to run training script which will download needed dataset in data/ folder if it is not already there and start training the model. We advise that one first inspects `tf/src/train.py` file, especially possible parser arguments which could be easily understood and tweaked as desired.  

To train a simplest model (neural network with one hidden layer) which is currently defaulr `--model_architecture` parameter, just run the train script from your anaconda environment:
```bash
python tf/src/train.py
```
**note:** best performing model is convolutional neural network, but its training could last throught the entire day. To train this model, just call the script like this:
```bash
python tf/src/train.py --model_architecture=conv
```  

Next, once we are satisfied with the performance of our model (irregardles of whether model finished its training or yout interupted it (just be sure you have `labels.txt`)), we need to "freeze" our tensorflow graph so we could reuse it for predicting.
```bash
python tf/src/freeze.py \
--model_architecture=single_fc \
--start_checkpoint=tf/logs/single_fc/train_progress/single_fc.ck-pt-2400 \
--output_file=tf/models/single_fc_032.pb
```  

And finnaly, let's predict something with our graph:
```bash
python tf/src/label_wav.py \
--graph=tf/models/single_fc_032.pb \
--wav=data/left/a5d485dc_nohash_0.wav \
--labels=tf/logs/single_fc/train_progress/single_fc_labels.txt
```  
which gives us lousy (as expected) values
```
off (score = 0.44374)
up (score = 0.22972)
_silence_ (score = 0.16772)
```  

**Info:** In `models/`, there is also a "frozen" convolutional network model `conv0875.pb` we trained, which you can test to make sure it really is performing better than rest, without the need for yout to train it for days. 

Additionally, you can record your own sounds and test them, just make sure they are 1s long and of right format.  
**Hint:** Try with
```
arecord test.wav -r 16000 -d 1 -f S16_LE
```

