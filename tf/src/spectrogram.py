# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Spectogram input/output.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


def main(_):
  # enable logging
  tf.logging.set_verbosity(tf.logging.INFO)
  # start tensorflow session
  sess = tf.Session()
  
  ''' model settings '''
  desired_samples = int(FLAGS.sample_rate * FLAGS.clip_duration_ms / 1000)
  window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
  window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = FLAGS.dct_coefficient_count * spectrogram_length
  ''' ------------- '''
  
  wav_loader = io_ops.read_file(FLAGS.input_wav)
  wav_decoder = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=desired_samples)
   
  sample_rate, audio = sess.run(
    [
      wav_decoder.sample_rate,
      wav_decoder.audio
    ])
    
  '''print(audio.shape)
  step=int(10*sample_rate/1000)
  window=int(30*sample_rate/1000)
  fftn=1
  while fftn<window:
    fftn*=2
  S,f,t=plt.specgram(x=audio[0], NFFT=fftn, Fs=sample_rate,
         window=window, pad_to=window-step)
  S=abs(S[2:fftn*4000/sample_rate][:])
  S=S/max(S[:])
  S=max(S,10^(-40/10))
  S=min(S,10^(-3/10))
  res=np.flipud(math.log(S))'''
  #plt.figure(1)
  #plt.plot(np.concatenate(audio,axis=0))
  #plt.show()
  
  ''' '''
  
  scaled_foreground = audio
  
  time_shift_amount = np.random.randint(-FLAGS.time_shift, FLAGS.time_shift)
  if time_shift_amount > 0:
    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
    time_shift_offset = [0, 0]
  else:
    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
    time_shift_offset = [-time_shift_amount, 0]
    
  print('shifting :', time_shift_padding, 'ms')
    
  padded_foreground = tf.pad(
    scaled_foreground,
    time_shift_padding,
    mode='CONSTANT')
    
  padded_foreground = sess.run(padded_foreground)
  
  #plt.figure(0)
  #plt.plot(padded_foreground)
  #plt.show()
  
  print('padding :', time_shift_offset, 'ms')
  
  sliced_foreground = tf.slice(padded_foreground,
                               time_shift_offset,
                               [FLAGS.sample_rate, -1])
                               

  sliced_foreground = sess.run(sliced_foreground) 
  
  #plt.figure(1)
  #plt.plot(sliced_foreground)
  #plt.show()
  
  ''' '''
  
  spectrogram = contrib_audio.audio_spectrogram(
        sliced_foreground,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True)
        
  spectrogram = sess.run(spectrogram)
  print('spectrogram shape :', spectrogram.shape)
  
  fig, ax = plt.subplots()
  cax = ax.matshow(spectrogram[0], 
                   interpolation='nearest', 
                   aspect='auto', 
                   cmap='gist_ncar')
  fig.colorbar(cax)
  ax.set_title('spectrogram')
  #Showing mfcc_data
  plt.show()
  
  spectrogram_length = desired_samples / window_size_samples
  
  print('spectrogram length:', spectrogram_length)
  print('fingerprint_size  :', fingerprint_size)
  
  ''' '''
  
  mfcc = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=FLAGS.dct_coefficient_count)
        
  mfcc_features = sess.run(mfcc)#.flatten()
  print('mfcc shape :', mfcc.shape)
  #print(mfcc)
  
  fig, ax = plt.subplots()
  cax = ax.matshow(mfcc_features[0], 
                   interpolation='nearest', 
                   aspect='auto', 
                   cmap=cm.afmhot, 
                   origin='lower')
  fig.colorbar(cax)
  ax.set_title('MFCC')
  plt.show()
  
    

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    '--input_wav',
    type=str,
    default='',
    help='.wav file to create spectrogram from.')
    
  parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='.wav duration in ms.')
    
  parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='expected sample rate of wav files.')
    
  parser.add_argument(
    '--time_shift',
    type=float,
    default=200.0,
    help='range to randomly shift audio in time (ms).')
    
  parser.add_argument(
    '--scale_volume',
    type=float,
    default=0.8,
    help='coefficient to scale volume by.')
    
  parser.add_argument(
    '--window_size_ms',
    type=int,
    default=20,
    help=' --- ')
    
  parser.add_argument(
    '--window_stride_ms',
    type=int,
    default=8,
    help=' --- ')
    
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

