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
    
  ''' loads and reads audio file '''
  wav_loader = io_ops.read_file(FLAGS.input_wav)
  wav_decoder = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=desired_samples)  
  sample_rate, audio = sess.run(
    [
      wav_decoder.sample_rate,
      wav_decoder.audio
    ])

  #print(audio.shape)
  #print(audio)
  plt.figure(1)
  plt.plot(np.concatenate(audio,axis=0))
  plt.show()
  
  ''' scale shift and padd '''
  scaled_foreground = audio * FLAGS.scale_factor
  #print(scaled_foreground)
  
  time_shift_amount = np.random.randint(-FLAGS.time_shift, FLAGS.time_shift)
  if time_shift_amount > 0:
    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
    time_shift_offset = [0, 0]
  else:
    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
    time_shift_offset = [-time_shift_amount, 0]
    
  print('padding :', time_shift_offset, 'ms')
  print('shifting :', time_shift_padding)
    
  # padding
  padded_foreground = tf.pad(
    scaled_foreground,
    time_shift_padding,
    mode='CONSTANT')   
  padded_foreground = sess.run(padded_foreground)
  
  plt.figure(2)
  plt.plot(padded_foreground)
  plt.show()
  
  # slicing 
  sliced_foreground = tf.slice(padded_foreground,
                               time_shift_offset,
                               [FLAGS.sample_rate, -1])                              
  sliced_foreground = sess.run(sliced_foreground) 
  
  plt.figure(3)
  plt.plot(sliced_foreground)
  plt.show()

  current_wav = sliced_foreground

  if FLAGS.add_noise==True:
    ''' loads and reads noise audio file '''
    wav_loader = io_ops.read_file(FLAGS.noise_input_wav)
    wav_decoder = contrib_audio.decode_wav(
      wav_loader, desired_channels=1, desired_samples=desired_samples)  
    sample_rate, noise_audio = sess.run(
      [
        wav_decoder.sample_rate,
        wav_decoder.audio
      ])
    noise_audio *= FLAGS.noise_scale_factor

    plt.figure(4)
    plt.plot(np.concatenate(noise_audio,axis=0))
    plt.show()

    ''' add noise to audio '''
    noise_add = sess.run(tf.add(noise_audio, sliced_foreground))
    #print('add:', noise_add)
    plt.figure(4)
    plt.plot(np.concatenate(noise_add,axis=0))
    plt.show()
    
    noise_clamp = sess.run(tf.clip_by_value(noise_add, -1.0, 1.0))
    #print('clamp:',noise_clamp.shape)
    plt.figure(3)
    plt.plot(np.concatenate(noise_clamp,axis=0))
    plt.show()

    current_wav = noise_clamp


  ''' create spectrogram '''
  spectrogram = contrib_audio.audio_spectrogram(
        current_wav,
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
  plt.show()
  
  spectrogram_length = desired_samples / window_size_samples
  
  print('spectrogram length:', spectrogram_length)
  print('fingerprint_size  :', fingerprint_size)
  
  ''' create mfcc '''
  mfcc = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=FLAGS.dct_coefficient_count)       
  mfcc = sess.run(mfcc)#.flatten()
  print('mfcc shape :', mfcc.shape)
  #print(mfcc)
  
  fig, ax = plt.subplots()
  cax = ax.matshow(mfcc[0], 
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
    '--add_noise',
    type=bool,
    default=True,
    help='whether to add noise.')

  parser.add_argument(
    '--noise_input_wav',
    type=str,
    default='data/_background_noise_/doing_the_dishes.wav',
    help='noise .wav files.')

  parser.add_argument(
    '--noise_scale_factor',
    type=float,
    default=0.1,
    help='coefficient to scale noise volume by.')
    
  parser.add_argument(
    '--time_shift',
    type=float,
    default=200.0,
    help='range to randomly shift audio in time (ms).')
    
  parser.add_argument(
    '--scale_factor',
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

