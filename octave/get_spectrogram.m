function get_spectrogram
  % fs...audio rate ; y...audio data
  [y,fs]=audioread('ab00c4b2_nohash_0.wav'); 
  figure(1)
  plot(y,'b-')
  
  figure(2)
  step=fix(10*fs/1000); %one spectral slice every 10ms
  window=fix(30*fs/1000); %30ms data window
  fftn=2^nextpow2(window); 
  [S,f,t]=specgram(y,fftn,fs,window,window-step);
  S=abs(S(2:fftn*4000/fs,:));
  S=S/max(S(:));
  S=max(S,10^(-40/10));
  S=min(S,10^(-3/10));
  imagesc(flipud(log(S)));
  %imagesc(t,f,log(S));
  %set(gca,"ydir","normal");
end
