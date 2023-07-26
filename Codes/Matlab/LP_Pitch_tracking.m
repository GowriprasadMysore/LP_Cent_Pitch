clear all
close all
clc

%%

N = 4096;
M = N+1;
omega = linspace(-pi,pi,M); 
domega = abs(omega(2) - omega(1));

%% Generate the signal 

G = 1;
[x,Fs] = audioread("test_4_vocal.wav");
f0=220;
% sound(SigTime,Fs);
% TimeLen=(0:length(signal)-1)*Ts;
x=x./(1.01*abs(max(x)));

wlen = N;                        % window length (recomended to be power of 2)
hop = wlen/4;                       % hop size (recomended to be power of 2)
nfft = N; 
win = blackman(wlen, 'periodic');
p=20;
[aks,STFT,inv_pw,cent_spec,cent_spec_resampled,C_f_lin,t,f,C_f] = frame_process_LP(x,f0, win, hop, nfft, Fs, p);

%%

figure();
subplot(411)
imagesc(t,f(:,N/2+1:end),20*log(STFT(N/2+1:end,:)));
ylim([0,4000])
set(gca,'YDir','normal');
title('Regular STFT');
ylabel('Freq (Hz)');

subplot(412)
imagesc(t,C_f,20*log(cent_spec));
set(gca,'YDir','normal');
title('Regular Cent Scale');
ylabel('Cents')


subplot(413)
imagesc(t,C_f_lin,20*log(cent_spec_resampled));
set(gca,'YDir','normal');
title('Uniform Cent Scale');
ylabel('Cents')


subplot(414)
imagesc(t,C_f_lin,20*log(inv_pw(N/2:end,:)));
set(gca,'YDir','normal');
title('LP on Uniform Cent Scale');
xlabel('time (sec)')
ylabel('Freq (Hz)')


%% Track Pitch
% From regular spectrum

% loc = {};
% val = {};
for ii = 1:length(STFT(N/2:end,:))

[vals,locs] = findpeaks(STFT(N/2+10:end,ii));
[val(ii),loc(ii)] = maxk(vals,1); 

end



% From regular spectrum

% loc = {};
% val = {};
for ii = 1:length(20*log(inv_pw(N/2:end,:)))

[vals,locs] = findpeaks(20*log(inv_pw(N/2:end,ii)));
[val(ii),lo] = maxk(vals,1); 
loc(ii)=locs(lo)
end

%%
fx = f(:,N/2+1:end);
figure();
subplot(411)
imagesc(t,fx,20*log(STFT(N/2+1:end,:)));
ylim([0,1000])
set(gca,'YDir','normal');
hold on
% plot(t,fx(loc))

title('Regular STFT');
ylabel('Freq (Hz)');

subplot(412)
imagesc(t,C_f,20*log(cent_spec));
set(gca,'YDir','normal');
title('Regular Cent Scale');
ylabel('Cents')


subplot(413)
imagesc(t,C_f_lin,20*log(cent_spec_resampled));
set(gca,'YDir','normal');
title('Uniform Cent Scale');
ylabel('Cents')


subplot(414)
imagesc(t,C_f_lin,20*log(inv_pw(N/2:end,:)));
set(gca,'YDir','normal');
ylim([-2400,3600])
hold on
plot(t,C_f_lin(loc))
title('LP on Uniform Cent Scale');
xlabel('time (sec)')
ylabel('Freq (Hz)')

%%
d = 20*log(inv_pw(N/2:end,1000));    % assumed data
[vals,locs] = findpeaks(d);
[val,loc] = maxk(vals,2);    % first two peaks(mnaximum values) sample values and locations
plot(d); hold on;    % plot original samples
plot(locs(loc),val,'rv', 'MarkerFaceColor', 'r');    % plot peak values


