function [aks,STFT,inv_pw,cent_spec,cent_spec_resampled,C_f_lin,t,f,C_f] = frame_process_LP(x,f0, win, hop, nfft, Fs, p)
% function: [STFT, f, t] = stft(x, win, hop, nfft, fs)
%
% Input:
% x - signal in the time domain
% win - analysis window function
% hop - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
%
% Output:
% STFT - STFT-matrix (only unique points, time 
%        across columns, frequency across rows)
% f - frequency vector, Hz
% t - time vector, s

% representation of the signal as column-vector
x = x(:);
% determination of the signal length 
xlen = length(x);
% determination of the window length
wlen = length(win);

% stft matrix size estimation and preallocation
NUP = ceil((1+nfft)/2);     % calculate the number of unique fft points
M = nfft + 1;
L = 1+fix((xlen-wlen)/hop); % calculate the number of signal frames
f = Fs/2*linspace(-1,1,M);
f1 = f(:,(1+ceil(length(f)/2):end));
C_f = 1200.*log2(f1/f0);

C_f_lin = linspace(C_f(1),C_f(end),length(C_f));

STFT = zeros(M, L);       % preallocate the stft matrix
aks = zeros(p,L);
cent_spec = zeros(nfft/2,L);
cent_spec_resampled = zeros(nfft/2,L);
inv_pw = zeros(M, L);
% pitch = zeros(1,L);
% min_f0=40;
% STFT (via time-localized FFT)
for l = 0:L-1
    % windowing
    xw = x(1+l*hop : wlen+l*hop).*win;
    % FFT
    X = fft(xw, nfft);
    X = abs(fftshift(X))';
    X = [X,X(1)];
    isequal(X(:),flip(X(:)))
    
    X_half = X(:,(1+ceil(length(X)/2):end));

    for ii = 1 : length(C_f_lin)
    X_half_resampled(ii) = interp1(C_f,X_half,C_f_lin(ii));
    end
    X_resampled = [flip(X_half_resampled),X(ceil(length(X)/2)),X_half_resampled];

    [ak, in] = inverse_lp(X_resampled, M, p);


    aks(:,1+l) = ak;
    % update of the stft matrix
    STFT(:, 1+l) = X;
    inv_pw(:, 1+l) = in;
    cent_spec(:, 1+l) = X_half;
    cent_spec_resampled(:, 1+l) = X_half_resampled;
end

% calculation of the time and frequency vectors
t = (wlen/2:hop:wlen/2+(L-1)*hop)/(2*Fs);
% f = (0:NUP-1)*fs/nfft;

end