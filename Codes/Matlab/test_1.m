clear all
close all
clc

%%

N = 1025;
omega = linspace(-pi,pi,N); 
% spec  = abs(sin(omega)+cos(omega))+omega.^2;
%spec = spec./norm(spec);
domega = abs(omega(2) - omega(1));

%% Generate the signal 

G = 1; p = 10; 
% bk = randn(p,1) + 1j.*randn(p,1);
% bk = bk./norm(bk);
% for i = 1 : N
%     Pw(i) = G^2/abs(1 + sum(transpose(bk).*exp(-1j.*(1:p)*omega(i))))^2;
% end
% Pw = 0.5*(Pw + flip(Pw));


% [y,Fs] = audioread("test_C.wav");
% 
% x = y(1:1024,:);
% X = fft(x,1024);
% X = abs(fftshift(X))';
% Pw = [X,X(1)];
% 
% 
% plot(Pw,'-o','linewidth',2);


[y,Fs] = audioread("test_1.wav");

x = y(1:1024,:);
X = fft(x,1024);
X = abs(fftshift(X))';
X = [X,X(1)];
isequal(X(:),flip(X(:)))
% Pw = 0.5*(X + flip(X));


f = Fs/2*linspace(-1,1,1025);

f1 = f(:,(1+ceil(length(f)/2):end))

f0 = 265;
C_f = 1200.*log2(f1/f0);

C_f_lin = linspace(C_f(1),C_f(end),length(C_f));

X_half = X(:,(1+ceil(length(X)/2):end));

for ii = 1 : length(C_f_lin)
    X_half_resampled(ii) = interp1(C_f,X_half,C_f_lin(ii));
end

X_resampled = [flip(X_half_resampled),X(ceil(length(X)/2)),X_half_resampled];

Pw = X_resampled;

figure;
subplot(2,1,1);
plot(Pw,'-o','linewidth',2);

%% Inverse 

k = 10;
for i = 1 : k 
    R(i+1,1) = 1/(2*pi)*sum(Pw.*cos(i.*omega)*domega);
end
R(1,1) = 1/(2*pi)*sum(Pw*domega);




T = toeplitz(R(1:k,1));
v  = -R(2:end);
ak = T\v;
G = R(1,1);
for i = 1 : N
 in(i) =  G/abs(1 + sum(transpose(ak).*exp(-1j.*(1:1:k)*omega(i))))^2;
end

G = R(1,1)^2;

b = [G];
[H,F] = freqz(b,[1;ak],512, Fs);


% aa = [1,ak',zeros(1,1000)];
% sig = fft(aa,1000);
% sig1 = 1./sig;
% plot(abs(sig1))
% hold on
% plot(abs(spec),'-','linewidth',2)
hold on
plot(in,'-.','linewidth',2)
grid on
legend('$P(\omega)$','$\hat{P}(\omega)$','interpreter','latex')
set(gca,'fontsize',30)



subplot(2,1,2);
plot(Pw(:,512:end),'-o','linewidth',2);
hold on;
plot(abs(H))

%% Cent scale

% figure();
% subplot(211)
% plot(f,abs(X))
% subplot(212)
% plot(C_f,abs(X));

 
M = 6; % three formants

% compute Mth-order autocorrelation function:
% rx = zeros(1,M+1)';
% for i=1:M+1
%   rx(i) = rx(i) + speech(1:nsamps-i+1) ...
%                 * speech(1+i-1:nsamps)';
% end

rx= R;
% prepare the M by M Toeplitx covariance matrix:
covmatrix = zeros(M,M);
for i=1:M
  covmatrix(i,i:M) = rx(1:M-i+1)';
  covmatrix(i:M,i) = rx(1:M-i+1);
end

% solve "normal equations" for prediction coeffs

Acoeffs = - covmatrix \ rx(2:M+1)

Alp = [1,Acoeffs']; % LP polynomial A(z)

dbenvlp = 20*log(abs(freqz(1,Alp,1025)'));
dbsspecn = dbsspec + ones(1,nspec)*(max(dbenvlp) ...
                   - max(dbsspec)); % normalize
plot(f,[max(dbsspecn,-100);dbenv;dbenvlp]); grid;


%%

[SigTime,Fs] = audioread('Q6_FM.wav');
signal=SigTime(:,1);
Ts=1/Fs;
% sound(SigTime,Fs);
TimeLen=(0:length(signal)-1)*Ts;
signal=signal./(1.01*abs(max(signal)));

wlen = 1024;                        % window length (recomended to be power of 2)
hop = wlen/4;                       % hop size (recomended to be power of 2)
nfft = 4096; 
win = blackman(wlen, 'periodic');

[Sig_VAD]= get_VAD_Sig(signal,Fs);
[S, f, t, F, F_Cep] = stft(signal,Sig_VAD, win, hop, nfft, Fs);

%%
min_f0=20;
[f0,~,t_axis] = Get_pitch (Sig_VAD, Fs, min_f0);
f0(f0>1000)=0;







%%

[y,Fs] = audioread("test_1.wav");

x = y(1:1024,:);
X = fft(x,1024);



Pw_ifft = dct(Pw,N);
Rk = (Pw_ifft(1:k))';
T = toeplitz(Rk(1:k,1));
aks = T\Rk;


b = [1];
[H,F] = freqz(b,aks,512, Fs)

k = 30;
for i = 1 : k 
    R(i+1,1) = 1/(2*pi)*sum(Pw.*cos(i.*omega)*domega);
end
R(1,1) = 1/(2*pi)*sum(Pw*domega);
T = toeplitz(R(1:k,1));
v  = -R(2:end);
ak = T\v;
for i = 1 : N
 in(i) =  G^2/abs(1 + sum(transpose(ak).*exp(-1j.*(1:1:k)*omega(i))))^2;
end

