clear all
close all
clc
%%
N = 2048;
omega = linspace(-pi,pi,N); 
% spec  = abs(sin(omega)+cos(omega))+omega.^2;
%spec = spec./norm(spec);
domega = abs(omega(2) - omega(1));

%% Generate the signal
G = 1; p = 10; 
bk = randn(p,1) + 1j.*randn(p,1);
bk = bk./norm(bk);
for i = 1 : N
    Pw(i) = G^2/abs(1 + sum(transpose(bk).*exp(-1j.*(1:p)*omega(i))))^2;
end
Pw = 0.5*(Pw + flip(Pw));

[y,Fs] = audioread("test_C.mp3");
x = y(1:2000,:);
X = fft(x,2048);
Pw = abs(fftshift(X))';

plot(Pw,'-o','linewidth',2);

%% Inverse 

k = 16;
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


%% Cent scale
[y,Fs] = audioread("test_C.mp3");
x = y(1:2000,:);
X = fft(x,2048);
X = fftshift(X);
f = Fs/2*linspace(-1,1,2048);

f0 = 132;
C_f = 1200.*log2(f/f0);

figure();
subplot(211)
plot(f,abs(X))
subplot(212)
plot(C_f,abs(X));

 

