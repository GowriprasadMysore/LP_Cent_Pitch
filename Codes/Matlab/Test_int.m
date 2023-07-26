close all; clear all; clc;

% f = @(x) log(abs(x));
% I = integral(f,-1,1);
% x  = linspace(-1,1,1024); 
% dx = (2/length(x));
% I1 = 0.5*dx*(log(abs(x(1))) + log(abs(x(end)))) + sum(log(abs(x(2:end-1))).*dx); 
% xt = x.^3;
% J  = 1./(3*x.^2); 
% I2 = 0.5*dx*(log(abs(x(1)))*J(1) + J(end)*log(abs(x(end)))) + sum(log(abs(x(2:end-1))).*dx.*J(2:end-1));
% 
% C_0 = 1200.*log2(1/f0);
% C_end = 1200.*log2(Fs/(2*f0));
% 
% C_f = linspace(C_0,C_end,N);
% 
% f_warped = f0.*pow2(C_f/1200);
% f_warped(1) = 0;
% f = linspace(0,Fs/2,N);
% 
% Pw_ones = ones(1,length(f));
%% 
Fs = 1000;
N = 100;
f0 = 13;
M=2*N;

omega = linspace(-pi,pi,M); 
f_vec = (-N+1:1:N).*Fs/M;
f_pov = f_vec(N+1:end);

C_vec = 1200.*log2(f_pov./f0);
C_f = linspace(C_vec(1),C_vec(end),N);
f_warped = f0.*pow2(C_f/1200);

f_warped_full = [flip(-f_warped(1:end-1)),0,f_warped];
% dfreq = diff(f_warped_full);
omega_axis = 2*pi.*f_warped_full/Fs;
domega_axis = diff(omega_axis);
domega_axis = [domega_axis(end),domega_axis];


% plot(f_pov,C_vec)
% hold on
% scatter(f_warped,C_f)

%% Generate the Spectrum 

G = 1; p = 10; 
bk = randn(p,1) + 1j.*randn(p,1);
bk = bk./norm(bk);

for i = 1 : M
    Pw(i) = G^2/abs(1 + sum(transpose(bk).*exp(-1j.*(1:p)*f_warped_full(i))))^2;
end
Pw = 0.5*(Pw + flip(Pw));
% plot(Pw,'-o','linewidth',2);

Pw_half = Pw(M/2+1:end);


%% Resample the spectrum for desired points

Pw_ones = ones(1,length(f_vec));

for ii = 1 : length(f_warped_full)
    Pw_resampled(ii) = interp1(f_vec,Pw,f_warped_full(ii));
end

% plot(f_vec,Pw);
% hold on
% % scatter(f_warped_full,Pw_ones);
% scatter(f_warped_full, Pw_resampled)
% delta_f_warped = diff(f_warped);


%% 

% spec  = abs(sin(omega)+cos(omega))+omega.^2;
%spec = spec./norm(spec);

%% Inverse 


k = 16;
for i = 1 : k 
    R(i+1,1) = 1/(2*pi)*sum(Pw_resampled.*cos(i.*omega_axis).*domega_axis);
end
R(1,1) = 1/(2*pi)*sum(Pw_resampled.*domega_axis);
T = toeplitz(R(1:k,1));
v  = -R(2:end);
ak = T\v;
for i = 1 : M
 in(i) =  G^2/abs(1 + sum(transpose(ak).*exp(-1j.*(1:1:k)*omega_axis(i))))^2;
end


%%

plot(Pw_resampled,'-o','linewidth',2);

hold on
plot(in,'-.','linewidth',2)
grid on
legend('$P(\omega)$','$\hat{P}(\omega)$','interpreter','latex')
set(gca,'fontsize',30)


% plot(f_warped,f)
% 
% hold on
% plot(f_warped,C_f)

