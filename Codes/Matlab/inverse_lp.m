function [ak, in] = inverse_lp(Pw, M, p)

omega = linspace(-pi,pi,M); 
domega = abs(omega(2) - omega(1));
G = 1;

% p = 30;
for i = 1 : p 
    R(i+1,1) = 1/(2*pi)*sum(Pw.*cos(i.*omega)*domega);
end
R(1,1) = 1/(2*pi)*sum(Pw*domega);
T = toeplitz(R(1:p,1));
v  = -R(2:end);
ak = T\v;
for i = 1 : M
 in(i) =  G^2/abs(1 + sum(transpose(ak).*exp(-1j.*(1:1:p)*omega(i))))^2;
end


end