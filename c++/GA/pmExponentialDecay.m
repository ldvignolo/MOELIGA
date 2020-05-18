function pm = pmExponentialDecay(gamma1, gamma2, L, Gmax, G)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% pm = A * r^(G)

GAMMA1 = gamma1/L;
GAMMA2 = gamma2/L;

A = GAMMA1;
r = (GAMMA2/GAMMA1)^(1/Gmax);

pm = A*r.^(G);

end