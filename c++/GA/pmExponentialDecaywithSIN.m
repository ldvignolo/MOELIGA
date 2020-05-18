gamma1 = 100;
gamma2 = 1;
L = 1500;
Gmax = 1000;


%=========================
% VECTOR DE GENERACIONES
%=========================
G = 0:Gmax;


%====================================================
% PROBABILIDAD DE MUTACION (DECAIMIENTO EXPONENCIAL)
%====================================================
pm = pmExponentialDecay(gamma1, gamma2, L, Gmax, G);


%==============================================
% VARIACION SINUSOIDAL DE LA TASA DE MUTACION
%==============================================
t = linspace(0,1-1/Gmax,Gmax+1);
f = 5;
y = 0.35*pm.*sin(2*pi*f*t);


%=======================================================================
% GRAFICO DE LA PROBABILIDAD DE MUTACION DE ACUERDO A LAS GENERACIONES
%=======================================================================
plot(G,pm.*L,'-r')
hold on
plot(G,(pm+y).*L,'-b')
grid on
hold off