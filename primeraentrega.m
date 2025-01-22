% ---------------------------------------------
% Clasificación Bayesiana con Diferentes Probabilidades
% ---------------------------------------------

% Datos del problema
W1 = [-0.5, 0, 1, 0.5, 1; 1, 2, 1, 0.5, -1]';      % Clase W1
W2 = [-2, -0.5, -1, 0.5, 0; 0, -0.5, -2, -1, -2]'; % Clase W2
X = [-1, 0, 2, 2; 0, 0, -2, 0]';                   % Puntos a clasificar

% ---------------------------------------------
% Cálculo de medias y de la matriz de covarianza
% ---------------------------------------------
mu1 = mean(W1); % Media de W1
mu2 = mean(W2); % Media de W2
m = [mu1; mu2]; % Matriz de medias (cada fila corresponde a una clase)

% Matriz de covarianzas (asumimos que ambas clases comparten la misma)
S = cov([W1; W2]);

% ---------------------------------------------
% Definición de funciones discriminantes
% ---------------------------------------------
syms x1 x2
x = [x1; x2]; % Vector simbólico para las funciones discriminantes

% Caso 1: Probabilidades iguales P1 = P2 = 0.5
P1_1 = 0.5;
P2_1 = 0.5;

% Funciones discriminantes para P1 = P2
g1_caso1 = -0.5 * ((x - mu1')' / S) * (x - mu1');
g2_caso1 = -0.5 * ((x - mu2')' / S) * (x - mu2');

% Frontera de decisión para P1 = P2
frontera_decision_caso1 = simplify(g1_caso1 - g2_caso1);

% Caso 2: Probabilidades desiguales P1 = 1/3, P2 = 2/3
P1_2 = 1/3;
P2_2 = 2/3;

% Funciones discriminantes para P1 = 1/3 y P2 = 2/3
g1_caso2 = -0.5 * ((x - mu1')' / S) * (x - mu1') + log(P1_2);
g2_caso2 = -0.5 * ((x - mu2')' / S) * (x - mu2') + log(P2_2);

% Frontera de decisión para P1 = 1/3, P2 = 2/3
frontera_decision_caso2 = simplify(g1_caso2 - g2_caso2);

% ---------------------------------------------
% Clasificación de los puntos en X
% ---------------------------------------------
disp('Clasificación para P1 = P2 = 0.5:');
for i = 1:size(X, 1)
    x = X(i, :)'; % Punto actual
    valor_g1 = double(subs(g1_caso1, [x1, x2], x'));
    valor_g2 = double(subs(g2_caso1, [x1, x2], x'));
    
    % Clasificación según las funciones discriminantes
    if valor_g1 > valor_g2
        fprintf('Punto (%.f, %.f) pertenece a W1\n', x);
    else
        fprintf('Punto (%.f, %.f) pertenece a W2\n', x);
    end
end

disp('Clasificación para P1 = 1/3 y P2 = 2/3:');
for i = 1:size(X, 1)
    x = X(i, :)'; % Punto actual
    valor_g1 = double(subs(g1_caso2, [x1, x2], x'));
    valor_g2 = double(subs(g2_caso2, [x1, x2], x'));
    
    % Clasificación según las funciones discriminantes
    if valor_g1 > valor_g2
        fprintf('Punto (%.f, %.f) pertenece a W1\n', x);
    else
        fprintf('Punto (%.f, %.f) pertenece a W2\n', x);
    end
end

% ---------------------------------------------
% Clasificación usando la distancia de Mahalanobis
% ---------------------------------------------
disp('Clasificación usando la distancia de Mahalanobis:');
for i = 1:size(X, 1)
    x = X(i, :)'; % Punto actual
    m = [mu1', mu2']; % Matriz de medias (columnas: cada clase)
    z = clasificador_mahalanobis(m, S, x); % Índice de la clase más cercana
    
    % Mostrar resultados
    if z == 1
        fprintf('Punto (%.f, %.f) pertenece a W1\n', x);
    else
        fprintf('Punto (%.f, %.f) pertenece a W2\n', x);
    end
end


% ---------------------------------------------
% Graficar los resultados
% ---------------------------------------------
figure;
hold on;

% Graficar patrones de W1 y W2
scatter(W1(:, 1), W1(:, 2), 'r', 'filled', 'DisplayName', 'W1');
scatter(W2(:, 1), W2(:, 2), 'b', 'filled', 'DisplayName', 'W2');

% Graficar los puntos de X
scatter(X(:, 1), X(:, 2), 'k', 'filled', 'DisplayName', 'Puntos X');

% Graficar las fronteras de decisión
fimplicit(matlabFunction(frontera_decision_caso1), [-3, 3, -3, 3], 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Frontera P1=P2');
fimplicit(matlabFunction(frontera_decision_caso2), [-3, 3, -3, 3], 'g--', ...
    'LineWidth', 1.5, 'DisplayName', 'Frontera P1=1/3, P2=2/3');
legend('Location', 'best');
title('Clasificación Bayesiana con Diferentes Probabilidades');
xlabel('x1');
ylabel('x2');
grid on;
hold off;



% En este caso, las probabilidades a priori de ambas clases son iguales. Esto implica que la clasificación se basa únicamente en las distancias relativas a las medias de las clases, ponderadas por la matriz de covarianza.

% Aquí, se da mayor peso a la clase W2 debido a su mayor probabilidad a priori .Como resultado, más puntos se asignan a W2
%Esto demuestra que la probabilidad a priori puede inclinar la decisión hacia una clase, incluso si un punto está más cerca de la otra clase en términos de distancia.

% en ambos casos la clasificación depende únicamente de las distancias relativas.
