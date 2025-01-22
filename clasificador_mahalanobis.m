% Función para clasificar con distancia de Mahalanobis
function z = clasificador_mahalanobis(m, S, x)
    % Entrada:
    % m: Matriz de medias
    % S: Matriz de covarianzas
    % x: Vector de patrón a clasificar
    %
    % Salida:
    % z: Índice de la clase más cercana al patrón x según la distancia de Mahalanobis.
    distancias = zeros(1, size(m, 2)); % Inicializamos un vector para almacenar las 
    % distancias entre x y cada una de las clases
    for i = 1:size(m, 2)
        distancias(i) = ((x - m(:, i))' / S) * (x - m(:, i));
    end
    [~, z] = min(distancias); % Devuelve el índice de la clase más cercana
end