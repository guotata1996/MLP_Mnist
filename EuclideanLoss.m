classdef EuclideanLoss < handle

    properties
        name;
    end

    methods
        function layer = EuclideanLoss(name)
            layer = layer@handle();
            layer.name = name;
        end

        function loss = forward(layer, input, target)
            loss = sum(mean((target - input) .* (target - input), 2));
        end

        function delta = backprop(layer, input, target)
            delta = single(mean(input - target,2));
        end
    end
end
