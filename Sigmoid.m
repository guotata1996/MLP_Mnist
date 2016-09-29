classdef Sigmoid < Layer
    properties
        input;
        output;
        delta;
    end

    methods
        function layer = Sigmoid(name)
            layer = layer@Layer(name);
        end

        function layer = forward(layer, input)
            % Modified Code
            layer.input = single(input);
            layer.output = 1 ./ (1 + exp(-layer.input));
        end

        function layer = backprop(layer, delta)
            % Modified Code
            layer.delta = zeros(size(delta));
            for i = 1 : size(layer.output, 2)
                layer.delta = layer.delta + delta .* layer.output(:,i) .* (1 - layer.output(:,i));
            end
            layer.delta = layer.delta / size(layer.output, 2);
        end
    end
end
