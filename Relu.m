classdef Relu < Layer
    properties
        input;
        output;
        delta;
    end

    methods
        function layer = Relu(name)
            layer = layer@Layer(name);
        end

        function layer = forward(layer, input)
            % Modified Code
            layer.input = single(reshape(input,[], 1));
            layer.output = max(layer.input, 0);
        end

        function layer = backprop(layer, delta)
            % Your codes here
            
        end
    end
end
