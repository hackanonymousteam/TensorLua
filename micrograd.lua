-- MicroGrad Lua Library

local Value = {}
Value.__index = Value

local function ensure_value(x)
    if type(x) == "number" then
        return Value.new(x)
    end
    return x
end

function Value.new(data, children, op)
    local self = setmetatable({}, Value)
    self.data = data
    self.grad = 0
    self._backward = function() end
    self._prev = children or {}
    self._op = op or ''
    return self
end

function Value:__add(other)
    other = ensure_value(other)
    local out = Value.new(self.data + other.data, {self, other}, '+')
    
    out._backward = function()
        self.grad = self.grad + out.grad
        other.grad = other.grad + out.grad
    end
    
    return out
end

function Value:__mul(other)
    other = ensure_value(other)
    local out = Value.new(self.data * other.data, {self, other}, '*')
    
    out._backward = function()
        self.grad = self.grad + (other.data * out.grad)
        other.grad = other.grad + (self.data * out.grad)
    end
    
    return out
end

function Value:relu()
    local out = Value.new(math.max(0, self.data), {self}, 'ReLU')
    
    out._backward = function()
        if self.data > 0 then
            self.grad = self.grad + out.grad
        end
    end
    
    return out
end

function Value:backward()
    local topo = {}
    local visited = {}
    
    local function build_topo(v)
        if not visited[v] then
            visited[v] = true
            for _, child in ipairs(v._prev) do
                build_topo(child)
            end
            table.insert(topo, v)
        end
    end
    
    build_topo(self)
    
    self.grad = 1
    
    for i = #topo, 1, -1 do
        local v = topo[i]
        if v._backward then
            v._backward()
        end
    end
end

function Value:__unm()
    return self * -1
end

function Value:__neg()
    return self * -1
end

function Value:__sub(other)
    other = ensure_value(other)
    return self + (-other)
end

function Value:__truediv(other)
    other = ensure_value(other)
    local one = Value.new(1)
    return self * (one / other)
end

function Value:__tostring()
    return string.format("Value(data=%.6f, grad=%.6f)", self.data, self.grad)
end

function Value.zero_grad(list_of_values)
    for _, v in ipairs(list_of_values) do
        v.grad = 0
    end
end

function Value:__div(other)
    other = ensure_value(other)
    local out = Value.new(1 / other.data, {other}, '/')
    
    out._backward = function()
        local grad_out = out.grad
        other.grad = other.grad + (-1 / (other.data * other.data)) * grad_out
    end
    
    return self * out
end

setmetatable(Value, {
    __call = function(_, ...) 
        return Value.new(...) 
    end
})

local Module = {}
Module.__index = Module

function Module.new()
    local self = setmetatable({}, Module)
    return self
end

function Module:zero_grad()
    for _, p in ipairs(self:parameters()) do
        p.grad = 0
    end
end

function Module:parameters()
    return {}
end

local Neuron = {}
Neuron.__index = Neuron
setmetatable(Neuron, {__index = Module})

function Neuron.new(nin, activation)
    local self = setmetatable({}, Neuron)
    self.w = {}
    for i = 1, nin do
        table.insert(self.w, Value.new((math.random() * 2 - 1) * 0.1))
    end
    self.b = Value.new(0)
    self.activation = activation or 'relu'
    return self
end

function Neuron:__call(x)
    local act = self.b
    for i = 1, #self.w do
        act = act + self.w[i] * x[i]
    end
    
    if self.activation == 'relu' then
        return act:relu()
    end
    
    return act
end

function Neuron:parameters()
    local params = {}
    for _, w in ipairs(self.w) do
        table.insert(params, w)
    end
    table.insert(params, self.b)
    return params
end

local Layer = {}
Layer.__index = Layer
setmetatable(Layer, {__index = Module})

function Layer.new(nin, nout, activation)
    local self = setmetatable({}, Layer)
    self.neurons = {}
    for i = 1, nout do
        table.insert(self.neurons, Neuron.new(nin, activation))
    end
    return self
end

function Layer:__call(x)
    local outputs = {}
    for _, neuron in ipairs(self.neurons) do
        table.insert(outputs, neuron(x))
    end
    return outputs
end

function Layer:parameters()
    local params = {}
    for _, neuron in ipairs(self.neurons) do
        for _, p in ipairs(neuron:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

local MLP = {}
MLP.__index = MLP
setmetatable(MLP, {__index = Module})

function MLP.new(nin, nouts, activations)
    local self = setmetatable({}, MLP)
    local sz = {nin}
    for _, nout in ipairs(nouts) do
        table.insert(sz, nout)
    end
    
    self.layers = {}
    for i = 1, #sz - 1 do
        local activation
        if activations then
            activation = activations[i]
        else
            activation = i ~= #sz - 1 and 'relu' or 'none'
        end
        table.insert(self.layers, Layer.new(sz[i], sz[i + 1], activation))
    end
    
    return self
end

function MLP:__call(x)
    local output = x
    for i, layer in ipairs(self.layers) do
        output = layer(output)
        if i == #self.layers and type(output) == "table" and #output == 1 then
            return output[1]
        end
    end
    return output
end

function MLP:parameters()
    local params = {}
    for _, layer in ipairs(self.layers) do
        for _, p in ipairs(layer:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

local Optimizer = {}
Optimizer.__index = Optimizer

function Optimizer.new(parameters, lr)
    local self = setmetatable({}, Optimizer)
    self.parameters = parameters
    self.lr = lr or 0.01
    return self
end

function Optimizer:step()
    for _, p in ipairs(self.parameters) do
        p.data = p.data - self.lr * p.grad
    end
end

function Optimizer:zero_grad()
    for _, p in ipairs(self.parameters) do
        p.grad = 0
    end
end

local SGD = {}
SGD.__index = SGD
setmetatable(SGD, {__index = Optimizer})

function SGD.new(parameters, lr, momentum)
    local self = setmetatable({}, SGD)
    self.parameters = parameters
    self.lr = lr or 0.01
    self.momentum = momentum or 0.9
    self.velocities = {}
    for _, p in ipairs(parameters) do
        self.velocities[p] = 0
    end
    return self
end

function SGD:step()
    for _, p in ipairs(self.parameters) do
        local grad = p.grad or 0
        self.velocities[p] = self.momentum * self.velocities[p] + self.lr * grad
        p.data = p.data - self.velocities[p]
    end
end

local Loss = {}

function Loss.mse(predictions, targets)
    local total_loss = Value.new(0)
    local n = #predictions
    
    for i = 1, n do
        local diff = predictions[i] - targets[i]
        total_loss = total_loss + (diff * diff)
    end
    
    return total_loss / Value.new(n)
end

function Loss.mse_single(prediction, target)
    prediction = ensure_value(prediction)
    target = ensure_value(target)
    local diff = prediction - target
    return diff * diff
end

local Utils = {}

function Utils.to_value(x)
    if type(x) == "table" and x.data ~= nil then
        return x
    elseif type(x) == "number" then
        return Value.new(x)
    elseif type(x) == "table" then
        local result = {}
        for i, v in ipairs(x) do
            result[i] = Utils.to_value(v)
        end
        return result
    end
    return Value.new(0)
end

function Utils.to_number(values)
    if type(values) == "table" and values.data ~= nil then
        return values.data
    elseif type(values) == "table" then
        local result = {}
        for i, v in ipairs(values) do
            if type(v) == "table" and v.data ~= nil then
                result[i] = v.data
            else
                result[i] = v
            end
        end
        return result
    end
    return values
end

function Utils.generate_random_data(n, dims)
    local X, y = {}, {}
    for i = 1, n do
        local sample = {}
        for d = 1, dims do
            table.insert(sample, math.random())
        end
        table.insert(X, sample)
        table.insert(y, math.random())
    end
    return X, y
end

function Utils.shuffle_data(X, y)
    local n = #X
    for i = n, 2, -1 do
        local j = math.random(1, i)
        X[i], X[j] = X[j], X[i]
        y[i], y[j] = y[j], y[i]
    end
    return X, y
end

local Trainer = {}

function Trainer.train(model, X_train, y_train, epochs, learning_rate, options)
    options = options or {}
    local batch_size = options.batch_size or 1
    local loss_fn = options.loss_fn or Loss.mse_single
    local optimizer_type = options.optimizer or 'sgd'
    local verbose = options.verbose or false
    local shuffle = options.shuffle or true
    
    local X_values = {}
    for i = 1, #X_train do
        X_values[i] = Utils.to_value(X_train[i])
    end
    
    local y_values = {}
    for i = 1, #y_train do
        y_values[i] = Utils.to_value(y_train[i])
    end
    
    local optimizer
    if optimizer_type == 'sgd' then
        optimizer = SGD.new(model:parameters(), learning_rate, 0.9)
    else
        optimizer = Optimizer.new(model:parameters(), learning_rate)
    end
    
    local n_samples = #X_values
    local n_batches = math.ceil(n_samples / batch_size)
    
    for epoch = 1, epochs do
        local total_loss = 0
        
        if shuffle then
            X_values, y_values = Utils.shuffle_data(X_values, y_values)
        end
        
        for batch = 1, n_batches do
            local start_idx = (batch - 1) * batch_size + 1
            local end_idx = math.min(batch * batch_size, n_samples)
            
            local batch_loss = Value.new(0)
            local batch_count = 0
            
            for i = start_idx, end_idx do
                local prediction = model(X_values[i])
                local loss = loss_fn(prediction, y_values[i])
                batch_loss = batch_loss + loss
                batch_count = batch_count + 1
            end
            
            if batch_count > 0 then
                local n_value = Value.new(batch_count)
                local avg_batch_loss = batch_loss / n_value
                total_loss = total_loss + avg_batch_loss.data
                
                optimizer:zero_grad()
                model:zero_grad()
                avg_batch_loss:backward()
                optimizer:step()
            end
        end
        
        if verbose and epoch % 100 == 0 then
            local avg_loss = total_loss / n_batches
            print(string.format("Epoch %d, Loss: %.6f", epoch, avg_loss))
        end
    end
    
    return model
end

function Trainer.predict(model, X)
    local X_values = Utils.to_value(X)
    local predictions = {}
    
    if type(X_values[1]) ~= "table" then
        local pred = model(X_values)
        return pred.data
    end
    
    for i = 1, #X_values do
        local pred = model(X_values[i])
        table.insert(predictions, pred.data)
    end
    
    return predictions
end

local micrograd = {
    Value = Value,
    Module = Module,
    Neuron = Neuron,
    Layer = Layer,
    MLP = MLP,
    Optimizer = Optimizer,
    SGD = SGD,
    Loss = Loss,
    Utils = Utils,
    Trainer = Trainer
}

return micrograd