-- Tensor Library for Game Guardian Lua

local Tensor = {}
Tensor.__index = Tensor

function Tensor.new(data, shape)
    local self = setmetatable({}, Tensor)
    self.data = data or {}
    self.shape = shape or {}
    return self
end

function Tensor:size()
    if #self.shape == 0 then return 0 end
    local total = 1
    for _, dim in ipairs(self.shape) do
        total = total * dim
    end
    return total
end

function Tensor:dim()
    return #self.shape
end

function Tensor:reshape(new_shape)
    local total_elements = self:size()
    local new_total = 1
    
    local auto_dim = nil
    for i, dim in ipairs(new_shape) do
        if dim == -1 then
            if auto_dim then
                print("Apenas uma dimensão pode ser -1")
            end
            auto_dim = i
        else
            new_total = new_total * dim
        end
    end
    
    if auto_dim then
        new_shape[auto_dim] = math.floor(total_elements / new_total)
        new_total = new_total * new_shape[auto_dim]
    end
    
    if new_total ~= total_elements then
        error(string.format("Reshape incompatível: %d elementos -> %d elementos", total_elements, new_total))
    end
    
    return Tensor.new(self.data, new_shape)
end

function Tensor:__tostring()
    local shape_str = table.concat(self.shape, ", ")
    local sample_data = ""
    if #self.data > 0 then
        sample_data = string.format(" data[1]=%.3f", self.data[1])
    end
    return string.format("Tensor(shape={%s}, size=%d%s)", shape_str, self:size(), sample_data)
end


local TensorOps = {}

function TensorOps.zeros(shape)
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    
    local data = {}
    for i = 1, size do
        data[i] = 0.0
    end
    
    return Tensor.new(data, shape)
end

function TensorOps.ones(shape)
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    
    local data = {}
    for i = 1, size do
        data[i] = 1.0
    end
    
    return Tensor.new(data, shape)
end

function TensorOps.eye(n)
    local data = {}
    local idx = 1
    for i = 1, n do
        for j = 1, n do
            if i == j then
                data[idx] = 1.0
            else
                data[idx] = 0.0
            end
            idx = idx + 1
        end
    end
    
    return Tensor.new(data, {n, n})
end

function TensorOps.randn(shape, mean, std)
    mean = mean or 0.0
    std = std or 0.1
    
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    
    local data = {}
    for i = 1, size do
        local u1 = math.random()
        local u2 = math.random()
        local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data[i] = mean + z0 * std
    end
    
    return Tensor.new(data, shape)
end

function TensorOps.full(shape, value)
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    
    local data = {}
    for i = 1, size do
        data[i] = value
    end
    
    return Tensor.new(data, shape)
end

function TensorOps.map(fn, tensor)
    local size = tensor:size()
    local result_data = {}
    
    for i = 1, size do
        result_data[i] = fn(tensor.data[i])
    end
    
    return Tensor.new(result_data, tensor.shape)
end

function TensorOps.zip(fn, tensor_a, tensor_b)
    local size = tensor_a:size()
    local result_data = {}
    
    for i = 1, size do
        local a_val = tensor_a.data[i] or 0
        local b_val = tensor_b.data[i] or 0
        result_data[i] = fn(a_val, b_val)
    end
    
    return Tensor.new(result_data, tensor_a.shape)
end

function TensorOps.sum(tensor)
    local total = 0
    for i = 1, #tensor.data do
        total = total + tensor.data[i]
    end
    return total
end

function TensorOps.mean(tensor)
    return TensorOps.sum(tensor) / tensor:size()
end

function TensorOps.matmul(tensor_a, tensor_b)
    if #tensor_a.shape < 2 or #tensor_b.shape < 2 then
        print("Tensores devem ter pelo menos 2 dimensões")
    end
    
    local m = tensor_a.shape[1]
    local k = tensor_a.shape[2]
    local n = tensor_b.shape[2]
    
    if k ~= tensor_b.shape[1] then
        error(string.format("Dimensões incompatíveis para matmul: %d e %d", k, tensor_b.shape[1]))
    end
    
    local result_shape = {m, n}
    local result = TensorOps.zeros(result_shape)
    
    for i = 1, m do
        for j = 1, n do
            local sum = 0
            for l = 1, k do
                local a_idx = (i-1)*k + l
                local b_idx = (l-1)*n + j
                sum = sum + tensor_a.data[a_idx] * tensor_b.data[b_idx]
            end
            result.data[(i-1)*n + j] = sum
        end
    end
    
    return result
end

function TensorOps.add(tensor_a, tensor_b)
    return TensorOps.zip(function(a, b) return a + b end, tensor_a, tensor_b)
end

function TensorOps.sub(tensor_a, tensor_b)
    return TensorOps.zip(function(a, b) return a - b end, tensor_a, tensor_b)
end

function TensorOps.mul(tensor_a, tensor_b)
    return TensorOps.zip(function(a, b) return a * b end, tensor_a, tensor_b)
end

function TensorOps.relu(tensor)
    return TensorOps.map(function(x)
        return math.max(0, x)
    end, tensor)
end

function TensorOps.sigmoid(tensor)
    return TensorOps.map(function(x)
        return 1 / (1 + math.exp(-x))
    end, tensor)
end

function TensorOps.tanh(tensor)
    return TensorOps.map(function(x)
        return math.tanh(x)
    end, tensor)
end

function TensorOps.softmax(tensor)
    local max_val = -math.huge
    for i = 1, #tensor.data do
        max_val = math.max(max_val, tensor.data[i])
    end
    
    local exp_sum = 0
    local exps = {}
    for i = 1, #tensor.data do
        exps[i] = math.exp(tensor.data[i] - max_val)
        exp_sum = exp_sum + exps[i]
    end
    
    local result_data = {}
    for i = 1, #tensor.data do
        result_data[i] = exps[i] / exp_sum
    end
    
    return Tensor.new(result_data, tensor.shape)
end

function TensorOps.mse_loss(pred_tensor, target_tensor)
    local diff = TensorOps.sub(pred_tensor, target_tensor)
    local squared = TensorOps.map(function(x) return x * x end, diff)
    local sum_squared = TensorOps.sum(squared)
    return sum_squared / pred_tensor:size()
end


local TensorValue = {}
TensorValue.__index = TensorValue

function TensorValue.new(data, shape, requires_grad)
    local self = setmetatable({}, TensorValue)
    self.tensor = Tensor.new(data, shape)
    self.requires_grad = requires_grad or false
    self.grad = nil
    self._backward = function() end
    self._prev = {}
    self._op = ''
    
    if requires_grad then
        self.grad = TensorOps.zeros(shape)
    end
    
    return self
end

function TensorValue:backward(gradient)
    if not self.requires_grad then return end
    
    if not gradient then
        gradient = TensorOps.ones(self.tensor.shape)
    end
    
    if not self.grad then
        self.grad = TensorOps.zeros(self.tensor.shape)
    end
    
    for i = 1, #self.grad.data do
        self.grad.data[i] = self.grad.data[i] + gradient.data[i]
    end
    
    if self._backward then
        self._backward(gradient)
    end
end

function TensorValue:__add(other)
    local other_tensor
    if type(other) == "number" then
        other_tensor = TensorOps.full(self.tensor.shape, other)
    elseif getmetatable(other) == TensorValue then
        other_tensor = other.tensor
    else
        print("Operação inválida: " .. type(other))
    end
    
    local out_data = TensorOps.add(self.tensor, other_tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    if getmetatable(other) == TensorValue then
        table.insert(out._prev, other)
    end
    out._op = '+'
    
    out._backward = function(grad)
        if self.requires_grad then
            self:backward(grad)
        end
        
        if getmetatable(other) == TensorValue and other.requires_grad then
            other:backward(grad)
        end
    end
    
    return out
end

function TensorValue:__sub(other)
    local other_tensor
    if type(other) == "number" then
        other_tensor = TensorOps.full(self.tensor.shape, other)
    elseif getmetatable(other) == TensorValue then
        other_tensor = other.tensor
    else
        print("Operação inválida: " .. type(other))
    end
    
    local out_data = TensorOps.sub(self.tensor, other_tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    if getmetatable(other) == TensorValue then
        table.insert(out._prev, other)
    end
    out._op = '-'
    
    out._backward = function(grad)
        if self.requires_grad then
            self:backward(grad)
        end
        
        if getmetatable(other) == TensorValue and other.requires_grad then
            local neg_grad = TensorOps.map(function(x) return -x end, grad)
            other:backward(neg_grad)
        end
    end
    
    return out
end

function TensorValue:__mul(other)
    local other_tensor
    if type(other) == "number" then
        other_tensor = TensorOps.full(self.tensor.shape, other)
    elseif getmetatable(other) == TensorValue then
        other_tensor = other.tensor
    else
        print("Operação inválida: " .. type(other))
    end
    
    local out_data = TensorOps.mul(self.tensor, other_tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    if getmetatable(other) == TensorValue then
        table.insert(out._prev, other)
    end
    out._op = '*'
    
    out._backward = function(grad)
        if self.requires_grad then
            local grad_self = TensorOps.mul(grad, other_tensor)
            self:backward(grad_self)
        end
        
        if getmetatable(other) == TensorValue and other.requires_grad then
            local grad_other = TensorOps.mul(grad, self.tensor)
            other:backward(grad_other)
        end
    end
    
    return out
end

function TensorValue:relu()
    local out_data = TensorOps.relu(self.tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    out._op = 'ReLU'
    
    out._backward = function(grad)
        if self.requires_grad then
            local mask = TensorOps.map(function(x) return x > 0 and 1 or 0 end, self.tensor)
            local grad_self = TensorOps.mul(grad, mask)
            self:backward(grad_self)
        end
    end
    
    return out
end

function TensorValue:sigmoid()
    local out_data = TensorOps.sigmoid(self.tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    out._op = 'sigmoid'
    
    out._backward = function(grad)
        if self.requires_grad then
            local sig = TensorOps.sigmoid(self.tensor)
            local one_minus_sig = TensorOps.map(function(x) return 1 - x end, sig)
            local deriv = TensorOps.mul(sig, one_minus_sig)
            local grad_self = TensorOps.mul(grad, deriv)
            self:backward(grad_self)
        end
    end
    
    return out
end

function TensorValue:tanh()
    local out_data = TensorOps.tanh(self.tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    out._op = 'tanh'
    
    out._backward = function(grad)
        if self.requires_grad then
            local tanh_val = TensorOps.tanh(self.tensor)
            local deriv = TensorOps.map(function(x) return 1 - x*x end, tanh_val)
            local grad_self = TensorOps.mul(grad, deriv)
            self:backward(grad_self)
        end
    end
    
    return out
end

function TensorValue:softmax()
    local out_data = TensorOps.softmax(self.tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, self.requires_grad)
    out._prev = {self}
    out._op = 'softmax'
    
    out._backward = function(grad)
        if self.requires_grad then
            
            local softmax_val = TensorOps.softmax(self.tensor)
            local grad_self = TensorOps.mul(grad, softmax_val)
            self:backward(grad_self)
        end
    end
    
    return out
end

function TensorValue.matmul(a, b)
    local out_data = TensorOps.matmul(a.tensor, b.tensor)
    local out = TensorValue.new(out_data.data, out_data.shape, a.requires_grad or b.requires_grad)
    out._prev = {a, b}
    out._op = 'matmul'
    
    out._backward = function(grad)
        if a.requires_grad then
            local grad_a = TensorOps.matmul(grad, b.tensor:reshape({b.tensor.shape[2], b.tensor.shape[1]}))
            a:backward(grad_a)
        end
        
        if b.requires_grad then
            local grad_b = TensorOps.matmul(a.tensor:reshape({a.tensor.shape[2], a.tensor.shape[1]}), grad)
            b:backward(grad_b)
        end
    end
    
    return out
end

function TensorValue:__tostring()
    local shape_str = table.concat(self.tensor.shape, ", ")
    local grad_str = self.grad and "grad=true" or "grad=false"
    return string.format("TensorValue(shape={%s}, %s)", shape_str, grad_str)
end

function TensorValue.zero_grad(list)
    for _, v in ipairs(list) do
        if v.grad then
            for i = 1, #v.grad.data do
                v.grad.data[i] = 0
            end
        end
    end
end


return {
    Tensor = Tensor,
    TensorOps = TensorOps,
    TensorValue = TensorValue,
    
    
    zeros = TensorOps.zeros,
    ones = TensorOps.ones,
    randn = TensorOps.randn,
    eye = TensorOps.eye,
    full = TensorOps.full,
    
    
    map = TensorOps.map,
    zip = TensorOps.zip,
    sum = TensorOps.sum,
    mean = TensorOps.mean,
    add = TensorOps.add,
    sub = TensorOps.sub,
    mul = TensorOps.mul,
    matmul = TensorOps.matmul,
    
    
    relu = TensorOps.relu,
    sigmoid = TensorOps.sigmoid,
    tanh = TensorOps.tanh,
    softmax = TensorOps.softmax,
    
    
    mse_loss = TensorOps.mse_loss,
}