
local micrograd = require("micrograd")
local tensorlib = require("tensor")


print("="..string.rep("=", 50))

-- ====================================================
-- ✅ 1. MICROGRAD BÁSICO (Escalares) - CORRETO
-- ====================================================
function demo_micrograd_basico()
    print("\n✅ 1. MICROGRAD ESCALAR (CORRETO)")
    print("-"..string.rep("-", 40))
    
    local a = micrograd.Value(2.0)
    local b = micrograd.Value(-3.0)
    local c = micrograd.Value(10.0)

    local d = a * b
    local e = d + c
    local f = e:relu()

    print(string.format("a=%5.1f, b=%5.1f, c=%5.1f", a.data, b.data, c.data))
    print(string.format("a*b = %5.1f", d.data))
    print(string.format("(a*b)+c = %5.1f", e.data))
    print(string.format("ReLU((a*b)+c) = %5.1f", f.data))

    f:backward()
    
    print("\n📊 Gradientes calculados:")
    print(string.format("∂f/∂a = %5.1f", a.grad))
    print(string.format("∂f/∂b = %5.1f", b.grad))
    print(string.format("∂f/∂c = %5.1f", c.grad))
end

-- ====================================================
-- ✅ 2. PORTA AND - CORRETO
-- ====================================================
function demo_porta_and()
    print("\n✅ 2. NEURÔNIO APRENDENDO PORTA AND")
    print("-"..string.rep("-", 40))
    
    local X = {{0,0}, {0,1}, {1,0}, {1,1}}
    local y = {0, 0, 0, 1}
    
    local neuron = micrograd.Neuron.new(2, 'sigmoid')
    local optimizer = micrograd.SGD.new(neuron:parameters(), 0.5)
    
    print("Antes do treino:")
    for i, inputs in ipairs(X) do
        local pred = neuron(inputs)
        print(string.format("  [%d, %d] → %5.3f", inputs[1], inputs[2], pred.data))
    end
    
    -- Treinamento CORRETO
    for epoch = 1, 500 do
        local total_loss = 0
        
        for i = 1, #X do
            local pred = neuron(X[i])
            local target = micrograd.Value(y[i])
            local loss = (pred - target) * (pred - target)
            total_loss = total_loss + loss.data
            
            -- ORDEM CORRETA
            optimizer:zero_grad()
            neuron:zero_grad()
            loss:backward()
            optimizer:step()
        end
        
        if epoch % 100 == 0 then
            print(string.format("  Época %3d: Loss = %.6f", epoch, total_loss/#X))
        end
    end
    
    print("\nDepois do treino:")
    for i, inputs in ipairs(X) do
        local pred = neuron(inputs)
        local result = pred.data > 0.5 and 1 or 0
        print(string.format("  [%d, %d] → %5.3f → %d %s", 
            inputs[1], inputs[2], pred.data, result, 
            result == y[i] and "✓" or "✗"))
    end
end

-- ====================================================
-- ✅ 3. TREINAMENTO LINEAR SIMPLES (TensorValue)
-- ====================================================
function demo_tensorvalue_linear()
    print("\n✅ 3. TREINAMENTO LINEAR (TensorValue)")
    print("-"..string.rep("-", 40))
    
    -- Dados: y = 2x + 1 + ruído
    local X_data = {1, 2, 3, 4, 5}
    local y_data = {3.1, 4.9, 7.2, 8.8, 11.1}  -- ≈2x+1
    
    -- Parâmetros treináveis
    local w = tensorlib.TensorValue.new({math.random()}, {1}, true)
    local b = tensorlib.TensorValue.new({math.random()}, {1}, true)
    
    local learning_rate = 0.01
    
    print(string.format("Inicial: w=%.4f, b=%.4f", 
        w.tensor.data[1], b.tensor.data[1]))
    
    for epoch = 1, 1000 do
        local total_loss = 0
        
        for i = 1, #X_data do
            local x = tensorlib.TensorValue.new({X_data[i]}, {1}, true)
            local y_true = tensorlib.TensorValue.new({y_data[i]}, {1}, true)
            
            -- Forward pass
            local y_pred = x * w + b
            local diff = y_pred - y_true
            local loss = diff * diff
            
            total_loss = total_loss + loss.tensor.data[1]
            
            -- Backward pass (TensorValue)
            tensorlib.TensorValue.zero_grad({w, b, x, y_pred, diff, loss})
            loss:backward()
            
            -- Update
            w.tensor.data[1] = w.tensor.data[1] - learning_rate * w.grad.data[1]
            b.tensor.data[1] = b.tensor.data[1] - learning_rate * b.grad.data[1]
        end
        
        if epoch % 200 == 0 then
            print(string.format("  Época %4d: Loss=%.6f, w=%.4f, b=%.4f",
                epoch, total_loss/#X_data, w.tensor.data[1], b.tensor.data[1]))
        end
    end
    
    print(string.format("\nModelo treinado: y = %.4f*x + %.4f", 
        w.tensor.data[1], b.tensor.data[1]))
    
    -- Teste
    print("\nTeste do modelo:")
    for i, x in ipairs(X_data) do
        local pred = w.tensor.data[1] * x + b.tensor.data[1]
        print(string.format("  x=%d: pred=%.2f, real=%.2f, erro=%.2f",
            x, pred, y_data[i], math.abs(pred - y_data[i])))
    end
end

-- ====================================================
-- ⚠️ 4. ESTATÍSTICA (corrigido)
-- ====================================================
function estatistica_correta(dados)
    print("\n⚠️ 4. ESTATÍSTICA (VERSÃO CORRIGIDA)")
    print("-"..string.rep("-", 40))
    
    -- Usando number puro, NÃO micrograd
    local sum = 0
    local min_val = math.huge
    local max_val = -math.huge
    
    for i = 1, #dados do
        sum = sum + dados[i]
        min_val = math.min(min_val, dados[i])
        max_val = math.max(max_val, dados[i])
    end
    
    local media = sum / #dados
    
    -- Variância correta
    local soma_quadrados = 0
    for i = 1, #dados do
        local diff = dados[i] - media
        soma_quadrados = soma_quadrados + diff * diff
    end
    
    local variancia = soma_quadrados / #dados
    local desvio_padrao = math.sqrt(variancia)
    
    print("Dados:", table.concat(dados, ", "))
    print(string.format("N=%d, Mín=%.1f, Máx=%.1f", #dados, min_val, max_val))
    print(string.format("Média=%.4f", media))
    print(string.format("Variância=%.4f", variancia))
    print(string.format("Desvio padrão=%.4f", desvio_padrao))
    
    return {
        media = media,
        variancia = variancia,
        desvio_padrao = desvio_padrao
    }
end

-- ====================================================
-- ⚠️ 5. MONTE CARLO PI (corrigido)
-- ====================================================
function monte_carlo_pi_corrigido(samples)
    print("\n⚠️ 5. MONTE CARLO PI (VERSÃO CORRIGIDA)")
    print("-"..string.rep("-", 40))
    
    math.randomseed(os.time())  -- Melhor semente
    
    local inside_circle = 0
    local estimates = {}
    
    print(string.format("🎲 Estimando π com %d amostras", samples))
    
    for i = 1, samples do
        -- Geração correta: [0, 1) não [-1, 1]
        local x = math.random()  -- 0 a 1
        local y = math.random()  -- 0 a 1
        
        -- Distância da origem
        if x*x + y*y <= 1.0 then
            inside_circle = inside_circle + 1
        end
        
        if i % (samples/10) == 0 then
            local pi_estimate = 4 * inside_circle / i
            table.insert(estimates, pi_estimate)
        end
    end
    
    local final_pi = 4 * inside_circle / samples
    local erro_percentual = math.abs(final_pi - math.pi) / math.pi * 100
    
    print(string.format("\n✅ Estimativa final: π ≈ %.6f", final_pi))
    print(string.format("   Valor real:     π = %.6f", math.pi))
    print(string.format("   Erro:           %.4f%%", erro_percentual))
    
    return final_pi
end

-- ====================================================
-- ⚠️ 6. MINIMIZAÇÃO DE FUNÇÃO (corrigido)
-- ====================================================
function minimizacao_correta()
    print("\n⚠️ 6. MINIMIZAÇÃO (VERSÃO CORRIGIDA)")
    print("-"..string.rep("-", 40))
    
    -- Função: f(x) = x⁴ - 3x³ + 2
    -- Derivada: f'(x) = 4x³ - 9x²
    -- Mínimo local em x ≈ 2.25
    
    local x = micrograd.Value(0.5)  -- Ponto inicial melhor
    local optimizer = micrograd.SGD.new({x}, 0.01)
    
    print("Encontrando mínimo de f(x) = x⁴ - 3x³ + 2")
    print(string.format("Ponto inicial: x = %.2f", x.data))
    
    for i = 1, 200 do
        -- Forward
        local f = x*x*x*x - micrograd.Value(3)*x*x*x + micrograd.Value(2)
        
        -- Backward (ORDEM CORRETA)
        optimizer:zero_grad()
        f:backward()
        optimizer:step()
        
        if i % 40 == 0 then
            print(string.format("  It. %3d: x = %7.4f, f(x) = %7.4f, f'(x) = %7.4f",
                i, x.data, f.data, x.grad))
        end
    end
    
    -- Verificação numérica
    local h = 1e-4
    local f_x = x.data^4 - 3*x.data^3 + 2
    local derivada_numerica = ( (x.data+h)^4 - 3*(x.data+h)^3 + 2 - f_x ) / h
    
    print(string.format("\n✅ Mínimo em x ≈ %.4f", x.data))
    print(string.format("   f(%.4f) = %.4f", x.data, f_x))
    print(string.format("   f'(%.4f) ≈ %.6f (deveria ser ≈ 0)", x.data, derivada_numerica))
    
    if math.abs(derivada_numerica) < 0.01 then
        print("   ✓ Derivada próxima de zero (mínimo local)")
    end
end

-- ====================================================
-- ❌ 7. CRESCIMENTO POPULACIONAL (simplificado/correto)
-- ====================================================
function crescimento_populacional_simplificado()
    print("\n❌ 7. CRESCIMENTO LOGÍSTICO (SIMPLIFICADO)")
    print("-"..string.rep("-", 40))
    
    -- Modelo discreto simples, SEM autograd
    local P = 10      -- População inicial
    local r = 1.5     -- Taxa de crescimento
    local K = 100     -- Capacidade de suporte
    
    print(string.format("Modelo: Pₜ₊₁ = r * Pₜ * (1 - Pₜ/K)"))
    print(string.format("Inicial: P=%d, r=%.1f, K=%d", P, r, K))
    
    local history = {P}
    
    for t = 1, 20 do
        P = r * P * (1 - P/K)
        table.insert(history, P)
        
        if t <= 10 then
            print(string.format("  t=%2d: P=%6.1f", t, P))
        end
    end
    
    print(string.format("\nPopulação final: %.1f", P))
    print(string.format("Equilíbrio teórico: K*(r-1)/r = %.1f", K*(r-1)/r))
    
    return history
end

-- ====================================================
-- ✅ 8. AUTOCOMPLETE (funcional)
-- ====================================================
function sistema_autocomplete_refatorado()
    print("\n✅ 8. SISTEMA DE AUTOCOMPLETE")
    print("-"..string.rep("-", 40))
    
    local database = {
        "ola", "como", "esta", "voce", "bem", "mal", "sim", "nao",
        "gosto", "programar", "lua", "game", "guardian", "script",
        "funcao", "variavel", "valor", "texto", "palavra", "lista"
    }
    
    local function autocomplete(input, max_results)
        input = input:lower()
        local matches = {}
        
        for _, word in ipairs(database) do
            if word:sub(1, #input) == input then
                table.insert(matches, word)
            end
        end
        
        table.sort(matches)
        
        if #matches > max_results then
            for i = max_results+1, #matches do
                matches[i] = nil
            end
        end
        
        return matches
    end
    
    local testes = {"pro", "gam", "val", "fun", "tex"}
    
    for _, teste in ipairs(testes) do
        local sugestoes = autocomplete(teste, 3)
        if #sugestoes > 0 then
            print(string.format("'%s' → %s", teste, table.concat(sugestoes, ", ")))
        else
            print(string.format("'%s' → (sem sugestões)", teste))
        end
    end
end

-- ====================================================
-- 🚀 EXECUÇÃO PRINCIPAL
-- ====================================================

-- Executar demos na ordem
demo_micrograd_basico()
demo_porta_and()
demo_tensorvalue_linear()

-- Estatística com dados reais
local temperaturas = {22.5, 23.1, 21.8, 24.3, 22.9, 23.7, 21.5}
estatistica_correta(temperaturas)

-- Monte Carlo
monte_carlo_pi_corrigido(10000)

-- Minimização
minimizacao_correta()

-- Crescimento populacional
crescimento_populacional_simplificado()

-- Autocomplete
sistema_autocomplete_refatorado()

-- ====================================================
-- 🔧 FUNÇÕES DE UTILIDADE SEPARADAS
-- ====================================================

function verificar_compatibilidade()
    print("\n🔧 VERIFICAÇÃO DE COMPATIBILIDADE")
    print("-"..string.rep("-", 40))
    
    -- Teste MicroGrad
    local v1 = micrograd.Value(5)
    local v2 = v1 * micrograd.Value(2)
    v2:backward()
    print(string.format("MicroGrad: 5*2=%.1f, grad=%.1f ✓", v2.data, v1.grad))
    
    -- Teste TensorValue
    local success, tv1 = pcall(function()
        return tensorlib.TensorValue.new({1,2,3}, {3}, true)
    end)
    
    if success and tv1 then
        print("TensorValue: criação OK ✓")
        
        local success2, tv2 = pcall(function()
            local tv2 = tensorlib.TensorValue.new({4,5,6}, {3}, true)
            return tv1 + tv2
        end)
        
        if success2 then
            print("TensorValue: operações OK ✓")
        else
            print("TensorValue: operações FALHOU ✗")
        end
    else
        print("TensorValue: criação FALHOU ✗")
    end
    
    print("\n💡 RECOMENDAÇÕES:")
    print("1. Use MicroGrad para escalares e redes pequenas")
    print("2. Use TensorValue para operações tensoriais")
    print("3. NUNCA misture os dois sistemas")
    print("4. Para estatística, use number puro")
end

verificar_compatibilidade()

print("\n"..string.rep("=", 50))
