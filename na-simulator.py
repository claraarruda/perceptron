import random

class Perceptron:

#self = rede neural

    def __init__(self, samples, exits, learning_rate =0.1, time=1000, limiar=-1):
        self.samples = samples #amostras
        self.exits = exits #saidas
        self.learning_rate = learning_rate #taxa de aprendizado
        self.time = time
        self.limiar = limiar
        self.n_samples = len(samples)
        self.n_attributes = len(samples[0]) #atributos entrada
        self.weight = [] #peso

    def sinal(self, u ):
        if (u >= 0):
            return  1
        else:
            return 0
    
    def train(self): #treina a rede
         # Inserir o valor do limiar na posição "0" para cada amostra da lista "samples"
        for sample in self.samples: 
            sample.insert(0, -1)

        # Gerar valores randômicos entre 0 e 1 (pesos) conforme o número de atributos
        for i in range(self.n_attributes):
            self.weight.append(random.random())
        # Inserir o valor posição "0" do vetor de pesos
        self.weight.insert(0, self.limiar)
        
        #inicializa contador tempo
        n_time =0

        while True:

            # Inicializar variável erro
			# (quando terminar loop e erro continuar False, é pq não tem mais diferença entre valor calculado e desejado)
            error = False 

            #para cada amostra...
            for i in range(self.n_samples):
                #inicializar potencial de ativaçao
                u = 0
                #para cada atributo...
                for j in range(self.n_attributes + 1):
                    # Multiplicar amostra e seu peso e também somar com o potencial que já tinha
                    u += self.weight[j] * self.samples[i][j]
                # Obter a saída da rede considerando g a função sinal
                y = self.sinal(u)

            # Verificar se a saída da rede é diferente da saída desejada
                if (y != self.exits[i]): 
                    # Calcular o erro                   
                    error_aux = self.exits[i] - y
                    # Fazer o ajuste dos pesos para cada elemento da amostra                    
                    for j in range(self.n_attributes + 1):
                        self.weight[j] = self.weight[j] + self.learning_rate * error_aux * self.samples[i][j]
                    # Atualizar variável erro, já que erro é diferente de zero (existe)
                    error = True 
            # Atualizar contador
            n_time += 1 

            # Critérios de parada do loop: erro inexistente ou o número de épocas ultrapassar limite pré-estabelecido
            if not error or n_time > self.time:
                break
    
    # Testes para "novas" amostras
    def test(self, sample):
        # Inserir o valor do bias na posição "0" para cada amostra da lista "samples"
        sample.insert(0, -1)
        # Inicializar potencial de ativação
        u = 0
        #para cada atributo...
        for i in range(self.n_attributes + 1):
            u+= self.weight[i] * sample[i]

        #obter a saida da rede considerando G a funçao sinal
        y = self.sinal(u)
        print("Result : %d " %y)

# Amostras (entrada e saída) para treinamento
samples = [[0,0], [0,1], [1,0], [1,1]]
exits_or = [0, 1, 1, 1]
exits_and = [0, 0, 0, 1]

entry = int(input("Escolha a rede: 1- OR / 2- AND: "))

# Entrando com amostra para teste
if (entry==1):
    rede = Perceptron(samples, exits_or)
    rede.train()
    rede.test([0, 0])
    rede.test([0, 1])
    rede.test([1, 0])
    rede.test([1, 1])


elif (entry==2):
    rede = Perceptron(samples, exits_and)
    rede.train()
    rede.test([0, 0])
    rede.test([0, 1])
    rede.test([1, 0])
    rede.test([1, 1])