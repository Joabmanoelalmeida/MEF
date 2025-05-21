import numpy as np
from scipy.linalg import eigh

def calcula_dt_critico(K, M):
    # Cálculo dos autovalores (generalized eigenvalue problem)
    # K * phi = lambda * M * phi
    lambdas, _ = eigh(K, M)
    
    # Remove autovalores negativos ou muito próximos de zero (se houver)
    lambdas = lambdas[lambdas > 1e-12]
    
    # Calcula as frequências naturais
    omegas = np.sqrt(lambdas)
    
    # Frequência natural mais alta
    omega_max = np.max(omegas)
    
    # Passo de tempo crítico
    dt_crit = 2.0 / omega_max
    
    return dt_crit


def Newmark(dt, t0, tf, u0, v0, M, K, F, gamma=0.5, beta=0.25):
    """
    Algoritmo de Newmark para integração temporal de sistemas dinâmicos:
    
    M * u'' + K * u = F(t)
    
    Parâmetros:
    - dt: passo de tempo
    - t0: tempo inicial
    - tf: tempo final
    - u0: deslocamento inicial (array)
    - v0: velocidade inicial (array)
    - M: matriz de massa
    - K: matriz de rigidez
    - F: função que retorna o vetor de forças em t
    - gamma, beta: parâmetros do método (default = 0.5, 0.25 -> Newmark médio, não dissipativo)
    
    Retorna:
    - t: vetor de tempos
    - u: histórico de deslocamento
    - v: histórico de velocidade
    - a: histórico de aceleração
    """
    
    # Número de passos de tempo
    N = int(np.floor((tf - t0) / dt)) + 1
    t = np.linspace(t0, tf, N)
    
    ndofs = K.shape[0]
    
    # Inicialização dos vetores
    u = np.zeros((ndofs, N))
    v = np.zeros((ndofs, N))
    a = np.zeros((ndofs, N))
    
    # Pré-cálculo de coeficientes
    c1 = 1.0 / (beta * dt**2)
    c2 = 1.0 / (beta * dt)
    c3 = 1.0 / (2 * beta) - 1
    c4 = gamma / (beta * dt)
    c5 = gamma / beta - 1
    c6 = dt * (gamma / (2 * beta) - 1)
    
    # Matriz de rigidez efetiva
    Keff = c1 * M + K #  + c4*C 
    Keffinv = np.linalg.inv(Keff)
    
    # Condições iniciais
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = np.linalg.solve(M, F(t[0]) - K @ u0)  # - C @ v0 (termo de amortecimento comentado)
    
    # Algoritmo de Newmark
    for i in range(N - 1):
        # Força efetiva
        feff = (F(t[i + 1]) + 
                M @ (c1 * u[:, i] + c2 * v[:, i] + c3 * a[:, i]))
                # + C @ (c4 * u[:, i] + c5 * v[:, i] + c6 * a[:, i])  # Se desejar incluir amortecimento
        
        # Deslocamento
        u[:, i + 1] = Keffinv @ feff
        
        # Aceleração
        a[:, i + 1] = (c1 * (u[:, i + 1] - u[:, i] - dt * v[:, i])) - c3 * a[:, i]
        
        # Velocidade
        v[:, i + 1] = (c4 * (u[:, i + 1] - u[:, i])) - c5 * v[:, i] - c6 * a[:, i]
    
    return t, u, v, a

# Testando o código
if __name__ == "__main__":
    # Definindo parâmetros
    m1 = 1.0
    m2 = 1.0
    k1 = 1.0
    k2 = 10.0
    k3 = 1.0
    
    # Definindo a força externa como uma função do tempo
    F = lambda t: np.array([0, 0])
    # F = lambda t: np.array([0, 0]) if t < 10 else np.array([0, 1])

    # Condições iniciais
    u0 = np.array([0, 0])
    v0 = np.array([1, 0])
    
    # Intervalo de tempo
    t0 = 0.0
    tf = 84.0
    
    # Criando matrizes
    K = np.array([[(k1+k2), -k2],
                    [-k2, (k2+k3)]])
    M = np.array([[m1, 0],
                    [0, m2]])
    
    # Cálculo do passo de tempo crítico
    dt_crit = calcula_dt_critico(K, M)
    
    # Passo de tempo (usando o passo crítico)
    dt = dt_crit*0.01

    # Executando o método de Newmark
    t, u, v, a = Newmark(dt, t0, tf, u0, v0, M, K, F)

    # Exibindo resultados
    print("Passo de tempo crítico:", dt_crit)
    print("Passo de tempo utilizado:", dt)
    
    # Plotando resultados
    import matplotlib.pyplot as plt

    
    uAnalitic1 = (np.sin(t) + np.sin(t*np.sqrt(21))/np.sqrt(21))/ 2
    uAnalitic2 = (np.sin(t) - np.sin(t*np.sqrt(21))/np.sqrt(21))/ 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, uAnalitic1, label='u1 (Analítico)')
    plt.plot(t, u[0, :], label='u1 (Deslocamento 1)', linestyle='--')
    plt.plot(t, uAnalitic2, label='u2 (Analítico)')
    plt.plot(t, u[1, :], label='u2 (Deslocamento 2)', linestyle='--')
    plt.title('Deslocamentos ao longo do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Deslocamento (m)')
    plt.legend()
    plt.grid()
    plt.show()
    
