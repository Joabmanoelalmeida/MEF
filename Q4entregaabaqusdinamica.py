import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.linalg import eigh

def obter_face_nodes(nodes_e: List[int], face: str) -> List[int]:
    n = len(nodes_e)

    if n == 3:  # T3
        if face == 'S1':
            return [nodes_e[0], nodes_e[1]]
        elif face == 'S2':
            return [nodes_e[1], nodes_e[2]]
        elif face == 'S3':
            return [nodes_e[2], nodes_e[0]]
        else:
            raise ValueError(f"Face invalida para T3: {face}")

    elif n == 4:  # Q4
        if face == 'S1':
            return [nodes_e[0], nodes_e[1]]
        elif face == 'S2':
            return [nodes_e[1], nodes_e[2]]
        elif face == 'S3':
            return [nodes_e[2], nodes_e[3]]
        elif face == 'S4':
            return [nodes_e[3], nodes_e[0]]
        else:
            raise ValueError(f"Face invalida para Q4: {face}")

    elif n == 8:  # Q8
        if face == 'S1':
            return [nodes_e[0], nodes_e[1], nodes_e[4]]
        elif face == 'S2':
            return [nodes_e[1], nodes_e[2], nodes_e[5]]
        elif face == 'S3':
            return [nodes_e[2], nodes_e[3], nodes_e[6]]
        elif face == 'S4':
            return [nodes_e[3], nodes_e[0], nodes_e[7]]
        else:
            raise ValueError(f"Face invalida para Q8: {face}")
    else:
        raise ValueError(f"Elemento com numero de nos nao suportado: {n}")

def funcao_formaQ4(n, e):
    N14Q = 0.25*(1 - e)*(1 - n)
    N24Q = 0.25*(1 + e)*(1 - n)
    N34Q = 0.25*(1 + e)*(1 + n)
    N44Q = 0.25*(1 - e)*(1 + n)
    return [N14Q, N24Q, N34Q, N44Q]

def funcao_forma_derivada(n, e):
    N = funcao_formaQ4(n, e)
    dN_dn = [sp.diff(Ni, n) for Ni in N]
    dN_de = [sp.diff(Ni, e) for Ni in N]
    return dN_dn, dN_de

def matrix_jacobiano(n, e, coords):
    dN_dn, dN_de = funcao_forma_derivada(n, e)
    dN_nat = sp.Matrix([dN_de, dN_dn])
    J = dN_nat * coords
    
    detJ = J.det()
    if detJ == 0:
        raise ValueError("Determinante da jacobiana é zero ou muito pequeno.")

    invJ = J.inv()

    return J, detJ, invJ

def jacobiano_gauss(n_val, e_val, coords):
    n, e = sp.symbols('n e')
    J, detJ, invJ = matrix_jacobiano(n, e, coords)
    J_gauss = J.subs({n: n_val, e: e_val})
    detJ_gauss = detJ.subs({n: n_val, e: e_val})
    invJ_gauss = invJ.subs({n: n_val, e: e_val})
    return J_gauss, detJ_gauss, invJ_gauss

def ji(n, e):
    _, dN_de = funcao_forma_derivada(n, e)
    dN_dn, _ = funcao_forma_derivada(n, e)
    Ji = sp.Matrix([dN_de, dN_dn])
    return Ji

def ji_gauss(n_val, e_val):
    n, e = sp.symbols('n e')
    _, dN_de = funcao_forma_derivada(n, e)
    dN_dn, _ = funcao_forma_derivada(n, e)
    Ji = sp.Matrix([dN_de, dN_dn])
    return Ji.subs({n: n_val, e: e_val})

def matrix_Be(n, e, coords):
    dN_global = derivadas_global(n, e, coords)
    dx = [item[0] for item in dN_global]
    dy = [item[1] for item in dN_global]
    
    Be = sp.Matrix([
        [dx[0], 0,   dx[1], 0,   dx[2], 0,   dx[3], 0],
        [0,   dy[0], 0,   dy[1], 0,   dy[2], 0,   dy[3]],
        [dy[0], dx[0], dy[1], dx[1], dy[2], dx[2], dy[3], dx[3]]
    ])
    return Be

def derivadas_global(n, e, coords):
    _, _, invJ = matrix_jacobiano(n, e, coords)
    dN_dn, dN_de = funcao_forma_derivada(n, e)
    dN_global = []
    for dnde, dnd in zip(dN_de, dN_dn):
        dN_nat_vec = sp.Matrix([dnde, dnd])
        dxy = invJ * dN_nat_vec
        dN_global.append((dxy[0], dxy[1]))
    return dN_global

def matrix_constitutiva(E, nu):
    D = E/(1 - nu**2) * sp.Matrix([
        [1,   nu,            0],
        [nu,  1,             0],
        [0,   0,  (1 - nu)/2]
    ])
    return D

def matrix_Ke_global_elemento(coords, n, e, D, W, gauss_pontos):
    Ke_total = sp.zeros(8, 8)
    for pt in gauss_pontos:
        n_val, e_val = pt
        Be_numerico = matrix_Be(n, e, coords).subs({n: n_val, e: e_val})
        _, detJ_gauss, _ = jacobiano_gauss(n_val, e_val, coords)
        Ke = W*W*(Be_numerico.T * D * Be_numerico * detJ_gauss)*espessura
        Ke_total += Ke
    return Ke_total

def matrix_Ke_reduzido_elemento(K, constrained_dofs):
    total_no = list(range(K.shape[0]))
    novo_no = [dof for dof in total_no if dof not in constrained_dofs]
    K_reduced = K.extract(novo_no, novo_no)
    return K_reduced, novo_no

def deslocamento_reduzido(Ke_reduced, F_reduced):
    try:
        u_reduced = Ke_reduced.LUsolve(F_reduced)
    except Exception:
        u_reduced = Ke_reduced.pinv() * F_reduced
    return u_reduced

def deslocamento_global(u_reduced, novo_no, total_dofs):
    u_global = sp.zeros(total_dofs, 1)
    for i, dof in enumerate(novo_no):
        u_global[dof] = u_reduced[i]
    return u_global

def deformacao_gauss(u_local, n, e, coords, gauss_points):
    defor = []
    for n_val, e_val in gauss_points:
        Be_numerico = matrix_Be(n, e, coords).subs({n: n_val, e: e_val})
        tens = Be_numerico * u_local
        defor.append(tens)
    return defor 

def deformacao_gauss_reduzido(u_local, n, e, coords, gauss_points_reduzido):
    defor = []
    for n_val, e_val in gauss_points_reduzido:
        Be_numerico = matrix_Be(n, e, coords).subs({n: n_val, e: e_val})
        tens = Be_numerico * u_local
        defor.append(tens)
    return defor

def deformacao_nos_reduzido(u_local, n, e, coords, gauss_points_reduzido):
    
    defor = deformacao_gauss_reduzido(u_local, n, e, coords, gauss_points_reduzido)
    
    dx_gauss = sp.Matrix([defor[0] for defor in defor])
    dy_gauss = sp.Matrix([defor[1] for defor in defor])
    dxy_gauss = sp.Matrix([defor[2] for defor in defor])

    defor_vec = sp.Matrix([1,1,1,1])
    
    dx_nos = defor_vec * dx_gauss
    dy_nos = defor_vec * dy_gauss
    dxy_nos = defor_vec * dxy_gauss
    
    defor_nos_reduzido = sp.Matrix.hstack(dx_nos, dy_nos, dxy_nos)

    return defor_nos_reduzido

def tensoes_gauss(u_local, n, e, coords, gauss_points, D):
    strains = deformacao_gauss(u_local, n, e, coords, gauss_points)
    stresses = []
    for strain in strains:
        stress = D * strain
        stresses.append(stress)
    return stresses

def tensoes_gauss_reduzido(u_local, n, e, coords, gauss_points_reduzido, D):
    strains = deformacao_gauss(u_local, n, e, coords, gauss_points_reduzido)
    stresses = []
    for strain in strains:
        stress = D * strain
        stresses.append(stress)
    return stresses

def matriz_E():
    a, b = 1 + sp.sqrt(3), 1 - sp.sqrt(3)
    E = sp.Matrix([
                [a/4, a/4, b/4, b/4],
                [b/4, a/4, a/4, b/4],
                [b/4, b/4, a/4, a/4],
                [a/4, b/4, b/4, a/4]
            ])
    return E

def tensoes_nos(u_local, n, e, coords, gauss_points, D):

    t_gauss = tensoes_gauss(u_local, n, e, coords, gauss_points, D)

    tx_gauss = sp.Matrix([stress[0] for stress in t_gauss])
    ty_gauss = sp.Matrix([stress[1] for stress in t_gauss])
    txy_gauss = sp.Matrix([stress[2] for stress in t_gauss])

    tx_nos = matriz_E() * tx_gauss
    ty_nos = matriz_E() * ty_gauss
    txy_nos = matriz_E() * txy_gauss
    
    t_nos = sp.Matrix.hstack(tx_nos, ty_nos, txy_nos).evalf()
    return t_nos

def tensoes_nos_reduzido(u_local, n, e, coords, gauss_pontos_reduzido, D):

    t_gauss = tensoes_gauss_reduzido(u_local, n, e, coords, gauss_pontos_reduzido, D)

    tx_gauss = sp.Matrix([stress[0] for stress in t_gauss])
    ty_gauss = sp.Matrix([stress[1] for stress in t_gauss])
    txy_gauss = sp.Matrix([stress[2] for stress in t_gauss])

    ones_vec = sp.Matrix([1,1,1,1])
    
    tx_nos = ones_vec * tx_gauss
    ty_nos = ones_vec * ty_gauss
    txy_nos = ones_vec * txy_gauss
    
    t_nos = sp.Matrix.hstack(tx_nos, ty_nos, txy_nos)

    return t_nos

def matriz_Ke_global(global_coords, elements, n, e, D, W, gauss_points):
    total_dofs = global_coords.shape[0] * 2
    K_global = sp.zeros(total_dofs, total_dofs)
    for elem in elements:
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        Ke_local = matrix_Ke_global_elemento(local_coords, n, e, D, W, gauss_points)
        for i_local, global_node in enumerate(elem):
            dofs_i = [global_node*2, global_node*2+1]
            for j_local, global_node_j in enumerate(elem):
                dofs_j = [global_node_j*2, global_node_j*2+1]
                K_global[dofs_i[0], dofs_j[0]] += Ke_local[2*i_local, 2*j_local]
                K_global[dofs_i[0], dofs_j[1]] += Ke_local[2*i_local, 2*j_local+1]
                K_global[dofs_i[1], dofs_j[0]] += Ke_local[2*i_local+1, 2*j_local]
                K_global[dofs_i[1], dofs_j[1]] += Ke_local[2*i_local+1, 2*j_local+1]
    return K_global

def matrix_Me_global_elemento(coords, n, e, rho, espessura, W, gauss_pontos):

    N_funcs = funcao_formaQ4(n, e)
    npts = len(N_funcs)
    ngl = 2 * npts
    Me = sp.zeros(ngl, ngl)
    for (n_val, e_val) in gauss_pontos:

        N_eval = [Ni.subs({n: n_val, e: e_val}) for Ni in N_funcs]

        Nmat = sp.zeros(2, ngl)
        for k, Nk in enumerate(N_eval):
            Nmat[0, 2*k]   = Nk
            Nmat[1, 2*k+1] = Nk

        _, detJ, _ = jacobiano_gauss(n_val, e_val, coords)
        Me += rho * (Nmat.T * Nmat) * detJ * W * W * espessura
    return Me

def matriz_Me_global(global_coords, elements, n, e, rho, espessura, W, gauss_pontos):

    total_dofs = global_coords.shape[0] * 2
    M_global = sp.zeros(total_dofs, total_dofs)
    for elem in elements:
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        Me_local = matrix_Me_global_elemento(local_coords, n, e, rho, espessura, W, gauss_pontos)
        # assemble
        for i_local, gi in enumerate(elem):
            dofs_i = [gi*2, gi*2+1]
            for j_local, gj in enumerate(elem):
                dofs_j = [gj*2, gj*2+1]
                M_global[dofs_i[0], dofs_j[0]] += Me_local[2*i_local,   2*j_local]
                M_global[dofs_i[0], dofs_j[1]] += Me_local[2*i_local,   2*j_local+1]
                M_global[dofs_i[1], dofs_j[0]] += Me_local[2*i_local+1, 2*j_local]
                M_global[dofs_i[1], dofs_j[1]] += Me_local[2*i_local+1, 2*j_local+1]
    return M_global


def matrix_Me_reduzido(M_global, constrained_dofs):
    total_dofs = list(range(M_global.shape[0]))
    novo_dofs = [dof for dof in total_dofs if dof not in constrained_dofs]
    M_reduzido = M_global.extract(novo_dofs, novo_dofs)
    return M_reduzido

def aplicar_forca_face(f: np.ndarray, face_nodes: List[int], intensidade: float,
                         dir: Tuple[float, float], coords: np.ndarray) -> None:
    n_nodes = len(face_nodes)
    if n_nodes == 3:
        n1, n2, n3 = face_nodes
    elif n_nodes == 2:
        n1, n2 = face_nodes
    else:
        raise ValueError(f"Face com numero de nos nao suportado: {n_nodes}")

    x1, y1 = coords[n1]
    x2, y2 = coords[n2]

    coef = 2 * (n_nodes - 1)
    comprimento = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    f_local = (intensidade * comprimento) / coef

    f[2 * n1] += f_local * dir[0]
    f[2 * n1 + 1] += f_local * dir[1]
    f[2 * n2] += f_local * dir[0]
    f[2 * n2 + 1] += f_local * dir[1]

    if n_nodes == 3:
        f[2 * n3] += 2 * f_local * dir[0]
        f[2 * n3 + 1] += 2 * f_local * dir[1]

def normalize_vector(vx: float, vy: float) -> Tuple[float, float]:
    norm = np.sqrt(vx**2 + vy**2)
    if norm == 0:
        raise ValueError("Vector norm is zero.")
    return vx / norm, vy / norm

def calcula_dt_critico(K, M):

    K_np = np.array(K.tolist(), dtype=float)
    M_np = np.array(M.tolist(), dtype=float)
    lambdas, _ = eigh(K_np, M_np)
    
    lambdas = lambdas[lambdas > 1e-12]

    omegas = np.sqrt(lambdas)

    omega_max = np.max(omegas)

    dt_crit = 2.0 / omega_max
    
    return dt_crit

def Newmark(dt, t0, tf, u0, v0, M, K, F, gamma=0.5, beta=0.25):
    
    N = int(np.floor((tf - t0) / dt)) + 1
    t = np.linspace(t0, tf, N)
    
    ndofs = K.shape[0]
    M = np.array(M.tolist(), dtype=float)
    K = np.array(K.tolist(), dtype=float)
    
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
    Keff_np = np.array(Keff.tolist(), dtype=float)
    Keffinv = np.linalg.inv(Keff_np)
    
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

def ler_inp(path: str):
    coords = []
    conn = []
    Elist = []
    nulist = []
    rholist = []
    esp = 1.0
    plane_state = 'stress'
    gdl_restritos = []
    f = []

    nodes = {}
    elconn = {}
    nsets = {}
    elsets = {}
    surfaces = {}
    dsloads = {}

    E = nu = 0.0
    rho = 7850.0

    with open(path, 'r') as file:
        linhas = file.readlines()

    in_part = in_nodes = in_elements = False
    in_elastic_block = in_section = in_boundary = in_cload = False
    in_surface = in_nset = in_elset = in_dsload = False
    in_assembly = False
    current_set = ''

    for i, linha in enumerate(linhas):
        l = linha.strip()
        if l == '' or l.startswith('**'):
            continue

        if l.startswith('*'):
            in_boundary = in_cload = in_surface = in_nset = in_elset = False
            in_nodes = in_elements = in_section = False

            l_lower = l.lower()

            if '*surface' in l_lower and in_assembly:
                in_surface = True
                current_set = [p.split('=')[-1].strip() for p in l.split(',') if 'name=' in p.lower()][0]
                surfaces[current_set] = ('', '')
                continue
            if '*assembly' in l_lower:
                in_assembly = True
                continue
            if '*end assembly' in l_lower:
                in_assembly = False
                continue

            if '*nset' in l_lower and in_assembly:
                in_nset = True
                current_set = [p.split('=')[-1].strip() for p in l.split(',') if 'nset=' in p.lower()][0]
                nsets[current_set] = []
                continue
            if '*elset' in l_lower and in_assembly:
                in_elset = True
                current_set = [p.split('=')[-1].strip() for p in l.split(',') if 'elset=' in p.lower()][0]
                elsets[current_set] = []
                continue

            if '*part' in l_lower:
                in_part = True
                continue
            if '*end part' in l_lower:
                in_part = False
                continue

            if not in_part:
                if '*dsload' in l_lower:
                    in_dsload = True
                    continue
                if '*boundary' in l_lower:
                    in_boundary = True
                    continue
                if '*cload' in l_lower:
                    in_cload = True
                    continue
                if '*elastic' in l_lower:
                    in_elastic_block = True
                    continue

            in_nodes = '*node' in l_lower
            in_elements = '*element' in l_lower
            in_section = '*solid section' in l_lower
            continue

        if in_surface:
            parts = l.split(',')
            surfaces[current_set] = (parts[0].strip(), parts[1].strip())
            in_surface = False
            continue

        elif in_assembly and (in_nset or in_elset):
            parts = [p.strip() for p in l.split(',') if p.strip()]
            if len(parts) == 3 and 'generate' in linhas[max(i-1, 0)].lower():
                inicio = int(float(parts[0]))
                fim = int(float(parts[1]))
                passo = int(float(parts[2]))
                conjunto = list(range(inicio, fim + 1, passo))
            else:
                conjunto = [int(float(p)) for p in parts]
            if in_nset:
                nsets[current_set].extend(conjunto)
            elif in_elset:
                elsets[current_set].extend(conjunto)

        elif in_part and in_nodes:
            parts = l.split(',')
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[node_id] = (x, y)

        elif in_part and in_elements:
            parts = l.split(',')
            if len(parts) >= 5:
                elem_id = int(parts[0])
                elconn[elem_id] = [int(p) for p in parts[1:]]

        elif in_elastic_block:
            parts = l.split(',')
            if len(parts) >= 2:
                E = float(parts[0])
                nu = float(parts[1])
                in_elastic_block = False

        elif in_part and in_section:
            parts = l.split(',')
            if parts[0].strip() != '':
                esp = float(parts[0])

        elif in_boundary:
            parts = l.split(',')
            nome = parts[0].strip()
            if nome in nsets:
                for nid in nsets[nome]:
                    gdl_restritos.append(2 * (nid - 1))
                    gdl_restritos.append(2 * nid - 1)

        elif in_dsload:
            parts = l.split(',')
            nome_surf = parts[0].strip()
            tipo = parts[1].strip()

            if tipo == 'TRVEC':
                intensidade = float(parts[2])
                vx = float(parts[3])
                vy = float(parts[4])
                vz = float(parts[5])

                vx, vy = normalize_vector(vx, vy)
                dsloads[nome_surf] = (intensidade, (vx, vy, vz))

        elif in_cload:
            parts = l.split(',')
            nome = parts[0].strip()
            direcao = int(parts[1])
            valor = float(parts[2])

            if len(f) == 0:
                f = np.zeros(2 * len(nodes))

            if nome in nsets:
                for nid in nsets[nome]:
                    idx = 2 * (nid - 1) if direcao == 1 else 2 * nid - 1
                    f[idx] += valor
            elif nome.isdigit():
                nid = int(nome)
                idx = 2 * nid - 1 if direcao == 1 else 2 * nid
                f[idx] += valor

    node_ids_sorted = sorted(nodes.keys())
    nodemap = {nid: idx for idx, nid in enumerate(node_ids_sorted)}
    coords = np.array([nodes[nid] for nid in node_ids_sorted])

    conn = [[nodemap[n] for n in elconn[eid]] for eid in sorted(elconn.keys())]

    Elist = [E] * len(conn)
    nulist = [nu] * len(conn)
    rholist = [rho] * len(conn)

    for surf_name, (elset_name, face) in surfaces.items():
        if surf_name in dsloads:
            intensidade, (vx, vy, _) = dsloads[surf_name]
            elementos = elsets[elset_name]

            for eid in elementos:
                nodes_e = conn[eid - 1]
                face_nodes = obter_face_nodes(nodes_e, face)
                aplicar_forca_face(f, face_nodes, intensidade, (vx, vy), coords)

    return coords, conn, Elist, nulist, rholist, esp, plane_state, gdl_restritos, f

if __name__ == "__main__":
    
    path = "beam_08.inp"  # Substitua pelo caminho correto do arquivo .inp
    coords, conn, Elist, nulist, rholist, esp, plane_state, gdl_restritos, f = ler_inp(path)

    n, e = sp.symbols('n e')

    print("Utilização da formulação paramétrica")

    espessura = 0.15
    print("Espessura da chapa: ", espessura)
#   E =2e4
    E =Elist[0]
    print("Módulo de elasticidade: ", E)
#   nu = 0.25
    nu = nulist[0]
    print("Coeficiente de Poisson: ", nu)

    D = matrix_constitutiva(E, nu)
    print("Matriz constitutiva D:")
    sp.pprint(D)

    W = 1
    print("Pontos de Gauss:") 
    gauss_pontos = [(-0.577, -0.577), (0.577, -0.577), (0.577, 0.577), (-0.577, 0.577)]
    sp.pprint(gauss_pontos)
    
    W_reduzido = 2  
    print("Ponto de Gauss reduzido:") 
    gauss_pontos_reduzido = [(0, 0)]
    sp.pprint(gauss_pontos)
    
    print("Coordenadas globais dos nós:")
    print("  [x, y]")
    global_coords = coords
    
    print("\nConectividade dos elementos:")
    elementos = conn
    sp.pprint(elementos)

    total_no = global_coords.shape[0] * 2

    for idx, elem in enumerate(elementos):
        #print("\nResultados para o elemento {}".format(idx + 1))
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        #print("Coordenadas locais do elemento:")
      #  sp.pprint(local_coords)

        #print("\nResultados da funcao_forma (expressões simbólicas):")
        N = funcao_formaQ4(n, e)
     #   sp.pprint(N)

        #print("\nResultados da funcao_forma_derivada (d/dn):")
        dN_dn, dN_de = funcao_forma_derivada(n, e)
     #   sp.pprint(dN_dn)
        #print("Resultados da funcao_forma_derivada (d/de):")
      #  sp.pprint(dN_de)

        #print("\nMatriz ji (com símbolos n e e):")
     #   sp.pprint(ji(n, e))

        for i, (n_val, e_val) in enumerate(gauss_pontos, start=1):
    #        print("\nPonto de Gauss {}:".format(i))
     #       print("Matriz Jacobiana:")
            J, detJ, invJ = jacobiano_gauss(n_val, e_val, local_coords)
     #       sp.pprint(J)
     #       print("Determinante da Jacobiana:")
       #     sp.pprint(detJ)
      #      print("Inversa da Jacobiana:")
       #     sp.pprint(invJ)

     #       print("\nMatriz Be:")
            Be = matrix_Be(n, e, local_coords).subs({n: n_val, e: e_val})
      #      sp.pprint(Be)
            
      #      print("\nMatriz jacobiana gauss:")
      #      sp.pprint(ji_gauss(n_val, e_val))
            
        for i, (n_val, e_val) in enumerate(gauss_pontos_reduzido, start=1):
     #       print("\nPonto de Gauss reduzido {}:".format(i))
     #       print("Matriz Jacobiana com ponto de gauss reduzido:")
            J_gauss_reduzido, detJ_gauss_reduzido, invJ_gauss_reduzido = jacobiano_gauss(n_val, e_val, local_coords)
     #       sp.pprint(J_gauss_reduzido)
      #      print("Determinante da Jacobiana com ponto de gauss reduzido:")
      #      sp.pprint(detJ_gauss_reduzido)
      #      print("Inversa da Jacobiana com ponto de gauss reduzido:")
      #      sp.pprint(invJ_gauss_reduzido)

     #       print("\nMatriz Be com ponto de gauss reduzido:")
            Be_gauss_reduzido = matrix_Be(n, e, local_coords).subs({n: n_val, e: e_val})
      #      sp.pprint(Be_gauss_reduzido)
            
       #     print("\nMatriz jacobiana gauss reduzido:")
       #     sp.pprint(ji_gauss(n_val, e_val))

     #   print("\nDerivadas globais para o elemento:")
        deriv_glob = derivadas_global(n, e, local_coords)
    #    sp.pprint(deriv_glob)

        print("\nMatriz Ke Elemento:")
        Ke_elem = matrix_Ke_global_elemento(local_coords, n, e, D, W, gauss_pontos)
        sp.pprint(Ke_elem)
        
        print("\nMatriz Ke Elemento utilizando ponto de gauss reduzido:")
        Ke_elem_gauss_reduzido = matrix_Ke_global_elemento(local_coords, n, e, D, W, gauss_pontos_reduzido)
        sp.pprint(Ke_elem)

    K_global = matriz_Ke_global(global_coords, elementos, n, e, D, W, gauss_pontos)
    print("Matriz de rigidez global K:")
    sp.pprint(K_global)
    
    K_global_gauss_reduzido = matriz_Ke_global(global_coords, elementos, n, e, D, W, gauss_pontos_reduzido)
    print("Matriz de rigidez global K gauss reduzido:")
    sp.pprint(K_global_gauss_reduzido)

    F_global = sp.zeros(total_no, 1)
#    F_global[27] = -10
    for i, valor in enumerate(f):
        F_global[i] = valor
    print("\nVetor de forças nodais equivalentes F:")
    sp.pprint(F_global)

    constrained_dofs = gdl_restritos
    #constrained_dofs = [0, 1, 14, 15, 28, 29]

    K_reduzido, novo_no = matrix_Ke_reduzido_elemento(K_global, constrained_dofs)
    print("\nMatriz de rigidez reduzida: ")
    sp.pprint(K_reduzido)
    
    K_reduzido_gauss_reduzido, novo_no = matrix_Ke_reduzido_elemento(K_global_gauss_reduzido, constrained_dofs)
    print("\nMatriz de rigidez reduzida gauss reduzido: ")
    sp.pprint(K_reduzido_gauss_reduzido)

    F_reduzido = F_global.extract(novo_no, [0])
    print("\nVetor de forças reduzido: ")
    sp.pprint(F_reduzido)

    u_reduzido = deslocamento_reduzido(K_reduzido, F_reduzido)
    print("\nVetor de deslocamento reduzido: ")
    sp.pprint(u_reduzido)
    
    u_reduzido_gauss_reduzido = deslocamento_reduzido(K_reduzido_gauss_reduzido, F_reduzido)
    print("\nVetor de deslocamento reduzido gauss reduzido: ")
    sp.pprint(u_reduzido_gauss_reduzido)

    u_global = deslocamento_global(u_reduzido, novo_no, total_no)
    print("\nVetor de deslocamento global u:")
    sp.pprint(u_global)
    
    u_global_gauss_reduzido = deslocamento_global(u_reduzido_gauss_reduzido, novo_no, total_no)
    print("\nVetor de deslocamento global u gauss reduzido:")
    sp.pprint(u_global_gauss_reduzido)

    print("\nDeformações e tensões em cada elemento:")
        
    for idx, elem in enumerate(elementos):
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        u_local = sp.Matrix(8, 1, lambda i, _: u_global[elem[i//2]*2 + (i % 2)])
        defor = deformacao_gauss(u_local, n, e, local_coords, gauss_pontos)
        tens = tensoes_gauss(u_local, n, e, local_coords, gauss_pontos, D)
        print(f"\nElemento {idx+1}:")
        print("Deformações nos pontos de Gauss:")
        for pt_idx, d in enumerate(defor, start=1):
            print(f"  Gauss {pt_idx}:")
            sp.pprint(d)
        print("Tensões nos pontos de Gauss:")
        for pt_idx, t in enumerate(tens, start=1):
            print(f"  Gauss {pt_idx}:")
            sp.pprint(t)
        print("Tensões nos nós do elemento:")
        
        t_nos = tensoes_nos(u_local, n, e, local_coords, gauss_pontos, D)
        for i, row in enumerate(t_nos.tolist(), start=1):
            print(f"Nó {i}: {row}")
            
        print("Deformações nos nós do elemento:")
        strains_nos = []
        D_inv = D.inv()
        for i, row in enumerate(t_nos.tolist(), start=1):
            sigma_node = sp.Matrix(row)
            epsilon_node = D_inv * sigma_node
            strains_nos.append(epsilon_node)
            print(f"Nó {i} - Deformação: {epsilon_node.tolist()}")
            
    for idx, elem in enumerate(elementos):
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        u_local_gauss_reduzido = sp.Matrix(8, 1, lambda i, _: u_global_gauss_reduzido[elem[i//2]*2 + (i % 2)])
        defor_gauss_reduzido = deformacao_gauss_reduzido(u_local, n, e, local_coords, gauss_pontos_reduzido)
        tens_gauss_reduzido = tensoes_gauss_reduzido(u_local_gauss_reduzido, n, e, local_coords, gauss_pontos_reduzido, D)
        print(f"\nElemento {idx+1}:")
        print("Deformações nos pontos de Gauss reduzido:")
        for pt_idx, d in enumerate(defor_gauss_reduzido, start=1):
            print(f"  Gauss reduzido {pt_idx}:")
            sp.pprint(d)
        print("Tensões nos pontos de Gauss reduzido:")
        for pt_idx, t in enumerate(tens_gauss_reduzido, start=1):
            print(f"  Gauss reduzido {pt_idx}:")
            sp.pprint(t)
        print("Tensões nos nós do elemento gauss reduzido:")
        
        t_nos_gauss_reduzido = tensoes_nos_reduzido(u_local_gauss_reduzido, n, e, local_coords, gauss_pontos_reduzido, D)
        for i, row in enumerate(t_nos_gauss_reduzido.tolist(), start=1):
            print(f"Nó {i}: {row}")
        
        print("Deformação nos nós do elemento gauss reduzido:")
        defor_nos_gauss_reduzido = deformacao_nos_reduzido(u_local_gauss_reduzido, n, e, local_coords, gauss_pontos_reduzido)
        for i, row in enumerate(defor_nos_gauss_reduzido.tolist(), start=1):
            print(f"Nó {i}: {row}")

 #   rho = 7850
    rho = rholist[0]   
    M_global = matriz_Me_global(global_coords,
                                elementos,
                                n, e,
                                rho,
                                espessura,
                                W,
                                gauss_pontos)
 #   print("\nMatriz de massa global M:")
 #   sp.pprint(M_global)
    
    M_global_gauss_reduzido = matriz_Me_global(global_coords,
                                elementos,
                                n, e,
                                rho,
                                espessura,
                                W_reduzido,
                                gauss_pontos_reduzido)
 #   print("\nMatriz de massa global M gauss reduzido:")
 #   sp.pprint(M_global_gauss_reduzido)
    
    M_reduzida = matrix_Me_reduzido(M_global, constrained_dofs)
  #  print("\nMatriz de massa reduzida M_reduzida:")
 #   sp.pprint(M_reduzida)
    
    M_reduzida_gauss_reduzido = matrix_Me_reduzido(M_global_gauss_reduzido, constrained_dofs)
  #  print("\nMatriz de massa reduzida M_reduzida gauss reduzido:")
  #  sp.pprint(M_reduzida_gauss_reduzido)
  
    # tempo crítico
    dt_crit = calcula_dt_critico(K_global, M_global)
    print("\nTempo crítico dt_crit:")
    sp.pprint(dt_crit)
    
     # Definindo a força externa como uma função do tempo
    F = lambda t: np.zeros(M_reduzida.shape[0])
    # F = lambda t: np.array([0, 0]) if t < 10 else np.array([0, 1])

    # Condições iniciais para o sistema reduzido
    u0 = np.zeros(M_reduzida.shape[0])
    v0 = np.zeros(M_reduzida.shape[0])
    
    # Intervalo de tempo
    t0 = 0.0
    tf = 84.0
    
     # Passo de tempo (usando o passo crítico)
    dt = dt_crit*0.01

    # Executando o método de Newmark
    t, u, v, a = Newmark(dt, t0, tf, u0, v0, M_reduzida, K_reduzido, F)

    # Convertendo coordenadas e deslocamentos para arrays numéricos
    n_nos = global_coords.shape[0]
    orig_nodes = np.array([[float(global_coords[i, 0]), float(global_coords[i, 1])] for i in range(n_nos)])
    # Deslocamento vem como lista de floats para gauss
    u_global_num = np.array([float(u_global[i]) for i in range(u_global.shape[0])]).reshape(n_nos, 2)
    # Deslocamento vem como lista de floats para gauss reduzido
    u_global_num_gauss_reduzido = np.array([float(u_global_gauss_reduzido[i]) for i in range(u_global_gauss_reduzido.shape[0])]).reshape(n_nos, 2)
    
    # Escala para amplificar os deslocamentos, se necessário
    escala = 1  
    def_nodes = orig_nodes + escala * u_global_num
    def_nodes_gauss_reduzido = orig_nodes + escala * u_global_num_gauss_reduzido

    # Converter vetor de forças para array numérico e organizar por nó
    n_nos = global_coords.shape[0]
    F_global_num = np.array([float(F_global[i, 0]) for i in range(F_global.shape[0])]) \
                       .reshape(n_nos, 2)

    # Plotando em dois subplots: um para gauss e outro para gauss reduzido
    plt.figure(figsize=(12, 6))

   # PARÂMETROS PARA ESCALA DAS SETAS DE FORÇA
    # quanto menor o 'scale', maior a seta; ajustar conforme necessário
    escala_forca = 45
    largura_seta = 0.008

    # Plot para Gauss
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(orig_nodes[:, 0], orig_nodes[:, 1], c='blue', label='Nós Originais')
    ax1.scatter(def_nodes[:, 0], def_nodes[:, 1], c='red',  label='Nós Deformados (Gauss)')
    # linhas dos elementos
    for elem in elementos:
        x_o = [orig_nodes[i, 0] for i in elem] + [orig_nodes[elem[0], 0]]
        y_o = [orig_nodes[i, 1] for i in elem] + [orig_nodes[elem[0], 1]]
        x_d = [def_nodes[i, 0] for i in elem]  + [def_nodes[elem[0], 0]]
        y_d = [def_nodes[i, 1] for i in elem]  + [def_nodes[elem[0], 1]]
        ax1.plot(x_o, y_o, 'b--', linewidth=1)
        ax1.plot(x_d, y_d, 'r-',  linewidth=1)
    # vetor de força em cada nó (seta na posição deformada)
    ax1.quiver(
        def_nodes[:, 0], def_nodes[:, 1],
        F_global_num[:, 0], F_global_num[:, 1],
        color='purple',
        angles='xy', scale_units='xy',
        scale=escala_forca,
        width=largura_seta,
        label='Força nodal'
    )
    ax1.set_title("Malha e Força (Gauss completo)")
    ax1.set_xlabel("x");  ax1.set_ylabel("y")
    ax1.axis('equal')
    ax1.legend()

    # Plot para Gauss reduzido
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(orig_nodes[:, 0], orig_nodes[:, 1], c='blue',  label='Nós Originais')
    ax2.scatter(def_nodes_gauss_reduzido[:, 0],
                def_nodes_gauss_reduzido[:, 1],
                c='green', label='Nós Deformados (Gauss Reduzido)')
    for elem in elementos:
        x_o = [orig_nodes[i, 0] for i in elem] + [orig_nodes[elem[0], 0]]
        y_o = [orig_nodes[i, 1] for i in elem] + [orig_nodes[elem[0], 1]]
        x_d = [def_nodes_gauss_reduzido[i, 0] for i in elem] \
              + [def_nodes_gauss_reduzido[elem[0], 0]]
        y_d = [def_nodes_gauss_reduzido[i, 1] for i in elem] \
              + [def_nodes_gauss_reduzido[elem[0], 1]]
        ax2.plot(x_o, y_o, 'b--', linewidth=1)
        ax2.plot(x_d, y_d, 'g-',  linewidth=1)
    ax2.quiver(
        def_nodes_gauss_reduzido[:, 0],
        def_nodes_gauss_reduzido[:, 1],
        F_global_num[:, 0],
        F_global_num[:, 1],
        color='purple',
        angles='xy', scale_units='xy',
        scale=escala_forca,
        width=largura_seta,
        label='Força nodal'
    )
    ax2.set_title("Malha e Força (Gauss Reduzido)")
    ax2.set_xlabel("x");  ax2.set_ylabel("y")
    ax2.axis('equal')
    ax2.legend()

    plt.tight_layout()
    
    uAnalitic1 = (np.sin(t) + np.sin(t*np.sqrt(21))/np.sqrt(21)) / 2
    uAnalitic2 = (np.sin(t) - np.sin(t*np.sqrt(21))/np.sqrt(21)) / 2

    plt.figure(figsize=(12, 7))
    
    plt.plot(t, uAnalitic1, label='u1 (Analítico)', color='blue', linewidth=2)
    plt.plot(t, u[0, :], label='u1 (Deslocamento 1)', color='blue', linestyle='--', linewidth=2)
    
    plt.plot(t, uAnalitic2, label='u2 (Analítico)', color='red', linewidth=2)
    plt.plot(t, u[1, :], label='u2 (Deslocamento 2)', color='red', linestyle='--', linewidth=2)
    
    plt.title('Deslocamentos ao longo do tempo', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)', fontsize=14)
    plt.ylabel('Deslocamento (m)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

    

