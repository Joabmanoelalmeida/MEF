import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

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

def funcao_formaQ8(n, e):
    N18Q = (0.25*(1 - n)*(1 - e)) - (0.25*(1 - n**2)*(1 - e)) - (0.25*(1 - n)*(1 - e**2))
    N28Q = (0.25*(1 + n)*(1 - e)) - (0.25*(1 + n)*(1 - e**2)) - (0.25*(1 - n**2)*(1 - e))
    N38Q = (0.25*(1 + n)*(1 + e)) - (0.25*(1 - n**2)*(1 + e)) - (0.25*(1 + n)*(1 - e**2))
    N48Q = (0.25*(1 - n)*(1 + e)) - (0.25*(1 - n)*(1 - e**2)) - (0.25*(1 - n**2)*(1 + e))
    N58Q = 0.5*(1 - n**2)*(1 - e)
    N68Q = 0.5*(1 + n)*(1 - e**2)
    N78Q = 0.5*(1 - n**2)*(1 + e)
    N88Q = 0.5*(1 - n)*(1 - e**2)
    return [N18Q, N28Q, N38Q, N48Q, N58Q, N68Q, N78Q, N88Q]

def funcao_forma_derivada(n, e, tipo="Q8"):
    if tipo.upper() == "Q8":
        N = funcao_formaQ8(n, e)
    elif tipo.upper() == "Q4":
        N = funcao_formaQ4(n, e)
    else:
        raise ValueError("Tipo de elemento não suportado: apenas Q4 ou Q8 são válidos")
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

    J_func    = sp.lambdify((n, e), J, 'numpy')
    detJ_func = sp.lambdify((n, e), detJ, 'numpy')
    invJ_func = sp.lambdify((n, e), invJ, 'numpy')
    J_gauss    = sp.Matrix(np.array(J_func(n_val, e_val)))
    detJ_gauss = sp.sympify(detJ_func(n_val, e_val))
    invJ_gauss = sp.Matrix(np.array(invJ_func(n_val, e_val)))
    return J_gauss, detJ_gauss, invJ_gauss

def ji(n, e):
    _, dN_de = funcao_forma_derivada(n, e)
    dN_dn, _ = funcao_forma_derivada(n, e)
    Ji = sp.Matrix([dN_de, dN_dn])
    return Ji

def ji_gauss(n_val, e_val):
    n, e = sp.symbols('n e')
    Ji_sym = ji(n, e)
    Ji_func = sp.lambdify((n, e), Ji_sym, 'numpy')
    return sp.Matrix(np.array(Ji_func(n_val, e_val)))

def derivadas_global(n, e, coords):
    _, _, invJ = matrix_jacobiano(n, e, coords)
    dN_dn, dN_de = funcao_forma_derivada(n, e)
    dN_global = []
    for dnde, dnd in zip(dN_de, dN_dn):
        dN_nat_vec = sp.Matrix([dnde, dnd])
        dxy = invJ * dN_nat_vec
        dN_global.append((dxy[0], dxy[1]))
    return dN_global

def matrix_Be(n, e, coords):
    dN_global = derivadas_global(n, e, coords)
    dx = [item[0] for item in dN_global]
    dy = [item[1] for item in dN_global]
    if len(dx) == 4:
        Be = sp.Matrix([
            [dx[0], 0,   dx[1], 0,   dx[2], 0,   dx[3], 0],
            [0,   dy[0], 0,   dy[1], 0,   dy[2], 0,   dy[3]],
            [dy[0], dx[0], dy[1], dx[1], dy[2], dx[2], dy[3], dx[3]]
        ])
    elif len(dx) == 8:
        Be = sp.Matrix([
            [dx[0], 0,   dx[1], 0,   dx[2], 0,   dx[3], 0,   dx[4], 0,   dx[5], 0,   dx[6], 0,   dx[7], 0],
            [0,   dy[0], 0,   dy[1], 0,   dy[2], 0,   dy[3], 0,   dy[4], 0,   dy[5], 0,   dy[6], 0,   dy[7]],
            [dy[0], dx[0], dy[1], dx[1], dy[2], dx[2], dy[3], dx[3], dy[4], dx[4], dy[5], dx[5], dy[6], dx[6], dy[7], dx[7]]
        ])
    else:
        raise ValueError("Número de nós não suportado: apenas 4 ou 8 nós são válidos")
    return Be

def matrix_constitutiva(E, nu):
    """
    Matriz constitutiva para um material isotrópico em 2D.
    """
    D = E / (1 - nu**2) * sp.Matrix([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    return D

def matrix_Ke_global_elemento(coords, n, e, D, W, gauss_pontos):

    npts = coords.shape[0]
    ndof = npts * 2
    Ke_total = sp.zeros(ndof, ndof)

    Be_sym    = matrix_Be(n, e, coords)              
    detJ_sym  = matrix_jacobiano(n, e, coords)[1]      
    Be_func   = sp.lambdify((n, e), Be_sym, 'numpy')    
    detJ_func = sp.lambdify((n, e), detJ_sym, 'numpy')  
    D_num     = np.array(D, float)                    

    for pt in gauss_pontos:
        if len(pt) == 3:
            n_val, e_val, peso = pt
        else:
            n_val, e_val = pt
            peso = 1

        Be_num   = Be_func(n_val, e_val)            
        detJ_num = detJ_func(n_val, e_val)           

        Ke = peso * (Be_num.T @ D_num @ Be_num) * detJ_num * espessura
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

    Be_sym = matrix_Be(n, e, coords)
    Be_func = sp.lambdify((n, e), Be_sym, 'numpy')
    for pt in gauss_points:
        if len(pt) == 3:
            n_val, e_val, _ = pt
        else:
            n_val, e_val = pt
        Be_numerico = sp.Matrix(np.array(Be_func(n_val, e_val)))
        tens = Be_numerico * u_local
        defor.append(tens)
    return defor 

def deformacao_gauss_reduzido(u_local, n, e, coords, gauss_points_reduzido):
    defor = []

    Be_sym = matrix_Be(n, e, coords)
    Be_func = sp.lambdify((n, e), Be_sym, 'numpy')
    for pt in gauss_points_reduzido:
        if len(pt) == 3:
            n_val, e_val, _ = pt
        else:
            n_val, e_val = pt
        Be_numerico = sp.Matrix(np.array(Be_func(n_val, e_val)))
        tens = Be_numerico * u_local
        defor.append(tens)
    return defor

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
    a, b = 2.732050808, -0.732050808
    E = sp.Matrix([
                [a**2, a**2, b**2, b**2, a*b, a*b, a*b, a*b, 4],
                [b**2, a**2, a**2, b**2, a*b, a*b, a*b, a*b, 4],
                [b**2, b**2, a**2, a**2, a*b, a*b, a*b, a*b, 4],
                [a**2, b**2, b**2, a**2, a*b, a*b, a*b, a*b, 4],
                [a*b, a*b, a*b, a*b, b**2, a**2, b**2, a**2, 4],
                [a*b, a*b, a*b, a*b, a**2, b**2, a**2, b**2, 4],
                [a*b, a*b, a*b, a*b, a**2, b**2, a**2, b**2, 4],
                [a*b, a*b, a*b, a*b, b**2, a**2, b**2, a**2, 4]
            ])/36
    return E

def matriz_E_reduzido():
    a, b = 2.732050808, -0.732050808
    E = sp.Matrix([
                [a**2/4, a*b/4, b**2/4, a*b/4],
                [a*b/4, a**2/4, a*b/4, b**2/4],
                [b**2/4, a*b/4, a**2/4, a*b/4],
                [a*b/4, b**2/4, a*b/4, a**2/4],
                [a/4, a/4, b/4, b/4],
                [b/4, a/4, a/4, b/4],
                [b/4, b/4, a/4, a/4],
                [a/4, b/4, b/4, a/4]
            ])
    return E

def tensoes_nos(u_local, n, e, coords, gauss_points, D):
    t_gauss = tensoes_gauss(u_local, n, e, coords, gauss_points, D)

    if len(t_gauss) > 9:

        t_gauss = [t_gauss[i] for i in (0, 2, 8, 6)]
    tx_gauss = sp.Matrix([stress[0] for stress in t_gauss])
    ty_gauss = sp.Matrix([stress[1] for stress in t_gauss])
    txy_gauss = sp.Matrix([stress[2] for stress in t_gauss])
    tx_nos = matriz_E() * tx_gauss
    ty_nos = matriz_E() * ty_gauss
    txy_nos = matriz_E() * txy_gauss
    t_nos = sp.Matrix.hstack(tx_nos, ty_nos, txy_nos)
    return t_nos

def deformacao_nos(u_local, n, e, coords, gauss_points):
    defor_gauss = deformacao_gauss(u_local, n, e, coords, gauss_points)

    if len(defor_gauss) > 9:

        defor_gauss = [defor_gauss[i] for i in (0, 2, 8, 6)]
    deforx_gauss = sp.Matrix([strain[0] for strain in defor_gauss])
    defory_gauss = sp.Matrix([strain[1] for strain in defor_gauss])
    deforxy_gauss = sp.Matrix([strain[2] for strain in defor_gauss])
    deforx_nos = matriz_E() * deforx_gauss
    defory_nos = matriz_E() * defory_gauss
    deforxy_nos = matriz_E() * deforxy_gauss
    defor_nos = sp.Matrix.hstack(deforx_nos, defory_nos, deforxy_nos)
    return defor_nos

def tensoes_nos_reduzido(u_local, n, e, coords, gauss_points_reduzido, D):
    t_gauss = tensoes_gauss(u_local, n, e, coords, gauss_points_reduzido, D)

    if len(t_gauss) > 4:

        t_gauss = [t_gauss[i] for i in (0, 2, 8, 6)]
    tx_gauss = sp.Matrix([stress[0] for stress in t_gauss])
    ty_gauss = sp.Matrix([stress[1] for stress in t_gauss])
    txy_gauss = sp.Matrix([stress[2] for stress in t_gauss])
    tx_nos = matriz_E_reduzido() * tx_gauss
    ty_nos = matriz_E_reduzido() * ty_gauss
    txy_nos = matriz_E_reduzido() * txy_gauss
    t_nos = sp.Matrix.hstack(tx_nos, ty_nos, txy_nos)
    return t_nos

def deformacao_nos_reduzido(u_local, n, e, coords, gauss_points_reduzido):
    defor_gauss_reduzido = deformacao_gauss(u_local, n, e, coords, gauss_points_reduzido)

    if len(defor_gauss_reduzido) > 4:

        defor_gauss_reduzido = [defor_gauss_reduzido[i] for i in (0, 2, 8, 6)]
    deforx_gauss_reduzido = sp.Matrix([strain[0] for strain in defor_gauss_reduzido])
    defory_gauss_reduzido = sp.Matrix([strain[1] for strain in defor_gauss_reduzido])
    deforxy_gauss_reduzido = sp.Matrix([strain[2] for strain in defor_gauss_reduzido])
    deforx_nos = matriz_E_reduzido() * deforx_gauss_reduzido
    defory_nos = matriz_E_reduzido() * defory_gauss_reduzido
    deforxy_nos = matriz_E_reduzido() * deforxy_gauss_reduzido
    deformacao_nos_reduzido = sp.Matrix.hstack(deforx_nos, defory_nos, deforxy_nos)
    return deformacao_nos_reduzido

def matriz_Ke_global(global_coords, elements, n, e, D, W, gauss_points):
    total_dofs = global_coords.shape[0] * 2
    K_global = sp.zeros(total_dofs, total_dofs)
    for elem in elements:
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        Ke_local = matrix_Ke_global_elemento(local_coords, n, e, D, W, gauss_points)

        for i_local, global_node in enumerate(elem):
            dofs_i = [global_node * 2, global_node * 2 + 1]
            for j_local, global_node_j in enumerate(elem):
                dofs_j = [global_node_j * 2, global_node_j * 2 + 1]
                K_global[dofs_i[0], dofs_j[0]] += Ke_local[2 * i_local, 2 * j_local]
                K_global[dofs_i[0], dofs_j[1]] += Ke_local[2 * i_local, 2 * j_local + 1]
                K_global[dofs_i[1], dofs_j[0]] += Ke_local[2 * i_local + 1, 2 * j_local]
                K_global[dofs_i[1], dofs_j[1]] += Ke_local[2 * i_local + 1, 2 * j_local + 1]
    return K_global

def matrix_Me_global_elemento(coords, n, e, rho, espessura, W, gauss_pontos):

    if coords.shape[0] == 4:
        N_funcs = funcao_formaQ4(n, e)
    elif coords.shape[0] == 8:
        N_funcs = funcao_formaQ8(n, e)
    else:
        raise ValueError("Número de nós não suportado em matrix_Me_global_elemento: apenas 4 ou 8 nós são válidos")

    N_funcs_l = [sp.lambdify((n, e), Ni, 'numpy') for Ni in N_funcs]
    npts = len(N_funcs)
    ngl = 2 * npts
    Me = sp.zeros(ngl, ngl)
    for pt in gauss_pontos:
        if len(pt) == 3:
            n_val, e_val, peso = pt
        else:
            n_val, e_val = pt
            peso = 1
        N_eval = [func(n_val, e_val) for func in N_funcs_l]
        Nmat = sp.zeros(2, ngl)
        for k, Nk in enumerate(N_eval):
            Nmat[0, 2*k]   = Nk
            Nmat[1, 2*k+1] = Nk
        _, detJ, _ = jacobiano_gauss(n_val, e_val, coords)
        Me += rho * (Nmat.T * Nmat) * detJ * peso * espessura
    return Me

def matriz_Me_global(global_coords, elements, n, e, rho, espessura, W, gauss_pontos):
    """
    Monta a matriz de massa global, similar à montagem da matriz de rigidez global.
    """
    total_dofs = global_coords.shape[0] * 2
    M_global = sp.zeros(total_dofs, total_dofs)
    for elem in elements:
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        Me_local = matrix_Me_global_elemento(local_coords, n, e, rho, espessura, W, gauss_pontos)

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

# Leitor genérico de arquivos .inp do ABAQUS
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

    in_part = in_nodes = in_elements = in_elastic = False
    in_elastic_block = in_section = in_boundary = in_cload = False
    in_surface = in_nset = in_elset = in_dsload = False
    in_assembly = False
    current_set = ''

    for i, linha in enumerate(linhas):
        l = linha.strip()
        if l == '' or l.startswith('**'):
            continue

        # Identifica se l é um comando
        if l.startswith('*'):
            # Resetar todas as flags locais de blocos
            in_boundary = in_cload = in_surface = in_nset = in_elset = False
            in_nodes = in_elements = in_section = in_elastic = False

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

    # Preencher coords e conn ordenados
    node_ids_sorted = sorted(nodes.keys())
    nodemap = {nid: idx for idx, nid in enumerate(node_ids_sorted)}
    coords = np.array([nodes[nid] for nid in node_ids_sorted])

    conn = [[nodemap[n] for n in elconn[eid]] for eid in sorted(elconn.keys())]

    Elist = [E] * len(conn)
    nulist = [nu] * len(conn)
    rholist = [rho] * len(conn)

    # Aplicar carregamentos distribuídos às superfícies
    if len(f) == 0:
        f = np.zeros(2 * len(nodes))

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
    
    path = "Q8_24.inp"  # Substitua pelo caminho correto do arquivo .inp
    coords, conn, Elist, nulist, rholist, esp, plane_state, gdl_restritos, f = ler_inp(path)

    n, e = sp.symbols('n e')

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

    w_dict = { -0.7745966692: 0.5555556, 0.0: 0.88889, 0.7745966692: 0.5555556 }

    gauss_pontos = []
    for n_val in [-0.7745966692, 0.0, 0.7745966692]:
        for e_val in [-0.7745966692, 0.0, 0.7745966692]:
            peso = w_dict[n_val] * w_dict[e_val]
            gauss_pontos.append((n_val, e_val, peso))
 #   print("Pontos de Gauss com pesos:")
  #  sp.pprint(gauss_pontos)

    W_reduzido = 1
 #  print("Pontos de Gauss:") 
    gauss_pontos_reduzido = [(-0.577, -0.577), (0.577, -0.577), (0.577, 0.577), (-0.577, 0.577)]
 #  sp.pprint(gauss_pontos_reduzido)

 #   print("Coordenadas globais dos nós:")
 #   print("  [x, y]")
    
    # Novas coordenadas globais para elementos Q8 (8 nós por elemento)
    # Novas coordenadas globais para elementos Q8 (nós de canto e nós de meio de aresta)
    '''global_coords = sp.Matrix([
        # nós de canto
        [0.0, 0.0],  # 0
        [0.0, 2.0],  # 1
        [0.0, 4.0],  # 2
        [2.0, 0.0],  # 3
        [2.0, 2.0],  # 4
        [2.0, 4.0],  # 5
        [4.0, 0.0],  # 6
        [4.0, 2.0],  # 7
        [4.0, 4.0],  # 8
        # nós de meio de aresta (vertical)
        [0.0, 1.0],  # 9   entre 0–1
        [0.0, 3.0],  # 10  entre 1–2
        [2.0, 1.0],  # 11  entre 3–4
        [2.0, 3.0],  # 12  entre 4–5
        [4.0, 1.0],  # 13  entre 6–7
        [4.0, 3.0],  # 14  entre 7–8
        # nós de meio de aresta (horizontal)
        [1.0, 0.0],  # 15  entre 0–3
        [3.0, 0.0],  # 16  entre 3–6
        [1.0, 2.0],  # 17  entre 1–4
        [3.0, 2.0],  # 18  entre 4–7
        [1.0, 4.0],  # 19  entre 2–5
        [3.0, 4.0]   # 20  entre 5–8
    ])
 #   sp.pprint(global_coords)

    # Conectividade dos elementos Q8
 #   print("\nConectividade dos elementos (Q8):")
 #   print("  [nó_canto1, nó_medio12, nó_canto2, nó_medio23, nó_canto3, nó_medio34, nó_canto4, nó_medio41]")
    elementos = [
        # Elemento 1 (canto inferior-esquerdo)
        [0,  15, 3,  11, 4,  17, 1,  9],
        # Elemento 2 (canto inferior-direito)
        [3,  16, 6,  13, 7,  18, 4,  11],
        # Elemento 3 (canto superior-esquerdo)
        [1,  17, 4,  12, 5,  19, 2,  10],
        # Elemento 4 (canto superior-direito)
        [4,  18, 7,  14, 8,  20, 5,  12]
    ]'''
    global_coords = coords
    '''
    global_coords = sp.Matrix([
        [0.0, 0.0],   #  0
        [0.0, 2.0],   #  1
        [0.0, 4.0],   #  2
        [0.0, 6.0],   #  3
        [0.0, 8.0],   #  4
        [0.0, 10.0],  #  5
        [0.0, 12.0],  #  6
        [2.0, 0.0],   #  7
        [2.0, 2.0],   #  8
        [2.0, 4.0],   #  9
        [2.0, 6.0],   # 10
        [2.0, 8.0],   # 11
        [2.0, 10.0],  # 12
        [2.0, 12.0],  # 13
        [4.0, 0.0],   # 14
        [4.0, 2.0],   # 15
        [4.0, 4.0],   # 16
        [4.0, 6.0],   # 17
        [4.0, 8.0],   # 18
        [4.0, 10.0],  # 19
        [4.0, 12.0],  # 20
        # nós de meio de aresta (vertical) col=0
        [0.0, 1.0],   # 21  0–1
        [0.0, 3.0],   # 22  1–2
        [0.0, 5.0],   # 23  2–3
        [0.0, 7.0],   # 24 3–4
        [0.0, 9.0],   # 25  4–5
        [0.0, 11.0],  # 26 5–6
        # vertical col=1
        [2.0, 1.0],   # 27 7–8
        [2.0, 3.0],   # 28  8–9
        [2.0, 5.0],   # 29  9–10
        [2.0, 7.0],   # 30  10–11
        [2.0, 9.0],   # 31  11–12
        [2.0, 11.0],  # 32  12–13
        # vertical col=2
        [4.0, 1.0],   # 33  14–15
        [4.0, 3.0],   # 34  15–16
        [4.0, 5.0],   # 35  16–17
        [4.0, 7.0],   # 36  17–18
        [4.0, 9.0],   # 37  18–19
        [4.0, 11.0],  # 38  19–20
        # nós de meio de aresta (horizontal) entre col0–col1
        [1.0, 0.0],   # 39
        [1.0, 2.0],   # 40
        [1.0, 4.0],   # 41
        [1.0, 6.0],   # 42
        [1.0, 8.0],   # 43
        [1.0, 10.0],  # 44
        [1.0, 12.0],  # 45
        # nós de meio de aresta (horizontal) entre col1–col2
        [3.0, 0.0],   # 46
        [3.0, 2.0],   # 47
        [3.0, 4.0],   # 48
        [3.0, 6.0],   # 49
        [3.0, 8.0],   # 50
        [3.0, 10.0],  # 51
        [3.0, 12.0],  # 52
    ])'''

    print("Global coords (com nós de meio de aresta):")
    sp.pprint(global_coords)
    elementos = conn
    '''
    elementos = [
        [ 0, 39,  7, 27,  8, 40,  1, 22],
        [ 7, 46, 14, 33, 15, 47,  8, 27],
        [ 8, 47, 15, 34, 16, 48,  9, 28],
        [ 1, 40,  8, 28,  9, 41,  2, 22],
        [ 2, 41,  9, 29, 10, 42,  3, 23],
        [ 9, 48, 16, 35, 17, 49, 10, 29],
        [10, 49, 17, 36, 18, 50, 11, 30],
        [ 3, 42, 10, 30, 11, 43,  4, 24],
        [ 4, 43, 11, 31, 12, 44,  5, 25],
        [11, 50, 18, 37, 19, 51, 12, 31],
        [ 5, 44, 12, 32, 13, 45,  6, 26],
        [12, 51, 19, 38, 20, 52, 13, 32],
    ]
    print("\nConectividade (Q8):")'''
    sp.pprint(elementos)

    
    total_no = global_coords.shape[0] * 2

    for idx, elem in enumerate(elementos):
 #       print("\nResultados para o elemento {}".format(idx + 1))
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
#        print("Coordenadas locais do elemento:")
#        sp.pprint(local_coords)

        for i, pt in enumerate(gauss_pontos, start=1):
            if len(pt) == 3:
                n_val, e_val, _ = pt
            else:
                n_val, e_val = pt
#            print("\nPonto de Gauss {}:".format(i))
#            print("Matriz Jacobiana:")
            J, detJ, invJ = jacobiano_gauss(n_val, e_val, local_coords)
 #           sp.pprint(J)
 #           print("Determinante da Jacobiana:")
  #          sp.pprint(detJ)
 #           print("Inversa da Jacobiana:")
 #           sp.pprint(invJ)
         
 #           print("\nMatriz Be:")
            Be = matrix_Be(n, e, local_coords).subs({n: n_val, e: e_val})
  #          sp.pprint(Be)
            
   #         print("\nMatriz jacobiana gauss:")
  #          sp.pprint(ji_gauss(n_val, e_val))
            J, detJ, invJ = jacobiano_gauss(n_val, e_val, local_coords)
  #          sp.pprint(J)
  #          print("Determinante da Jacobiana:")
  #          sp.pprint(detJ)
  #          print("Inversa da Jacobiana:")
   #         sp.pprint(invJ)

   #         print("\nMatriz Be:")
            Be = matrix_Be(n, e, local_coords).subs({n: n_val, e: e_val})
  #          sp.pprint(Be)
            
  #          print("\nMatriz jacobiana gauss:")
  #          sp.pprint(ji_gauss(n_val, e_val))
            
        for i, (n_val, e_val) in enumerate(gauss_pontos_reduzido, start=1):
  #          print("\nPonto de Gauss reduzido {}:".format(i))
   #         print("Matriz Jacobiana com ponto de gauss reduzido:")
            J_gauss_reduzido, detJ_gauss_reduzido, invJ_gauss_reduzido = jacobiano_gauss(n_val, e_val, local_coords)
  #          sp.pprint(J_gauss_reduzido)
   #         print("Determinante da Jacobiana com ponto de gauss reduzido:")
   #         sp.pprint(detJ_gauss_reduzido)
   #         print("Inversa da Jacobiana com ponto de gauss reduzido:")
   #         sp.pprint(invJ_gauss_reduzido)

  #         print("\nMatriz Be com ponto de gauss reduzido:")
            Be_gauss_reduzido = matrix_Be(n, e, local_coords).subs({n: n_val, e: e_val})
 #           sp.pprint(Be_gauss_reduzido)
            
    #        print("\nMatriz jacobiana gauss reduzido:")
    #        sp.pprint(ji_gauss(n_val, e_val))

    #    print("\nDerivadas globais para o elemento:")
        deriv_glob = derivadas_global(n, e, local_coords)
   #     sp.pprint(deriv_glob)

  #      print("\nMatriz Ke Elemento:")
        Ke_elem = matrix_Ke_global_elemento(local_coords, n, e, D, w_dict, gauss_pontos)
 #       sp.pprint(Ke_elem)
        
  #      print("\nMatriz Ke Elemento utilizando ponto de gauss reduzido:")
        Ke_elem_gauss_reduzido = matrix_Ke_global_elemento(local_coords, n, e, D, w_dict, gauss_pontos_reduzido)
  #      sp.pprint(Ke_elem_gauss_reduzido)

    K_global = matriz_Ke_global(global_coords, elementos, n, e, D, w_dict, gauss_pontos)
 #   print("Matriz de rigidez global K:")
 #   sp.pprint(K_global)
    
    K_global_gauss_reduzido = matriz_Ke_global(global_coords, elementos, n, e, D, w_dict, gauss_pontos_reduzido)
 #   print("Matriz de rigidez global K gauss reduzido:")
 #   sp.pprint(K_global_gauss_reduzido)

    F_global = sp.zeros(total_no, 1)
    #F_global[27] = -15
    for i, valor in enumerate(f):
        F_global[i] = valor
    print("\nVetor de forças nodais equivalentes F:")
 #   print("\nVetor de forças nodais equivalentes F:")
 #   sp.pprint(F_global)
 
    constrained_dofs = gdl_restritos
   # constrained_dofs = [0, 1, 14, 15, 28, 29]

    K_reduzido, novo_no = matrix_Ke_reduzido_elemento(K_global, constrained_dofs)
#    print("\nMatriz de rigidez reduzida: ")
 #   sp.pprint(K_reduzido)
    
    K_reduzido_gauss_reduzido, novo_no = matrix_Ke_reduzido_elemento(K_global_gauss_reduzido, constrained_dofs)
#    print("\nMatriz de rigidez reduzida gauss reduzido: ")
 #   sp.pprint(K_reduzido_gauss_reduzido)

    F_reduzido = F_global.extract(novo_no, [0])
#    print("\nVetor de forças reduzido: ")
#    sp.pprint(F_reduzido)

    u_reduzido = deslocamento_reduzido(K_reduzido, F_reduzido)
 #   print("\nVetor de deslocamento reduzido: ")
#    sp.pprint(u_reduzido)
    
    u_reduzido_gauss_reduzido = deslocamento_reduzido(K_reduzido_gauss_reduzido, F_reduzido)
 #   print("\nVetor de deslocamento reduzido gauss reduzido: ")
 #   sp.pprint(u_reduzido_gauss_reduzido)

    u_global = deslocamento_global(u_reduzido, novo_no, total_no)
    print("\nVetor de deslocamento global u:")
    sp.pprint(u_global)
    
    u_global_gauss_reduzido = deslocamento_global(u_reduzido_gauss_reduzido, novo_no, total_no)
    print("\nVetor de deslocamento global u gauss reduzido:")
    sp.pprint(u_global_gauss_reduzido)

    print("\nDeformações e tensões em cada elemento:")
        
    for idx, elem in enumerate(elementos):
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        u_local = sp.Matrix(2*len(elem), 1, lambda i, _: u_global[elem[i//2]*2 + (i % 2)])
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
        defor_nos = deformacao_nos(u_local, n, e, local_coords, gauss_pontos)
        for i, row in enumerate(defor_nos.tolist(), start=1):
            print(f"Nó {i}: {row}")
            
    for idx, elem in enumerate(elementos):
        local_coords = sp.Matrix([global_coords[i, :] for i in elem])
        u_local_gauss_reduzido = sp.Matrix(2*len(elem), 1, lambda i, _: u_global_gauss_reduzido[elem[i//2]*2 + (i % 2)])
        defor_gauss_reduzido = deformacao_gauss_reduzido(u_local_gauss_reduzido, n, e, local_coords, gauss_pontos_reduzido)
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
        print("Deformações nos nós do elemento:")
        defor_nos = deformacao_nos_reduzido(u_local, n, e, local_coords, gauss_pontos_reduzido)
        for i, row in enumerate(defor_nos.tolist(), start=1):
            print(f"Nó {i}: {row}")
            
    rho = 7850   
    M_global = matriz_Me_global(global_coords, elementos, n, e, rho, espessura, w_dict, gauss_pontos)
#    print("\nMatriz de massa global M:")
#    sp.pprint(M_global)
    
    M_global_gauss_reduzido = matriz_Me_global(global_coords, elementos, n, e, rho, espessura, W_reduzido, gauss_pontos_reduzido)
#    print("\nMatriz de massa global M gauss reduzido:")
#    sp.pprint(M_global_gauss_reduzido)
    
    M_reduzida = matrix_Me_reduzido(M_global, constrained_dofs)
 #   print("\nMatriz de massa reduzida M_reduzida:")
 #   sp.pprint(M_reduzida)
    
    M_reduzida_gauss_reduzido = matrix_Me_reduzido(M_global_gauss_reduzido, constrained_dofs)
 #   print("\nMatriz de massa reduzida M_reduzida gauss reduzido:")
 #   sp.pprint(M_reduzida_gauss_reduzido)

    # Conversão para arrays numéricos para plotagem
    n_nos = global_coords.shape[0]
    orig_nodes = np.array([[float(global_coords[i, 0]), float(global_coords[i, 1])] for i in range(n_nos)])
    u_global_num = np.array([float(u_global[i]) for i in range(u_global.shape[0])]).reshape(n_nos, 2)
    u_global_num_gauss_reduzido = np.array([float(u_global_gauss_reduzido[i]) for i in range(u_global_gauss_reduzido.shape[0])]).reshape(n_nos, 2)
    
    escala = 1  
    def_nodes = orig_nodes + escala * u_global_num
    def_nodes_gauss_reduzido = orig_nodes + escala * u_global_num_gauss_reduzido

    F_num = np.array([float(F_global[i]) for i in range(F_global.shape[0])]).reshape(n_nos, 2)
    fx, fy = F_num[:, 0], F_num[:, 1]

    constr_x = sorted({dof // 2 for dof in constrained_dofs if dof % 2 == 0})
    constr_y = sorted({dof // 2 for dof in constrained_dofs if dof % 2 == 1})

    plt.figure(figsize=(12, 6))
    
    # --- Gauss completo ---
    plt.subplot(1, 2, 1)
    plt.scatter(orig_nodes[:, 0], orig_nodes[:, 1], c='blue', label='Nós Originais')
    plt.scatter(def_nodes[:, 0], def_nodes[:, 1], c='red',  label='Nós Deformados (Gauss)')
    # Desenha elementos (usando os 4 primeiros nós como cantos do elemento)
    for elem in elementos:
        elem_corners = elem[:4]
        xo = [orig_nodes[i, 0] for i in elem_corners] + [orig_nodes[elem_corners[0], 0]]
        yo = [orig_nodes[i, 1] for i in elem_corners] + [orig_nodes[elem_corners[0], 1]]
        plt.plot(xo, yo, 'b--', linewidth=1)
        xd = [def_nodes[i, 0] for i in elem_corners] + [def_nodes[elem_corners[0], 0]]
        yd = [def_nodes[i, 1] for i in elem_corners] + [def_nodes[elem_corners[0], 1]]
        plt.plot(xd, yd, 'r-',  linewidth=1)
    # Forças nodais
    plt.quiver(orig_nodes[:, 0], orig_nodes[:, 1], fx, fy,
               color='black', angles='xy', scale_units='xy',
               scale=15, width=0.005, label='Forças Nodal')
    # Restrições X (setas horizontais)
    if constr_x:
        plt.quiver(orig_nodes[constr_x, 0], orig_nodes[constr_x, 1],
                   np.ones(len(constr_x)), np.zeros(len(constr_x)),
                   color='black', angles='xy', scale_units='xy',
                   scale=20, width=0.002, label='Restrição X')
    # Restrições Y (setas verticais)
    if constr_y:
        plt.quiver(orig_nodes[constr_y, 0], orig_nodes[constr_y, 1],
                   np.zeros(len(constr_y)), np.ones(len(constr_y)),
                   color='black', angles='xy', scale_units='xy',
                   scale=20, width=0.002, label='Restrição Y')
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Malha Original e Deformada (Gauss completo)")
    plt.legend(); plt.axis('equal')

    # --- Gauss reduzido ---
    plt.subplot(1, 2, 2)
    plt.scatter(orig_nodes[:, 0], orig_nodes[:, 1], c='blue', label='Nós Originais')
    plt.scatter(def_nodes_gauss_reduzido[:, 0], def_nodes_gauss_reduzido[:, 1],
                c='green', label='Nós Deformados (Gauss Reduzido)')
    for elem in elementos:
        elem_corners = elem[:4]
        xo = [orig_nodes[i, 0] for i in elem_corners] + [orig_nodes[elem_corners[0], 0]]
        yo = [orig_nodes[i, 1] for i in elem_corners] + [orig_nodes[elem_corners[0], 1]]
        plt.plot(xo, yo, 'b--', linewidth=1)
        xg = [def_nodes_gauss_reduzido[i, 0] for i in elem_corners] + [def_nodes_gauss_reduzido[elem_corners[0], 0]]
        yg = [def_nodes_gauss_reduzido[i, 1] for i in elem_corners] + [def_nodes_gauss_reduzido[elem_corners[0], 1]]
        plt.plot(xg, yg, 'g-', linewidth=1)
    # Forças nodais
    plt.quiver(orig_nodes[:, 0], orig_nodes[:, 1], fx, fy,
               color='black', angles='xy', scale_units='xy',
               scale=15, width=0.005)
    # Restrições X (setas horizontais)
    if constr_x:
        plt.quiver(orig_nodes[constr_x, 0], orig_nodes[constr_x, 1],
                   np.ones(len(constr_x)), np.zeros(len(constr_x)),
                   color='black', angles='xy', scale_units='xy',
                   scale=20, width=0.002)
    # Restrições Y (setas verticais)
    if constr_y:
        plt.quiver(orig_nodes[constr_y, 0], orig_nodes[constr_y, 1],
                   np.zeros(len(constr_y)), np.ones(len(constr_y)),
                   color='black', angles='xy', scale_units='xy',
                   scale=20, width=0.002)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Malha Original e Deformada (Gauss Reduzido)")
    plt.legend(); plt.axis('equal')

    plt.tight_layout()
    plt.show()

