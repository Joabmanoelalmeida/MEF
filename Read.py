import numpy as np
from typing import List, Tuple, Dict

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

        if l.startswith('*'):
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

    print("✅ Arquivo lido com sucesso!")

    print("Número de nós:", coords.shape[0])
    print("Coordenadas:")
    print(coords)

    print("Número de elementos:", len(conn))
    print("Conexões:")
    for i, c in enumerate(conn, start=1):
        print(f"Elemento {i}: {c}")

    print("Espessura:", esp)
    print("Tipo de análise plana:", plane_state)

    print("Graus de liberdade restritos:", gdl_restritos)

    print("Módulos de elasticidade:")
    print(Elist)

    print("Coeficientes de Poisson:")
    print(nulist)

    print("Densidades:")
    print(rholist)

    print("Forças nodais:")
    for i, valor in enumerate(f):
        print(f"GDL {i + 1}: {valor}")