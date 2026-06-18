
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import os
import warnings
from tqdm import tqdm
from k_means_constrained import KMeansConstrained
import networkx as nx
import metis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')
CAMINHO_VAGAS        = '../df_vagas_2023.csv'
CAMINHO_CONCORRENCIA = '../df_concorrencia_2023.csv'
LISTA_K              = [2, 3, 4, 10, 50, 100, 500, 1000]
def carregar_dados():
    print("[1/3] Lendo arquivos originais do SISU...")
    if not os.path.exists(CAMINHO_VAGAS) or not os.path.exists(CAMINHO_CONCORRENCIA):
        raise FileNotFoundError(
            "Certifique-se de que os arquivos de vagas e concorrência estão "
            "no caminho configurado."
        )

    df_vagas = pd.read_csv(CAMINHO_VAGAS, usecols=['ID_CURSO', 'QT_VAGAS_CONCORRENCIA'])
    df_vagas = df_vagas[df_vagas['QT_VAGAS_CONCORRENCIA'] > 0].copy()
    cursos_lista = sorted(df_vagas['ID_CURSO'].unique().tolist())

    df_cand = pd.read_csv(
        CAMINHO_CONCORRENCIA,
        usecols=['ID_CANDIDATO', 'ID_CURSO', 'OPCAO'],
        dtype={'ID_CANDIDATO': str, 'ID_CURSO': int, 'OPCAO': int}
    )
    df_cand = df_cand[df_cand['OPCAO'].isin([1, 2])].copy()
    df_cand = df_cand.drop_duplicates(subset=['ID_CANDIDATO', 'OPCAO'], keep='first')
    df_cand = df_cand.sort_values(['ID_CANDIDATO', 'OPCAO'])

    print(f"      Cursos com vagas : {len(cursos_lista):,}")
    print(f"      Candidatos       : {df_cand['ID_CANDIDATO'].nunique():,}")

    return cursos_lista, df_cand
def construir_estruturas_topologicas(cursos_lista, df_cand):
    print("[2/3] Filtrando candidatos e construindo grafo de co-inscrição...")
    curso2idx = {c: i for i, c in enumerate(cursos_lista)}

    candidatos_por_curso = (
        df_cand[df_cand['ID_CURSO'].isin(curso2idx)]
        .groupby('ID_CURSO')['ID_CANDIDATO']
        .nunique()
        .to_dict()
    )

    cands_escolhas = (
        df_cand.groupby('ID_CANDIDATO')['ID_CURSO']
        .apply(list)
        .to_dict()
    )

    pares_validos = [
        (cursos[0], cursos[1])
        for cursos in cands_escolhas.values()
        if len(cursos) == 2
        and cursos[0] != cursos[1]
        and cursos[0] in curso2idx
        and cursos[1] in curso2idx
    ]
    print(f"      Candidatos com 2 opções distintas: {len(pares_validos):,}")

    G_base = nx.Graph()
    G_base.add_nodes_from(cursos_lista)
    for c in cursos_lista:
        G_base.nodes[c]['weight'] = candidatos_por_curso.get(c, 1)

    rows, cols = [], []
    for c1, c2 in pares_validos:
        i1, i2 = curso2idx[c1], curso2idx[c2]
        rows.extend([i1, i2])
        cols.extend([i2, i1])

        if G_base.has_edge(c1, c2):
            G_base[c1][c2]['weight'] += 1
        else:
            G_base.add_edge(c1, c2, weight=1)

    data = np.ones(len(rows), dtype=np.int32)
    n = len(cursos_lista)
    M_co = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int32)
    M_co.sum_duplicates()

    D = M_co.toarray().astype(np.float32)
    normas = np.linalg.norm(D, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    features_kmeans = D / normas

    return features_kmeans, G_base, pares_validos, candidatos_por_curso
def rodar_experimento(cursos_lista, features_km, G_base, pares_validos,
                       candidatos_por_curso):
    print("[3/3] Executando pipeline de clusterização (K-Means vs METIS)...")

    res = {
        'k': [],
        'km_front_abs': [], 'km_front_pct': [], 'km_internos': [],
        'mt_front_abs': [], 'mt_front_pct': [], 'mt_internos': [],
    }

    n_total_validos = len(pares_validos)
    print(f"      Total de candidatos com concorrência dupla analisados: "
          f"{n_total_validos:,}")

    nodes_list  = list(G_base.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    G_metis     = nx.relabel_nodes(G_base, node_to_idx)
    G_metis.graph['node_weight_attr'] = 'weight'
    G_metis.graph['edge_weight_attr'] = 'weight'

    print("\n" + "=" * 78)
    print(f"{'K':<6} | {'K-Means Front.':<15} | {'K-Means Int.':<14} | "
          f"{'METIS Front.':<14} | {'METIS Int.':<12}")
    print("-" * 78)

    for k in tqdm(LISTA_K, desc="Variando K"):
        res['k'].append(k)
        try:
            min_sz = max(1, len(cursos_lista) // (k + 1))
            max_sz = max(1, int(np.ceil(len(cursos_lista) / k * 1.5)))

            km = KMeansConstrained(
                n_clusters=k, size_min=min_sz, size_max=max_sz,
                n_init=2, random_state=42
            )
            labels_km = km.fit_predict(features_km)

            c2c_km = {cursos_lista[i]: int(labels_km[i]) for i in range(len(cursos_lista))}

            fronteira_km = sum(
                c2c_km.get(c1) != c2c_km.get(c2) for c1, c2 in pares_validos
            )
            internos_km = n_total_validos - fronteira_km

            res['km_front_abs'].append(fronteira_km)
            res['km_front_pct'].append((fronteira_km / n_total_validos) * 100)
            res['km_internos'].append(internos_km)
        except Exception as e:
            print(f"\n      [Erro K-Means K={k}]: {e}")
            res['km_front_abs'].append(np.nan)
            res['km_front_pct'].append(np.nan)
            res['km_internos'].append(np.nan)
        try:
            if k >= G_metis.number_of_nodes():
                raise ValueError(f"K={k} maior ou igual ao número de nós.")

            _, parts = metis.part_graph(
                G_metis, nparts=k, objtype='cut', ncuts=3, niter=10
            )

            c2c_metis = {nodes_list[idx]: parts[idx] for idx in range(len(nodes_list))}

            fronteira_metis = sum(
                c2c_metis.get(c1) != c2c_metis.get(c2)
                for c1, c2 in pares_validos
                if c1 in c2c_metis and c2 in c2c_metis
            )
            internos_metis = n_total_validos - fronteira_metis

            res['mt_front_abs'].append(fronteira_metis)
            res['mt_front_pct'].append((fronteira_metis / n_total_validos) * 100)
            res['mt_internos'].append(internos_metis)
        except Exception as e:
            print(f"\n      [Erro METIS K={k}]: {e}")
            res['mt_front_abs'].append(np.nan)
            res['mt_front_pct'].append(np.nan)
            res['mt_internos'].append(np.nan)
        km_f = res['km_front_abs'][-1]
        km_i = res['km_internos'][-1]
        mt_f = res['mt_front_abs'][-1]
        mt_i = res['mt_internos'][-1]
        print(f"{k:<6} | {km_f:<15,.0f} | {km_i:<14,.0f} | "
              f"{mt_f:<14,.0f} | {mt_i:<12,.0f}")

    print("=" * 78)

    return res
def gerar_grafico_simples(res):
    print("\n[Plot] Salvando dados brutos em CSV para backup...")
    df_backup = pd.DataFrame(res)
    os.makedirs('figuras/resultados', exist_ok=True)
    df_backup.to_csv('figuras/resultados/comparativo_kmeans_metis.csv', index=False)

    print("[Plot] Gerando gráfico simplificado...")
    style = ('seaborn-v0_8-whitegrid'
              if 'seaborn-v0_8-whitegrid' in plt.style.available
              else 'default')
    plt.style.use(style)

    k_validos    = [res['k'][i] for i in range(len(res['k']))
                     if not np.isnan(res['km_front_abs'][i])]
    km_front_abs = [x for x in res['km_front_abs'] if not np.isnan(x)]
    mt_front_abs = [x for x in res['mt_front_abs'] if not np.isnan(x)]
    km_internos  = [x for x in res['km_internos'] if not np.isnan(x)]
    mt_internos  = [x for x in res['mt_internos'] if not np.isnan(x)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_validos, km_front_abs, 'o-', color='#b91c1c',
            linewidth=2.5, markersize=7, label='K-Means: Fronteira')
    ax.plot(k_validos, mt_front_abs, 's-', color='#1e3a8a',
            linewidth=2.5, markersize=7, label='METIS: Fronteira')
    ax.plot(k_validos, km_internos, 'o--', color='#f59e0b',
            linewidth=2.5, markersize=7, label='K-Means: Internos')
    ax.plot(k_validos, mt_internos, 's--', color='#10b981',
            linewidth=2.5, markersize=7, label='METIS: Internos')

    ax.set_xlabel('Número de Clusters ($K$)', fontsize=11)
    ax.set_ylabel('Número de Candidatos', fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks(LISTA_K)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax.set_title('Candidatos de Fronteira e Internos: K-Means Constrained vs METIS',
                 fontsize=12, weight='bold', pad=15)
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    caminho = 'figuras/resultados/comparativo_kmeans_metis.png'
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo em: '{caminho}'")
if __name__ == "__main__":
    cursos_lista, df_cand = carregar_dados()
    features_km, G_base, pares_validos, candidatos_por_curso = \
        construir_estruturas_topologicas(cursos_lista, df_cand)

    resultados = rodar_experimento(
        cursos_lista, features_km, G_base, pares_validos, candidatos_por_curso
    )

    gerar_grafico_simples(resultados)

    print("\n[EXPERIMENTO FINALIZADO]")