import pandas as pd
import numpy as np
import networkx as nx
import metis
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import matplotlib.ticker as ticker
CAMINHO_VAGAS        = '../df_vagas_2023.csv'
CAMINHO_CONCORRENCIA = '../df_concorrencia_2023.csv'

df_vagas = pd.read_csv(CAMINHO_VAGAS, usecols=['ID_CURSO', 'NO_CURSO'])
df_conc  = pd.read_csv(
    CAMINHO_CONCORRENCIA,
    usecols=['ID_CANDIDATO', 'ID_CURSO', 'OPCAO'],
    dtype={'ID_CANDIDATO': str, 'ID_CURSO': int, 'OPCAO': int}
)
df_conc = df_conc[df_conc['OPCAO'].isin([1, 2])].copy()
df_conc = df_conc.drop_duplicates(subset=['ID_CANDIDATO', 'OPCAO'], keep='first')
df_conc = df_conc.sort_values(['ID_CANDIDATO', 'OPCAO'])
ids_medicina = set(
    df_vagas[df_vagas['NO_CURSO'].str.upper() == 'MEDICINA']['ID_CURSO'].unique()
)
print(f"  Cursos de Medicina identificados: {len(ids_medicina)}")
print(f"  Candidatos totais               : {df_conc['ID_CANDIDATO'].nunique():,}")
print(f"  Inscrições totais               : {len(df_conc):,}")

valores_k = [2, 3, 4, 10, 50, 100, 500, 1000]
def executar_experimento_metis(df_base, ks, nome_cenario=""):
    df_limpo = (df_base
                .drop_duplicates(subset=['ID_CANDIDATO', 'OPCAO'], keep='first')
                .sort_values(['ID_CANDIDATO', 'OPCAO'])
                .copy())
    cands_escolhas = (
        df_limpo
        .groupby('ID_CANDIDATO')['ID_CURSO']
        .apply(list)
        .to_dict()
    )
    pares_validos = {
        cid: cursos
        for cid, cursos in cands_escolhas.items()
        if len(cursos) == 2 and cursos[0] != cursos[1]
    }
    print(f"\n  [{nome_cenario}] Candidatos com 2 opções distintas: "
          f"{len(pares_validos):,}")
    G = nx.Graph()
    for cursos in pares_validos.values():
        c1, c2 = cursos[0], cursos[1]
        if G.has_edge(c1, c2):
            G[c1][c2]['weight'] += 1
        else:
            G.add_edge(c1, c2, weight=1)

    print(f"  [{nome_cenario}] Nós (cursos) : {G.number_of_nodes():,}")
    print(f"  [{nome_cenario}] Arestas      : {G.number_of_edges():,}")
    nodes_list  = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    G_metis     = nx.relabel_nodes(G, node_to_idx)

    fronteiras_por_k = {}

    for k in tqdm(ks, desc=f"Particionando com METIS [{nome_cenario}]"):
        n_nos = G_metis.number_of_nodes()
        if k >= n_nos:
            print(f"\n  [Aviso] K={k} >= número de nós ({n_nos}). Pulando...")
            continue
        edgecuts, parts = metis.part_graph(
            G_metis,
            nparts=k,
            objtype='cut',
            ncuts=3,    
            niter=10    
        )
        curso_para_cluster = {
            nodes_list[idx]: parts[idx]
            for idx in range(len(nodes_list))
        }
        total_fronteira = sum(
            1
            for cursos in pares_validos.values()
            if (cursos[0] in curso_para_cluster
                and cursos[1] in curso_para_cluster
                and curso_para_cluster[cursos[0]] != curso_para_cluster[cursos[1]])
        )

        fronteiras_por_k[k] = total_fronteira

    return fronteiras_por_k
print("\n>> Executando Cenário 1: Todos os Cursos (Com Medicina)...")
resultados_com_medicina = executar_experimento_metis(
    df_conc, valores_k, nome_cenario="Com Medicina"
)

print("\n>> Executando Cenário 2: Expurgo de Medicina...")
df_conc_sem_med = df_conc[~df_conc['ID_CURSO'].isin(ids_medicina)].copy()
resultados_sem_medicina = executar_experimento_metis(
    df_conc_sem_med, valores_k, nome_cenario="Sem Medicina"
)
ks_finais = [
    k for k in valores_k
    if k in resultados_com_medicina and k in resultados_sem_medicina
]

print("\n--- TABELA RESUMO DE CANDIDATOS DE FRONTEIRA ---")
print(f"{'K':<6} | {'Com Medicina':<15} | {'Sem Medicina':<15} | "
      f"{'Redução Absoluta':<18} | {'Redução (%)':<12}")
print("-" * 75)
for k in ks_finais:
    com = resultados_com_medicina[k]
    sem = resultados_sem_medicina[k]
    dif = com - sem
    pct = (dif / com * 100) if com > 0 else 0
    print(f"{k:<6} | {com:<15,} | {sem:<15,} | {dif:<18,} | {pct:<12.1f}")
print("\nGerando gráfico comparativo...")

linhas_com = [resultados_com_medicina[k] for k in ks_finais]
linhas_sem = [resultados_sem_medicina[k] for k in ks_finais]

estilo = ('seaborn-v0_8-whitegrid'
          if 'seaborn-v0_8-whitegrid' in plt.style.available
          else 'default')
plt.style.use(estilo)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(ks_finais, linhas_com, 'o-',
        color='#1e3a8a', label='Todos os Cursos (Com Medicina)',
        linewidth=2.5, markersize=7)
ax.plot(ks_finais, linhas_sem, 's--',
        color='#b91c1c', label='Sem Cursos de Medicina',
        linewidth=2.5, markersize=7)

ax.set_title(
    "Comportamento do Volume de Candidatos de Fronteira\n",
    fontsize=12, weight='bold', pad=15
)
ax.set_xlabel("Número de Clusters ($K$)", fontsize=11)
ax.set_ylabel("Candidatos de Fronteira", fontsize=11)
ax.set_xscale('log')
ax.set_xticks(ks_finais)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ','))
)
for k_val in [min(ks_finais), max(ks_finais)]:
    if k_val in resultados_com_medicina:
        ax.annotate(
            f"{resultados_com_medicina[k_val]:,}",
            (k_val, resultados_com_medicina[k_val]),
            textcoords="offset points", xytext=(0, 10),
            ha='center', fontweight='bold', color='#1e3a8a', fontsize=9
        )
    if k_val in resultados_sem_medicina:
        ax.annotate(
            f"{resultados_sem_medicina[k_val]:,}",
            (k_val, resultados_sem_medicina[k_val]),
            textcoords="offset points", xytext=(0, -16),
            ha='center', fontweight='bold', color='#b91c1c', fontsize=9
        )
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
os.makedirs('figuras/resultados', exist_ok=True)
caminho = 'figuras/resultados/comparativo_fronteira_metis.jpg'
plt.savefig(caminho, dpi=300, bbox_inches='tight')
plt.close()
print(f">> Gráfico salvo em '{caminho}'")
