import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

CAMINHO_VAGAS = '../df_vagas_2023.csv'
CAMINHO_CONCORRENCIA = '../df_concorrencia_2023.csv'

def gerar_grafico_relacionados():
    try:
        df_vagas = pd.read_csv(CAMINHO_VAGAS)
        map_nomes = {}
        for _, r in df_vagas.iterrows():
            map_nomes[int(r['ID_CURSO'])] = f"{r['NO_CURSO']} ({r['SG_IES']} - {r['NO_CAMPUS']})"
    except FileNotFoundError:
        print(f"Erro: O arquivo '{CAMINHO_VAGAS}' não foi encontrado.")
        return

    try:
        df_conc = pd.read_csv(CAMINHO_CONCORRENCIA, usecols=['ID_CANDIDATO', 'ID_CURSO', 'OPCAO'])
        df_conc = df_conc.drop_duplicates(subset=['ID_CANDIDATO', 'OPCAO'])
    except FileNotFoundError:
        print(f"Erro: O arquivo '{CAMINHO_CONCORRENCIA}' não foi encontrado.")
        return
    cands = df_conc.groupby('ID_CANDIDATO')['ID_CURSO'].apply(list).to_dict()

    ids_alvo = [3376, 717, 3434]
    relacionamentos = {id_alvo: defaultdict(int) for id_alvo in ids_alvo}

    for cursos in cands.values():
        if len(cursos) == 2:
            c1, c2 = int(cursos[0]), int(cursos[1])
            if c1 != c2:
                if c1 in relacionamentos:
                    relacionamentos[c1][c2] += 1
                if c2 in relacionamentos:
                    relacionamentos[c2][c1] += 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    for idx_hub, id_alvo in enumerate(ids_alvo):
        nome_completo_alvo = map_nomes.get(id_alvo, f"ID {id_alvo}")
        nome_alvo = nome_completo_alvo.split(' (')[0]
        sub_df_vagas = df_vagas[df_vagas['ID_CURSO'] == id_alvo]
        sigla_ies = sub_df_vagas.iloc[0]['SG_IES'] if not sub_df_vagas.empty else "N/A"
        titulo_aba = f"{nome_alvo} ({sigla_ies})"
        
        top5 = sorted(relacionamentos[id_alvo].items(), key=lambda x: x[1], reverse=True)[:5]
        
        dados_sub = []
        for id_viz, qtd in top5:
            nome_terminal = map_nomes.get(id_viz, f"ID {id_viz}")
            if " (" in nome_terminal:
                parte_curso = nome_terminal.split(' (')[0].replace("MEDICINA", "MED").replace("ENFERMAGEM", "ENF")
                parte_ies_campus = nome_terminal.split(' (')[1].replace(')', '').replace(' - ', '-')
                if "UNIVERSIDADE FEDERAL" in parte_ies_campus:
                    parte_ies_campus = parte_ies_campus.replace("UNIVERSIDADE FEDERAL DO ", "UF").replace("UNIVERSIDADE FEDERAL DE ", "UF").replace("UNIVERSIDADE FEDERAL ", "UF")
                nome_grafico = f"{parte_curso} ({parte_ies_campus})"
            else:
                nome_grafico = nome_terminal
            dados_sub.append({'Curso': nome_grafico, 'Candidatos': qtd})
            
        df_sub = pd.DataFrame(dados_sub)
        
        ax = axes[idx_hub]
        sns.barplot(data=df_sub, x="Candidatos", y="Curso", ax=ax, palette="Blues_r", 
                    hue="Curso", legend=False, edgecolor='#1e293b', width=0.6)
        
        ax.set_title(titulo_aba.upper(), pad=12, fontweight='bold', fontsize=13, color='#1e3a8a', loc='left')
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("", fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=10)
        
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 15, p.get_y() + p.get_height()/2, f'{int(width):,}', 
                    va='center', ha='left', fontsize=11, weight='bold', color='#0f172a')

    fig.text(0.5, 0.04, 'Volume de Candidatos Compartilhados (Peso da Aresta)', ha='center', fontsize=12, weight='bold', color='#334155')

    
    plt.subplots_adjust(hspace=0.45, bottom=0.1)
    
    os.makedirs('figuras/resultados', exist_ok=True)
    caminho_saida = 'figuras/resultados/top_relacionados_hubs.jpg'
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    gerar_grafico_relacionados()