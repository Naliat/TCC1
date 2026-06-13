import pandas as pd

CAMINHO_CONCORRENCIA = 'df_concorrencia_2023.csv'
CAMINHO_VAGAS = 'df_vagas_2023.csv'
def identificar_curso_completo():
    try:
        id_busca = input("Digite o ID_CURSO que deseja identificar ex: 3376 medicina ufmg: ").strip()
        if not id_busca:
            return
        id_busca = int(id_busca)
    except ValueError:
        print("Erro: Digite um número inteiro válido.")
        return

    try:
        df_vagas = pd.read_csv(CAMINHO_VAGAS)
        resultado_vagas = df_vagas[df_vagas['ID_CURSO'] == id_busca]
        
        df_conc = pd.read_csv(CAMINHO_CONCORRENCIA, usecols=['ID_CANDIDATO', 'ID_CURSO'])
        resultado_conc = df_conc[df_conc['ID_CURSO'] == id_busca]
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. {e}")
        return

    if resultado_vagas.empty:
        print(f"\n[!] O ID_CURSO {id_busca} não foi encontrado no arquivo de vagas.")
        return

    info = resultado_vagas.iloc[0]
    total_inscritos = int(resultado_conc['ID_CANDIDATO'].nunique())
    print(f"        CURSO E FACULDADE IDENTIFICADOS PARA O ID: {id_busca}")
    print(f"Curso        : {info.get('NO_CURSO', 'Não encontrado')}")
    print(f"Grau         : {info.get('DS_GRAU', 'Não encontrado')} ({info.get('DS_TURNO', 'Não encontrado')})")
    print(f"Instituição  : {info.get('NO_IES', 'Não encontrado')} ({info.get('SG_IES', 'Não encontrado')})")
    print(f"Organização  : {info.get('DS_ORGANIZACAO_ACADEMICA', 'Não encontrado')} - {info.get('DS_CATEGORIA_ADM', 'Não encontrado')}")
    print(f"Campus       : {info.get('NO_CAMPUS', 'Não encontrado')}")
    print(f"Localização  : {info.get('NO_MUNICIPIO_CAMPUS', 'Não encontrado')} - {info.get('SG_UF_CAMPUS', 'Não encontrado')} ({info.get('DS_REGIAO', 'Não encontrado')})")
    print("-"*70)
    print(f"Total de Candidatos Inscritos : {total_inscritos:,}")
    print(f"Vagas Totais Autorizadas      : {info.get('NU_VAGAS_AUTORIZADAS', 'N/A')}")


if __name__ == "__main__":
    identificar_curso_completo()