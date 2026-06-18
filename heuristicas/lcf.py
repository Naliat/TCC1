import pandas as pd
import numpy as np
from collections import deque, defaultdict
import heapq
import time

CAMINHO_VAGAS        = 'df_vagas_2023.csv'
CAMINHO_CONCORRENCIA = 'df_concorrencia_2023.csv'
VERBOSE              = True
N_ITER_VERBOSE        = 30
REMANEJAMENTO = {
    'LB_PPI':     ['LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_I':       ['LB_I','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LI_PPI':     ['LI_PPI','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_Q','LI_PCD','LI_EP','AC'],
    'LI_Q':       ['LI_Q','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_PCD','LI_EP','AC'],
    'LI_PCD':     ['LI_PCD','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_EP','AC'],
    'LI_EP':      ['LI_EP','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','AC'],
    'LI_PP':      ['LI_PP','LB_PPI','LB_PP','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LI_PCD_PPI': ['LI_PCD_PPI','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LI_PCD_PP':  ['LI_PCD_PP','LB_PP','LB_PCD','LB_EP','LI_PP','LI_PCD','LI_EP','AC'],
    'LB_Q':       ['LB_Q','LB_PPI','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_PCD':     ['LB_PCD','LB_PPI','LB_Q','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_EP':      ['LB_EP','LB_PPI','LB_Q','LB_PCD','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_PP':      ['LB_PP','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_PP','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_PCD_PPI': ['LB_PCD_PPI','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PCD_PPI','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'LB_PCD_PP':  ['LB_PCD_PP','LB_PP','LB_PCD','LB_EP','LI_PCD_PP','LI_PP','LI_PCD','LI_EP','AC'],
    'LI_I':       ['LI_I','LB_PPI','LB_Q','LB_PCD','LB_EP','LI_PPI','LI_Q','LI_PCD','LI_EP','AC'],
    'B1': ['AC'],
    'B2': ['AC'],
    'AC': ['AC'],
}

ORDEM_GLOBAL_SISU = [
    'AC','LB_PPI','LB_PP','LB_I','LB_Q','LB_PCD','LB_EP',
    'LI_PPI','LI_PP','LI_I','LI_Q','LI_PCD','LI_EP',
    'LI_PCD_PPI','LI_PCD_PP','LB_PCD_PPI','LB_PCD_PP',
    'B1','B2',
]

_MOD_TO_INT = {m: i for i, m in enumerate(ORDEM_GLOBAL_SISU)}
_INT_TO_MOD = {i: m for m, i in _MOD_TO_INT.items()}
_N_MODS     = len(ORDEM_GLOBAL_SISU)

_PRIO_MATRIX = np.full((_N_MODS, _N_MODS), 999, dtype=np.int32)
for mod_vaga, prioridades in REMANEJAMENTO.items():
    if mod_vaga not in _MOD_TO_INT:
        continue
    iv = _MOD_TO_INT[mod_vaga]
    for mod_orig in prioridades:
        if mod_orig in _MOD_TO_INT:
            io = _MOD_TO_INT[mod_orig]
            _PRIO_MATRIX[iv, io] = prioridades.index(mod_orig)

MOD_ESPECIFICAS     = {'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'}
MOD_BX               = {'B1', 'B2'}
AC_INT               = _MOD_TO_INT['AC']
MOD_ESPECIFICAS_INT  = {_MOD_TO_INT[m] for m in MOD_ESPECIFICAS if m in _MOD_TO_INT}

_NOTAS_COLS = ['NOTA_CANDIDATO', 'NOTA_R', 'NOTA_L', 'NOTA_M', 'NOTA_CN', 'NOTA_CH']
class CandidatosArray:
    def __init__(self, n: int):
        self.n            = n
        self.tem_op        = np.zeros((n, 2), dtype=bool)
        self.id_curso      = np.full((n, 2), -1, dtype=np.int64)
        self.modalidade    = np.full((n, 2), -1, dtype=np.int32)
        self.notas         = np.zeros((n, 2, 6), dtype=np.float64)
        self.notas_tuple   = [[None, None] for _ in range(n)]
        self.ids           = [''] * n

    def get_mod_int(self, idx: int, opcao: int) -> int:
        return int(self.modalidade[idx, opcao])

    def get_notas(self, idx: int, opcao: int) -> np.ndarray:
        return self.notas[idx, opcao]

    def opcao_para_curso(self, idx: int, id_curso: int) -> int:
        if self.tem_op[idx, 0] and self.id_curso[idx, 0] == id_curso:
            return 0
        if self.tem_op[idx, 1] and self.id_curso[idx, 1] == id_curso:
            return 1
        return -1
def comparar_notas_np(na, nb) -> bool:
    for i in range(6):
        if na[i] > nb[i]:
            return True
        if na[i] < nb[i]:
            return False
    return False


def compute_score_alg2(mod_int: int, notas: tuple, mod_vaga_int: int):
    if mod_vaga_int == AC_INT:
        return (1, notas) if mod_int == AC_INT else (1, notas)
    pa = 3 if mod_int == mod_vaga_int else (2 if mod_int != AC_INT else 1)
    return (pa, notas)
def elegivel_para_vaga(cand_idx: int, opcao: int, id_curso_vaga: int, mod_vaga_int: int,
                        cands: CandidatosArray, vagas_dict: dict) -> bool:
    chave = (id_curso_vaga, mod_vaga_int)
    if chave not in vagas_dict:
        return False

    if mod_vaga_int == AC_INT:
        return True

    if mod_vaga_int in MOD_ESPECIFICAS_INT:
        if cands.id_curso[cand_idx, opcao] != id_curso_vaga:
            return False
        mod_cand_int = int(cands.modalidade[cand_idx, opcao])
        return _PRIO_MATRIX[mod_vaga_int, mod_cand_int] < 999 if mod_cand_int >= 0 else False

    for op in (0, 1):
        if cands.tem_op[cand_idx, op]:
            mod_op = int(cands.modalidade[cand_idx, op])
            if mod_op >= 0 and mod_op not in MOD_ESPECIFICAS_INT:
                if _PRIO_MATRIX[mod_vaga_int, mod_op] < 999:
                    return True

    return False
def precalcular_razoes(vagas_dict: dict, elegibilidade_base: list) -> dict:
    contagens = defaultdict(int)
    for elegiveis_op0, elegiveis_op1 in elegibilidade_base:
        for cm in elegiveis_op0:
            contagens[cm] += 1
        for cm in elegiveis_op1:
            contagens[cm] += 1

    razao = {}
    for chave, cnt in contagens.items():
        vagas = vagas_dict.get(chave, 0)
        if vagas > 0:
            razao[chave] = cnt / vagas
    return razao
def build_all_propostas_lcf(cands: CandidatosArray, elegibilidade_base: list,
                             razao_por_vaga: dict) -> list:
    propostas_todas = []
    for idx in range(cands.n):
        lista = []
        for opcao, elegiveis in enumerate(elegibilidade_base[idx]):
            if not elegiveis:
                continue
            id_curso = elegiveis[0][0]
            nao_ac   = [cm for cm in elegiveis if cm[1] != AC_INT]
            tem_ac   = (id_curso, AC_INT) in elegiveis
            nao_ac.sort(key=lambda cm: razao_por_vaga.get(cm, float('inf')))
            if tem_ac:
                lista.append((id_curso, AC_INT))
            lista.extend(nao_ac)
        propostas_todas.append(lista)
    return propostas_todas
def carregar_candidatos(caminho: str) -> CandidatosArray:
    colunas = ['ID_CANDIDATO', 'ID_CURSO', 'OPCAO', 'MODALIDADE_INSCRICAO',
               'NOTA_CANDIDATO', 'NOTA_R', 'NOTA_L', 'NOTA_M', 'NOTA_CN', 'NOTA_CH']
    df = pd.read_csv(caminho, usecols=colunas, dtype={'ID_CANDIDATO': str})
    df['MODALIDADE_INSCRICAO'] = df['MODALIDADE_INSCRICAO'].replace({'B1': 'AC', 'B2': 'AC'})

    df = df[df['OPCAO'].isin([1, 2])]
    df = df.drop_duplicates(subset=['ID_CANDIDATO', 'OPCAO'], keep='first')
    df = df.sort_values(['ID_CANDIDATO', 'OPCAO'])

    df['MOD_INT'] = df['MODALIDADE_INSCRICAO'].map(_MOD_TO_INT).fillna(-1).astype(np.int32)

    ids_unicos = df['ID_CANDIDATO'].unique()
    n          = len(ids_unicos)
    id_to_idx  = {cid: i for i, cid in enumerate(ids_unicos)}
    df['CAND_IDX'] = df['ID_CANDIDATO'].map(id_to_idx)

    cands     = CandidatosArray(n)
    cands.ids = list(ids_unicos)

    for op in (1, 2):
        sub = df[df['OPCAO'] == op]
        if sub.empty:
            continue
        idxs    = sub['CAND_IDX'].values
        opcao_i = op - 1
        cands.tem_op[idxs, opcao_i]     = True
        cands.id_curso[idxs, opcao_i]   = sub['ID_CURSO'].values.astype(np.int64)
        cands.modalidade[idxs, opcao_i] = sub['MOD_INT'].values
        cands.notas[idxs, opcao_i, :]   = sub[_NOTAS_COLS].values.astype(np.float64)

        notas_arr = cands.notas[idxs, opcao_i, :]
        for i, cand_idx in enumerate(idxs):
            cands.notas_tuple[cand_idx][opcao_i] = tuple(notas_arr[i])

    return cands
def carregar_vagas(caminho: str) -> dict:
    df = pd.read_csv(caminho, usecols=['ID_CURSO', 'MODALIDADE', 'QT_VAGAS_CONCORRENCIA'])
    df = df[df['QT_VAGAS_CONCORRENCIA'] > 0].copy()
    df['MODALIDADE'] = df['MODALIDADE'].replace({'B1': 'AC', 'B2': 'AC'})
    df['MOD_INT']    = df['MODALIDADE'].map(_MOD_TO_INT)
    df = df.dropna(subset=['MOD_INT'])
    df['MOD_INT'] = df['MOD_INT'].astype(np.int32)

    return {
        (int(r.ID_CURSO), int(r.MOD_INT)): int(r.QT_VAGAS_CONCORRENCIA)
        for r in df.itertuples(index=False)
    }
def construir_elegibilidade_base(cands: CandidatosArray, vagas_dict: dict) -> list:
    elegibilidade_base = []
    for i in range(cands.n):
        elegiveis_op0, elegiveis_op1 = [], []
        for opcao in (0, 1):
            if cands.tem_op[i, opcao]:
                id_curso = int(cands.id_curso[i, opcao])
                for mod_int in range(_N_MODS):
                    if elegivel_para_vaga(i, opcao, id_curso, mod_int, cands, vagas_dict):
                        if opcao == 0:
                            elegiveis_op0.append((id_curso, mod_int))
                        else:
                            elegiveis_op1.append((id_curso, mod_int))
        elegibilidade_base.append((elegiveis_op0, elegiveis_op1))
    return elegibilidade_base
def gale_shapley(cands: CandidatosArray, vagas_dict: dict,
                  propostas: list, compute_score, verbose: bool = False):
    n            = cands.n
    idx_prop     = np.zeros(n, dtype=np.int32)
    alocado      = np.zeros(n, dtype=bool)
    alocacoes    = {k: [] for k in vagas_dict}
    capacidades  = dict(vagas_dict)
    fila         = deque(range(n))

    n_props = n_aloc_dir = n_trocas = n_sem_vaga = n_log = 0

    if verbose:
        print("\n" + "─" * 70)
        print("  LOG DAS PRIMEIRAS {} PROPOSTAS".format(N_ITER_VERBOSE))
        print("─" * 70)

    while fila:
        idx = fila.popleft()
        if alocado[idx]:
            continue
        if idx_prop[idx] >= len(propostas[idx]):
            continue

        id_curso, mod_vaga_int = propostas[idx][idx_prop[idx]]
        idx_prop[idx] += 1
        n_props += 1
        chave = (id_curso, mod_vaga_int)

        if chave not in alocacoes:
            n_sem_vaga += 1
            if idx_prop[idx] < len(propostas[idx]):
                fila.append(idx)
            continue

        op = 0 if cands.id_curso[idx, 0] == id_curso else 1

        melhor_mod  = int(cands.modalidade[idx, op])
        melhor_prio = _PRIO_MATRIX[mod_vaga_int, melhor_mod] if melhor_mod >= 0 else 999
        other_op    = 1 - op
        if cands.tem_op[idx, other_op]:
            mod_other = int(cands.modalidade[idx, other_op])
            if mod_other >= 0 and mod_other not in MOD_ESPECIFICAS_INT:
                if _PRIO_MATRIX[mod_vaga_int, mod_other] < melhor_prio:
                    melhor_mod = mod_other

        notas_novo = cands.notas_tuple[idx][op]
        score_novo = compute_score(melhor_mod, notas_novo, mod_vaga_int)
        item_novo  = (score_novo, idx, melhor_mod)

        lista = alocacoes[chave]
        cap   = capacidades[chave]

        if len(lista) < cap:
            heapq.heappush(lista, item_novo)
            alocado[idx] = True
            n_aloc_dir  += 1
            if verbose and n_log < N_ITER_VERBOSE:
                n_log += 1
                print(f"  [{n_log:2d}] ALOCADO DIRETAMENTE (vaga livre)")
                print(f"       Candidato: {cands.ids[idx][:14]}  "
                      f"mod={_INT_TO_MOD.get(melhor_mod, 'N/A')}  nota={notas_novo[0]:.1f}")
                print(f"       Vaga: ID_CURSO={id_curso}  "
                      f"Mod={_INT_TO_MOD[mod_vaga_int]}  ocup={len(lista)}/{cap}\n")
        else:
            if score_novo > lista[0][0]:
                pior_item = heapq.heappushpop(lista, item_novo)
                alocado[idx] = True
                pior_idx     = pior_item[1]
                alocado[pior_idx] = False
                n_trocas += 1
                if idx_prop[pior_idx] < len(propostas[pior_idx]):
                    fila.append(pior_idx)
                if verbose and n_log < N_ITER_VERBOSE:
                    n_log += 1
                    print(f"  [{n_log:2d}] TROCA")
                    print(f"       Entrou: {cands.ids[idx][:14]}  "
                          f"mod={_INT_TO_MOD.get(melhor_mod, 'N/A')}  nota={notas_novo[0]:.1f}")
                    print(f"       Saiu:   {cands.ids[pior_idx][:14]}  "
                          f"mod={_INT_TO_MOD.get(pior_item[2], 'N/A')}  nota={pior_item[0][1]:.1f}\n")
            else:
                if idx_prop[idx] < len(propostas[idx]):
                    fila.append(idx)

    total    = sum(len(v) for v in alocacoes.values())
    metricas = {
        'alocados':    total,
        'propostas':   n_props,
        'aloc_dir':    n_aloc_dir,
        'trocas':      n_trocas,
        'sem_vaga':    n_sem_vaga,
        'prop_p_cand': round(n_props / n, 2) if n else 0,
    }
    alocacoes_limpas = {k: [item[1] for item in v] for k, v in alocacoes.items()}
    return total, metricas, alocacoes_limpas
def alocacoes_para_df(alocacoes_limpas: dict, cands: CandidatosArray) -> pd.DataFrame:
    linhas = []
    for (id_curso, mod_vaga_int), cand_idxs in alocacoes_limpas.items():
        if not cand_idxs:
            continue
        mod_vaga_str = _INT_TO_MOD.get(mod_vaga_int, str(mod_vaga_int))
        for cand_idx in cand_idxs:
            op = 0 if (cands.tem_op[cand_idx, 0] and
                       int(cands.id_curso[cand_idx, 0]) == id_curso) else 1
            mod_insc_int = int(cands.modalidade[cand_idx, op])
            mod_insc_str = _INT_TO_MOD.get(mod_insc_int, str(mod_insc_int))
            notas        = cands.notas[cand_idx, op, :]
            vaga_rem     = mod_insc_str if mod_vaga_str != mod_insc_str else ''
            linhas.append({
                'ID_CANDIDATO':         cands.ids[cand_idx],
                'ID_CURSO':             id_curso,
                'MODALIDADE':           mod_vaga_str,
                'MODALIDADE_INSCRICAO': mod_insc_str,
                'NOTA_CANDIDATO':       notas[0],
                'NOTA_R':               notas[1],
                'NOTA_L':               notas[2],
                'NOTA_M':               notas[3],
                'NOTA_CN':              notas[4],
                'NOTA_CH':              notas[5],
                'VAGA_REMANEJADA_CR':   vaga_rem,
            })
    cols = ['ID_CANDIDATO', 'ID_CURSO', 'MODALIDADE', 'MODALIDADE_INSCRICAO',
            'NOTA_CANDIDATO', 'NOTA_R', 'NOTA_L', 'NOTA_M', 'NOTA_CN', 'NOTA_CH',
            'VAGA_REMANEJADA_CR']
    return pd.DataFrame(linhas, columns=cols)
def main():
    print("=" * 70)
    print("ALGORITMO LCF — Least Competitive First")
    print("=" * 70)

    print("\nPASSO 1: Carregando vagas...")
    t0 = time.time()
    vagas_dict  = carregar_vagas(CAMINHO_VAGAS)
    total_vagas = sum(vagas_dict.values())
    print(f"   Total de vagas : {total_vagas:,}")
    print(f"   Combinações    : {len(vagas_dict):,}")
    print(f"   {time.time() - t0:.2f}s")

    print("\nPASSO 2: Carregando candidatos...")
    t0 = time.time()
    cands   = carregar_candidatos(CAMINHO_CONCORRENCIA)
    n_cands = cands.n
    print(f"   Candidatos únicos: {n_cands:,}")
    print(f"   {time.time() - t0:.2f}s")

    print("\nPASSO 3: Construindo elegibilidade base...")
    t0 = time.time()
    elegibilidade_base = construir_elegibilidade_base(cands, vagas_dict)
    print(f"   {time.time() - t0:.2f}s")

    print("\nPASSO 4: Pré-calculando razões candidatos/vaga (critério LCF)...")
    t0 = time.time()
    razao_por_vaga = precalcular_razoes(vagas_dict, elegibilidade_base)
    print(f"   Combinações com elegíveis: {len(razao_por_vaga)}")
    print(f"   {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print("EXECUTANDO ALGORITMO LCF")
    t1 = time.time()
    propostas = build_all_propostas_lcf(cands, elegibilidade_base, razao_por_vaga)
    total, metricas, alocacoes = gale_shapley(
        cands, vagas_dict, propostas, compute_score_alg2, verbose=VERBOSE
    )
    t1 = time.time() - t1

    print(f"\n   Alocados : {total:,} ({total / total_vagas * 100:.3f}%)")
    print(f"   Tempo    : {t1:.3f}s")
    print(f"   Propostas: {metricas['propostas']:,}")
    print(f"   Trocas   : {metricas['trocas']:,}")

    print("\nExportando CSV...")
    df_aloc = alocacoes_para_df(alocacoes, cands)
    df_aloc.to_csv('alocados_LCF.csv', index=False, encoding='utf-8-sig')
    print(f"   alocados_LCF.csv: {len(df_aloc):,} linhas")


if __name__ == "__main__":
    main()