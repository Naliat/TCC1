using CSV
using DataFrames
using JuMP
using Gurobi
using Statistics
const VAGAS_FILE = "../df_vagas_2023.csv"
const CONCORRENCIA_FILE = "../df_concorrencia_2023.csv"

function load_dataframe(path::String)::DataFrame
    !isfile(path) && error("Arquivo não encontrado: $path")
    filesize(path) == 0 && return DataFrame()
    return DataFrame(CSV.File(path))
end

function normalize_course_name(value)::String
    s = lowercase(string(value))
    s = replace(s, "á" => "a", "à" => "a", "ã" => "a", "â" => "a")
    s = replace(s, "ç" => "c", "é" => "e", "ê" => "e", "í" => "i")
    s = replace(s, "ó" => "o", "ô" => "o", "ú" => "u", "ü" => "u")
    return s
end

function is_exact_ciencia_da_computacao_course(name)::Bool
    return normalize_course_name(name) == "ciencia da computacao"
end

function rss_mb()::Float64
    return Sys.maxrss() / (1024.0^2)
end

vagas = load_dataframe(VAGAS_FILE)
concorrencia = load_dataframe(CONCORRENCIA_FILE)

println("Linhas em $VAGAS_FILE: $(nrow(vagas))")
println("Linhas em $CONCORRENCIA_FILE: $(nrow(concorrencia))")

if nrow(vagas) == 0
    error("Nenhuma linha de vagas disponível em $VAGAS_FILE")
end

vagas = filter(row -> !ismissing(row.MODALIDADE) && !ismissing(row.QT_VAGAS_CONCORRENCIA) && is_exact_ciencia_da_computacao_course(row.NO_CURSO), vagas)

vagas[!, :capacity] = Int.(round.(coalesce.(vagas[!, :QT_VAGAS_CONCORRENCIA], 0)))

hospital_lookup = unique(vagas[:, [:ID_CURSO, :MODALIDADE, :ID_CURSO_U, :NO_CURSO, :capacity]], keep=:first)
sort!(hospital_lookup, [:ID_CURSO, :MODALIDADE])

hosp_nome = Dict{Int, String}()
q = Dict{Int, Int}()
for (idx, row) in enumerate(eachrow(hospital_lookup))
    hosp_nome[idx] = string(row.ID_CURSO_U)
    q[idx] = row.capacity
end

N_HOSP = nrow(hospital_lookup)
concorrencia = leftjoin(concorrencia, hospital_lookup[:, [:ID_CURSO, :MODALIDADE, :ID_CURSO_U]],
    on = [:ID_CURSO => :ID_CURSO, :MODALIDADE_INSCRICAO => :MODALIDADE])
concorrencia = filter(row -> !ismissing(row.ID_CURSO_U), concorrencia)

tie_count = 0
total_vagas = 0

if nrow(concorrencia) == 0
    @warn "df_concorrencia_2023.csv não gerou pares candidato-hospital válidos."
    cand_nome = Dict{Int, String}()
    P = Dict{Int, Vector{Int}}()
    N_CAND = 0
else
    cand_keys = sort(unique(concorrencia[!, :ID_CANDIDATO]))
    cand_nome = Dict{Int, String}()
    cand_id_map = Dict{eltype(cand_keys), Int}()
    for (idx, cand) in enumerate(cand_keys)
        cand_id_map[cand] = idx
        cand_nome[idx] = string(cand)
    end

    hospital_keys = unique(concorrencia[!, :ID_CURSO_U])
    hosp_id_map = Dict{eltype(hospital_keys), Int}()
    for (idx, hosp) in enumerate(hospital_keys)
        hosp_id_map[hosp] = idx
    end

    hosp_name_map = Dict{Int, String}()
    hosp_capacity_map = Dict{Int, Int}()
    for row in eachrow(hospital_lookup)
        hosp_key = row.ID_CURSO_U
        if haskey(hosp_id_map, hosp_key)
            idx = hosp_id_map[hosp_key]
            hosp_name_map[idx] = string(hosp_key)
            hosp_capacity_map[idx] = row.capacity
        end
    end

    hosp_nome = Dict{Int, String}()
    q = Dict{Int, Int}()
    for idx in 1:length(hosp_name_map)
        q[idx] = hosp_capacity_map[idx]
        hosp_nome[idx] = hosp_name_map[idx]
    end

    total_vagas = sum(values(q))
    N_HOSP = length(hosp_name_map)
    N_CAND = length(cand_keys)

    candidate_options = combine(groupby(concorrencia, [:ID_CANDIDATO, :ID_CURSO_U]),
        :CLASSIFICACAO_CM => minimum => :CLASSIFICACAO_CM,
        :NOTA_CANDIDATO => maximum => :NOTA_CANDIDATO)

    tie_groups = combine(groupby(candidate_options, [:ID_CURSO_U, :NOTA_CANDIDATO]), nrow => :tie_size)
    tie_count = Int(sum(max.(tie_groups[!, :tie_size] .- 1, 0)))

    tie_score_groups = filter(row -> row.tie_size > 1, tie_groups)
    if nrow(tie_score_groups) > 0
        println("Detalhes dos empates por nota (mesmo hospital):")
        for row in eachrow(tie_score_groups)
            tied_cands = filter(r -> r.ID_CURSO_U == row.ID_CURSO_U && r.NOTA_CANDIDATO == row.NOTA_CANDIDATO, candidate_options)
            candidate_ids = sort(tied_cands[!, :ID_CANDIDATO])
            println("  Hospital $(row.ID_CURSO_U), nota $(row.NOTA_CANDIDATO): candidatos $(candidate_ids) (total=$(row.tie_size))")
        end
    else
        println("Nenhum empate por nota detectado.")
    end

    indiff_groups = combine(groupby(candidate_options, [:ID_CANDIDATO, :CLASSIFICACAO_CM]), nrow => :indiff_size)
    indiff_groups = filter(row -> row.indiff_size > 1, indiff_groups)
    if nrow(indiff_groups) > 0
        println("Detalhes de indiferença de preferência entre modalidades (mesma classificação):")
        for row in eachrow(indiff_groups)
            tied_opts = filter(r -> r.ID_CANDIDATO == row.ID_CANDIDATO && r.CLASSIFICACAO_CM == row.CLASSIFICACAO_CM, candidate_options)
            option_ids = sort(tied_opts[!, :ID_CURSO_U])
            println("  Candidato $(row.ID_CANDIDATO), classificação $(row.CLASSIFICACAO_CM): modalidades $(option_ids) (total=$(row.indiff_size))")
        end
    else
        println("Nenhuma indiferença de modalidades detectada.")
    end
    P = Dict{Int, Vector{Int}}()
    rank_cand = Dict{Int, Dict{Int, Int}}()
    rank_hosp = Dict{Int, Dict{Int, Int}}()

    for cand in cand_keys
        cand_idx = cand_id_map[cand]
        cand_rows = filter(row -> row.ID_CANDIDATO == cand, candidate_options)
        sort!(cand_rows, :CLASSIFICACAO_CM)
        P[cand_idx] = Int[]
        rank_cand[cand_idx] = Dict{Int, Int}()
        for (pos, row) in enumerate(eachrow(cand_rows))
            hosp_idx = hosp_id_map[string(row.ID_CURSO_U)]
            push!(P[cand_idx], hosp_idx)
            rank_cand[cand_idx][hosp_idx] = pos
        end
    end

    for hosp in hospital_keys
        hosp_idx = hosp_id_map[hosp]
        hosp_rows = filter(row -> row.ID_CURSO_U == hosp, candidate_options)
        sort!(hosp_rows, :NOTA_CANDIDATO, rev=true)
        rank_hosp[hosp_idx] = Dict{Int, Int}()
        for (pos, row) in enumerate(eachrow(hosp_rows))
            cand_idx = cand_id_map[row.ID_CANDIDATO]
            rank_hosp[hosp_idx][cand_idx] = pos
        end
    end
end
println("Instância de computação selecionada: $N_HOSP cursos, $N_CAND candidatos analisados")
println("Total de vagas ofertadas na instância: $total_vagas")
println("Empates detectados nas listas de nota: $tie_count")
println("Primeiros hospitais:")
for (idx, name) in first(hosp_nome, min(10, N_HOSP))
    println("  $idx -> $name (capacidade=$(q[idx]))")
end

const REMANEJAMENTO_ORDER = Dict{String, Vector{String}}(
    "LB_PPI" => ["LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LB_PP"  => ["LB_PPI", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LB_I"   => ["LB_PPI", "LB_PP", "LB_Q", "LB_PCD", "LB_EP", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LB_Q"   => ["LB_PPI", "LB_PP", "LB_I", "LB_PCD", "LB_EP", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LB_PCD" => ["LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_EP", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LB_EP"  => ["LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "AC"],
    "LI_PPI" => ["LI_PP", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
    "LI_PP"  => ["LI_PPI", "LI_I", "LI_Q", "LI_PCD", "LI_EP", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
    "LI_I"   => ["LI_PPI", "LI_PP", "LI_Q", "LI_PCD", "LI_EP", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
    "LI_Q"   => ["LI_PPI", "LI_PP", "LI_I", "LI_PCD", "LI_EP", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
    "LI_PCD" => ["LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_EP", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
    "LI_EP"  => ["LI_PPI", "LI_PP", "LI_I", "LI_Q", "LI_PCD", "LB_PPI", "LB_PP", "LB_I", "LB_Q", "LB_PCD", "LB_EP", "AC"],
)

hosp_modalidade = Dict{Int, String}()
curso_hospitais = Dict{Any, Vector{Int}}()
curso_mod_to_hosp = Dict{Tuple{Any, String}, Int}()

for (idx, row) in enumerate(eachrow(hospital_lookup))
    hosp_key = row.ID_CURSO_U
    if isdefined(Main, :hosp_id_map) && haskey(hosp_id_map, hosp_key)
        hidx = hosp_id_map[hosp_key]
        hosp_modalidade[hidx] = string(row.MODALIDADE)

        id_curso = row.ID_CURSO
        if !haskey(curso_hospitais, id_curso)
            curso_hospitais[id_curso] = Int[]
        end
        push!(curso_hospitais[id_curso], hidx)
        curso_mod_to_hosp[(id_curso, string(row.MODALIDADE))] = hidx
    end
end

println()
println("─"^55)
println("  MAPEAMENTO DE MODALIDADES PARA REMANEJAMENTO")
println("─"^55)
modalidades_com_regra = 0
modalidades_sem_regra = 0
for (hidx, mod) in sort(collect(hosp_modalidade), by=x->x[1])
    tem_regra = haskey(REMANEJAMENTO_ORDER, mod) ? "✓" : "✗"
    if haskey(REMANEJAMENTO_ORDER, mod)
        global modalidades_com_regra += 1
    else
        global modalidades_sem_regra += 1
    end
    println("Hospital $hidx ($(hosp_nome[hidx])): modalidade=$mod remanejamento=$tem_regra")
end
println("Hospitais com regra de remanejamento: $modalidades_com_regra")
println("Hospitais sem regra (mantêm vagas): $modalidades_sem_regra")
println("Cursos distintos para remanejamento: $(length(curso_hospitais))")
println()
function compute_S(i::Int, j::Int)::Vector{Int}
    [h for h in P[i] if rank_cand[i][h] <= rank_cand[i][j]]
end

function compute_T(i::Int, j::Int)::Vector{Int}
    [r for r in 1:N_CAND
        if j in P[r] && haskey(rank_hosp[j], r) && rank_hosp[j][r] <= rank_hosp[j][i]]
end

function remanejar_vagas!(q, alocacoes_por_hosp, hosp_modalidade,
                          curso_hospitais, curso_mod_to_hosp, REMANEJAMENTO_ORDER)
    total_remanejado = 0
    log_remanejamento = NamedTuple{(:curso, :de_mod, :para_mod, :vagas), Tuple{Any, String, String, Int}}[]

    for (id_curso, hosps) in curso_hospitais
        for h_fonte in hosps
            mod_fonte = hosp_modalidade[h_fonte]
            !haskey(REMANEJAMENTO_ORDER, mod_fonte) && continue
            alocados = get(alocacoes_por_hosp, h_fonte, 0)
            vagas_ociosas = q[h_fonte] - alocados
            vagas_ociosas <= 0 && continue
            for mod_destino in REMANEJAMENTO_ORDER[mod_fonte]
                vagas_ociosas <= 0 && break
                chave = (id_curso, mod_destino)
                !haskey(curso_mod_to_hosp, chave) && continue

                h_destino = curso_mod_to_hosp[chave]
                q[h_destino] += vagas_ociosas
                q[h_fonte] -= vagas_ociosas

                push!(log_remanejamento, (
                    curso=id_curso,
                    de_mod=mod_fonte,
                    para_mod=mod_destino,
                    vagas=vagas_ociosas
                ))

                total_remanejado += vagas_ociosas
                vagas_ociosas = 0
            end
        end
    end

    return total_remanejado, log_remanejamento
end

if N_CAND == 0
    println("Nenhum candidato carregado a partir de df_concorrencia_2023.csv.")
    println("O modelo IP será montado apenas quando o arquivo de concorrência contiver dados reais.")
else

function run_hrt_remanejamento()
    q_original = Dict{Int, Int}(k => v for (k, v) in q)
    max_iter = 20
    iteracao = 0
    total_remanejado_global = 0
    log_remanejamento_global = NamedTuple{(:curso, :de_mod, :para_mod, :vagas), Tuple{Any, String, String, Int}}[]
    solver_seconds_total = 0.0
    memory_samples = Float64[]
    push!(memory_samples, rss_mb())
    last_status = nothing
    last_n_aloc = 0
    last_model = nothing

    println("="^55)
    println("  MAX-HRT + REMANEJAMENTO ITERATIVO")
    println("="^55)
    println()

    while iteracao < max_iter
        iteracao += 1
        println("─"^55)
        println("  ITERAÇÃO $iteracao")
        println("─"^55)
        println()
        model = Model(Gurobi.Optimizer)
        push!(memory_samples, rss_mb())
        set_silent(model)
        @variable(model, x[i in 1:N_CAND, j in P[i]], Bin)
        push!(memory_samples, rss_mb())
        @objective(model, Max,
            sum(x[i,j] for i in 1:N_CAND for j in P[i])
        )
        for i in 1:N_CAND
            @constraint(model, sum(x[i,j] for j in P[i]) <= 1)
        end
        for j in 1:N_HOSP
            cands_j = [i for i in 1:N_CAND if j in P[i]]
            isempty(cands_j) && continue
            @constraint(model, sum(x[i,j] for i in cands_j) <= q[j])
        end
        for i in 1:N_CAND
            for j in P[i]
                Sij = compute_S(i, j)
                Tij = compute_T(i, j)

                expr_S = isempty(Sij) ? 0.0 : sum(x[i,h] for h in Sij)
                expr_T = isempty(Tij) ? 0.0 : sum(x[r,j] for r in Tij)

                @constraint(model, q[j] * (1 - expr_S) - expr_T <= 0)
            end
        end

        push!(memory_samples, rss_mb())

    println("Candidatos considerados: $N_CAND")
    println("Hospitais considerados: $N_HOSP")
    println("Total de vagas (capacidade atual): $(sum(values(q)))")
    println()
        
    solver_start = time_ns()
    optimize!(model)
    solver_finish = time_ns()
        
    push!(memory_samples, rss_mb())
        
    status = termination_status(model)
    solver_seconds = (solver_finish - solver_start) / 1e9
    solver_seconds_total += solver_seconds
    num_vars = num_variables(model)
    constraint_count = JuMP.num_constraints(model; count_variable_in_set_constraints=true)
        
    println("Status do solver : $status")
    println("Tempo do solver (s): $(round(solver_seconds, digits=6))")
    println("Variáveis: $num_vars | Restrições: $constraint_count")
        
    if !(status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT)
        println("Nenhuma solução encontrada. Interrompendo remanejamento.")
        last_status = status
        last_model = model
        break
    end
    obj_val = objective_value(model)
    best_bound = objective_bound(model)
    
    gap_abs = abs(best_bound - obj_val)
    gap_rel = obj_val == 0 ? 0.0 : 100.0 * gap_abs / obj_val
    
    println("Valor da solução inteira (incumbent): $obj_val")
    println("Upper bound (best bound do Gurobi): $best_bound")
    println("Gap absoluto: $(round(gap_abs, digits=4))")
    println("Gap relativo (%): $(round(gap_rel, digits=4))")
    
    n_aloc = round(Int, obj_val)
    println("Alocações nesta iteração: $n_aloc")
    
    last_status = status
    last_n_aloc = n_aloc
    last_model = model
        alocacoes_por_hosp = Dict{Int, Int}()
        for j in 1:N_HOSP
            count = 0
            for i in 1:N_CAND
                if j in P[i] && value(x[i,j]) > 0.5
                    count += 1
                end
            end
            alocacoes_por_hosp[j] = count
        end
        println()
        println("Ocupação por hospital:")
        for j in 1:N_HOSP
            aloc_j = get(alocacoes_por_hosp, j, 0)
            mod_j = get(hosp_modalidade, j, "?")
            ociosas = q[j] - aloc_j
            status_str = ociosas > 0 ? "  [$(ociosas) ociosa(s)]" : ""
            println("  $j ($(hosp_nome[j]), mod=$mod_j): $aloc_j/$(q[j])$status_str")
        end
        n_remanejado, log_iter = remanejar_vagas!(
            q, alocacoes_por_hosp, hosp_modalidade,
            curso_hospitais, curso_mod_to_hosp, REMANEJAMENTO_ORDER
        )

        if n_remanejado == 0
            println()
            println("Nenhuma vaga ociosa para remanejar. Convergência atingida na iteração $iteracao.")
            break
        end

        total_remanejado_global += n_remanejado
        append!(log_remanejamento_global, log_iter)

        println()
        println("Remanejamento na iteração $iteracao:")
        for entry in log_iter
            println("  Curso $(entry.curso): $(entry.de_mod) → $(entry.para_mod) ($(entry.vagas) vaga(s))")
        end
        println("Total remanejado nesta iteração: $n_remanejado")
        println()
    end

    if iteracao >= max_iter
        println(" Limite de $max_iter iterações atingido.")
    end

    println()
    println("="^55)
    println("  RESULTADOS FINAIS — MAX-HRT + REMANEJAMENTO")
    println("="^55)
    println()

    peak_memory_mb = maximum(memory_samples)
    println("Iterações de remanejamento: $iteracao")
    println("Total de vagas remanejadas: $total_remanejado_global")
    println("Tempo total de resolução (s): $(round(solver_seconds_total, digits=6))")
    println("Pico de memória RAM (MiB): $(round(peak_memory_mb, digits=2))")
    println()
    if total_remanejado_global > 0
        println("─"^55)
        println("  DETALHES DO REMANEJAMENTO")
        println("─"^55)
        for entry in log_remanejamento_global
            println("  Curso $(entry.curso): $(entry.de_mod) → $(entry.para_mod) ($(entry.vagas) vaga(s))")
        end
        println()
        println("─"^55)
        println("  COMPARAÇÃO DE CAPACIDADES (ORIGINAL → FINAL)")
        println("─"^55)
        for j in 1:N_HOSP
            orig = get(q_original, j, 0)
            final_q = q[j]
            mod_j = get(hosp_modalidade, j, "?")
            diff_str = final_q != orig ? " ($(final_q > orig ? "+" : "")$(final_q - orig))" : ""
            println("  $j ($(hosp_nome[j]), $mod_j): $orig → $final_q$diff_str")
        end
        println()
        println("Total original: $(sum(values(q_original))) | Total final: $(sum(values(q)))")
        println()
    end
    if last_status == MOI.OPTIMAL || last_status == MOI.FEASIBLE_POINT
        model = last_model
        x = model[:x]
        n_aloc = last_n_aloc
        println("Tamanho do emparelhamento ótimo final: $n_aloc")
        println()

        display_limit = min(20, n_aloc)
        println("Alocações encontradas (amostra de $display_limit de $n_aloc):")
        println("-"^35)
        printed_allocations = 0
        for i in 1:N_CAND
            for j in P[i]
                if value(x[i,j]) > 0.5
                    if printed_allocations < display_limit
                        mod_j = get(hosp_modalidade, j, "?")
                        println("  Candidato $(cand_nome[i])  →  Hospital $(hosp_nome[j]) [$mod_j]")
                        printed_allocations += 1
                    end
                end
            end
        end
        println("Total de alocações registradas pelo modelo: $n_aloc")
        println()
    else
        println("Nenhuma solução ótima encontrada. Status: $last_status")
    end
end  

run_hrt_remanejamento()

end