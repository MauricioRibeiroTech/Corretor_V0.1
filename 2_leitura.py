import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, pearsonr
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Configurações da Página ---
st.set_page_config(
    page_title="Corretor e Analisador CESB",
    page_icon="📚",
    layout="wide"
)

# --- PARTE 1: LÓGICA E FUNÇÕES DE ANÁLISE TRI (Copiado do seu main_2.py) ---

class TRI_Simulator:
    def __init__(self):
        self.ability_range = np.linspace(-4, 4, 100)

    def probability_2pl(self, theta, a, b):
        """Função de resposta do item - modelo 2PL"""
        return 1 / (1 + np.exp(-a * (theta - b)))

    def fit_model(self, response_matrix):
        """Ajusta parâmetros TRI de forma simplificada"""
        n_students, n_items = response_matrix.shape
        p_values = response_matrix.mean(axis=0)
        p_values = np.clip(p_values, 0.001, 0.999)
        # Dificuldade (parâmetro b)
        difficulty = norm.ppf(1 - p_values) # Mapeamento mais comum
        
        discrimination = []
        total_scores = response_matrix.sum(axis=1)
        
        # Discriminação (parâmetro a)
        for i in range(n_items):
            try:
                # Correlação bisserial pontual como proxy para discriminação
                corr = pearsonr(response_matrix[:, i], total_scores)[0]
                # A normalização 1.7*corr é um ajuste comum
                discrimination.append(1.7 * corr if not np.isnan(corr) else 0.5)
            except:
                discrimination.append(0.5)
        
        # Habilidade (theta)
        student_p = (response_matrix.sum(axis=1) + 0.5) / (n_items + 1) # Adiciona suavização
        student_p = np.clip(student_p, 0.001, 0.999)
        ability = norm.ppf(student_p) # Mapeia percentil para score-z
        
        return {
            'difficulty': difficulty,
            'discrimination': np.array(discrimination),
            'ability': ability,
            'n_items': n_items,
            'n_students': n_students
        }

@st.cache_data
def run_advanced_tri_analysis(df):
    """Executa a análise TRI completa a partir do DataFrame de respostas."""
    if df.columns[0].lower() in ['aluno', 'user', 'nome', 'student']:
        student_names = df[df.columns[0]]
        df = df.set_index(df.columns[0])
    else:
        student_names = pd.Series([f'Aluno_{i + 1}' for i in range(len(df))])
    
    # Converte para numérico, tratando erros
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    response_matrix = df.to_numpy()
    
    simulator = TRI_Simulator()
    model_params = simulator.fit_model(response_matrix)
    
    # --- Resultados dos Alunos ---
    student_results = pd.DataFrame({
        'Aluno': student_names,
        'Proficiência (θ)': model_params['ability'],
        'Pontuação Total': response_matrix.sum(axis=1),
        'Percentual de Acerto': (response_matrix.sum(axis=1) / model_params['n_items'] * 100).round(2),
        'Z-Score': (model_params['ability'] - model_params['ability'].mean()) / model_params['ability'].std()
    })
    
    # --- Resultados dos Itens ---
    item_results = pd.DataFrame({
        'Questão': df.columns,
        'Dificuldade (b)': model_params['difficulty'],
        'Discriminação (a)': model_params['discrimination'],
        '% Acerto': (response_matrix.mean(axis=0) * 100).round(2),
        'Índice de Discriminação': [
            (response_matrix[model_params['ability'] > np.median(model_params['ability']), i].mean() -
             response_matrix[model_params['ability'] <= np.median(model_params['ability']), i].mean())
            for i in range(model_params['n_items'])
        ]
    })
    
    corr_bisserial = []
    for i in range(model_params['n_items']):
        try:
            corr = np.corrcoef(response_matrix[:, i], model_params['ability'])[0, 1]
            corr_bisserial.append(corr if not np.isnan(corr) else 0)
        except:
            corr_bisserial.append(0)
    item_results['Correlação Bisserial'] = corr_bisserial
    
    # --- Dados da Curva Característica do Item (CCI) ---
    cci_data = []
    for i, item in enumerate(df.columns):
        for theta in simulator.ability_range:
            prob = simulator.probability_2pl(theta,
                                             model_params['discrimination'][i],
                                             model_params['difficulty'][i])
            cci_data.append({
                'Questão': item, 'Theta': theta, 'Probabilidade': prob,
                'Dificuldade': model_params['difficulty'][i], 'Discriminação': model_params['discrimination'][i]
            })
    cci_df = pd.DataFrame(cci_data)
    
    return student_results, item_results, cci_df, model_params

# --- Funções de Gráfico (Copiadas do seu main_2.py) ---
def plot_cci(cci_df, selected_items):
    fig = go.Figure()
    for item in selected_items:
        item_data = cci_df[cci_df['Questão'] == item]
        fig.add_trace(go.Scatter(
            x=item_data['Theta'], y=item_data['Probabilidade'], mode='lines',
            name=f'{item} (b={item_data["Dificuldade"].iloc[0]:.2f})',
            hovertemplate=f'Questão: {item}<br>θ: %{{x:.2f}}<br>P(θ): %{{y:.3f}}<extra></extra>'
        ))
    fig.update_layout(title='Curvas Características dos Itens (CCI)', xaxis_title='Habilidade (θ)', yaxis_title='Probabilidade de Acerto P(θ)', hovermode='closest', height=500)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="P(θ) = 0.5")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig

def plot_information_curve(cci_df, selected_items):
    fig = go.Figure()
    for item in selected_items:
        item_data = cci_df[cci_df['Questão'] == item]
        p = item_data['Probabilidade']; a = item_data['Discriminação'].iloc[0]
        information = a ** 2 * p * (1 - p)
        fig.add_trace(go.Scatter(x=item_data['Theta'], y=information, mode='lines', name=f'{item}', hovertemplate=f'Questão: {item}<br>θ: %{{x:.2f}}<br>Info: %{{y:.3f}}<extra></extra>'))
    fig.update_layout(title='Função de Informação dos Itens', xaxis_title='Habilidade (θ)', yaxis_title='Informação de Fisher', hovermode='closest', height=500)
    return fig

def plot_theta_distribution(student_results):
    fig = make_subplots(rows=1, cols=1)
    hist_fig = px.histogram(student_results, x='Proficiência (θ)', nbins=20, opacity=0.7)
    fig.add_trace(hist_fig.data[0])
    x_norm = np.linspace(student_results['Proficiência (θ)'].min(), student_results['Proficiência (θ)'].max(), 100)
    y_norm = norm.pdf(x_norm, student_results['Proficiência (θ)'].mean(), student_results['Proficiência (θ)'].std())
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm * len(student_results) * (x_norm[1] - x_norm[0]), mode='lines', name='Distribuição Normal', line=dict(color='red', width=2)))
    fig.update_layout(title='Distribuição das Habilidades (Proficiência) dos Alunos', xaxis_title='Habilidade (θ)', yaxis_title='Frequência', height=400, showlegend=False)
    return fig

def plot_difficulty_discrimination_scatter(item_results):
    fig = px.scatter(item_results, x='Dificuldade (b)', y='Discriminação (a)', size='% Acerto', color='Correlação Bisserial', hover_name='Questão', title='Análise Multidimensional dos Itens', labels={'Dificuldade (b)': 'Dificuldade (b)', 'Discriminação (a)': 'Discriminação (a)', '% Acerto': '% de Acerto', 'Correlação Bisserial': 'Corr. Bisserial'})
    return fig

def plot_difficulty_discrimination_heatmap(item_results):
    item_results['Dificuldade_Cat'] = pd.cut(item_results['Dificuldade (b)'], bins=[-np.inf, -1, 0, 1, np.inf], labels=['Muito Fácil', 'Fácil', 'Difícil', 'Muito Difícil'])
    item_results['Discriminação_Cat'] = pd.cut(item_results['Discriminação (a)'], bins=[-np.inf, 0.3, 0.6, 1, np.inf], labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
    heatmap_data = pd.crosstab(item_results['Dificuldade_Cat'], item_results['Discriminação_Cat'])
    fig = px.imshow(heatmap_data, title='Mapa de Calor: Dificuldade vs Discriminação', color_continuous_scale='Viridis', aspect='auto')
    fig.update_layout(xaxis_title='Poder de Discriminação', yaxis_title='Nível de Dificuldade')
    return fig

def calculate_reliability(item_results):
    try:
        avg_correlation = item_results['Correlação Bisserial'].mean()
        n_items = len(item_results)
        alpha = (n_items * avg_correlation) / (1 + (n_items - 1) * avg_correlation)
        return max(0, min(1, alpha))
    except:
        return 0.7


# --- PARTE 2: FUNÇÃO DE RELATÓRIO HTML (MODIFICADA CONFORME SOLICITADO) ---

def generate_html_report(student_results, item_results):
    """
    Gera um relatório HTML completo com explicações, gráficos e tabelas.
    MODIFICADO: Inclui tabela completa com Nota de Acertos e Nota TRI.
    """
    # --- Estilo CSS para o Relatório ---
    html_style = """
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f9f9f9; padding: 20px; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; }
        h1 { font-size: 2.2em; } h2 { font-size: 1.8em; margin-top: 40px; } h3 { font-size: 1.4em; margin-top: 30px; border-bottom: 1px solid #ccc; }
        p, li { font-size: 1.1em; }
        .explanation { background-color: #e7f3ff; border-left: 5px solid #0056b3; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .glossary { background-color: #f5f5f5; border: 1px solid #ddd; padding: 15px; margin-top: 20px; border-radius: 5px; }
        .glossary dt { font-weight: bold; color: #333; }
        .glossary dd { margin-left: 20px; margin-bottom: 10px; color: #555; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #0056b3; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric-box { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }
        .metric { background: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; }
        .metric .label { font-size: 1.1em; color: #555; }
        .metric .value { font-size: 2em; font-weight: bold; color: #0056b3; }
        .warning { color: #d9534f; font-weight: bold; }
        .success { color: #5cb85c; font-weight: bold; }
    </style>
    """

    # --- Gráficos e Tabelas ---
    theta_dist_html = plot_theta_distribution(student_results).to_html(full_html=False, include_plotlyjs='cdn')
    item_scatter_html = plot_difficulty_discrimination_scatter(item_results).to_html(full_html=False, include_plotlyjs='cdn')
    
    # --- MODIFICAÇÃO: Tabela completa de Alunos (Acertos vs TRI) ---
    all_students_df = student_results[['Aluno', 'Percentual de Acerto', 'Proficiência (θ)']].copy()
    all_students_df.columns = ['Aluno', 'Nota Acertos (%)', 'Nota TRI (θ)']
    all_students_df['Nota TRI (θ)'] = all_students_df['Nota TRI (θ)'].round(3)
    all_students_html = all_students_df.sort_values('Nota TRI (θ)', ascending=False).to_html(index=False, justify='center')
    
    item_params_html = item_results.round(3).to_html(index=False, justify='center')

    # --- Lógica para Recomendações ---
    problematic_items = item_results[(item_results['Discriminação (a)'] < 0.3) | (item_results['% Acerto'] < 20) | (item_results['% Acerto'] > 90)]
    recommendations_html = ""
    if len(problematic_items) > 0:
        recommendations_html += f"<p class='warning'>Atenção: {len(problematic_items)} itens foram sinalizados e podem precisar de revisão.</p><ul>"
        for _, item in problematic_items.iterrows():
            issues = []
            if item['Discriminação (a)'] < 0.3: issues.append("baixa discriminação (não distingue bem os alunos)")
            if item['% Acerto'] < 20: issues.append("muito difícil (poucos acertaram)")
            if item['% Acerto'] > 90: issues.append("muito fácil (quase todos acertaram)")
            recommendations_html += f"<li><b>{item['Questão']}</b>: {', '.join(issues)}.</li>"
        recommendations_html += "</ul>"
    else:
        recommendations_html = "<p class='success'>Ótima notícia! Todos os itens apresentam características psicométricas adequadas.</p>"

    # --- Montagem do Corpo do HTML ---
    html_body = f"""
    <div class="container">
        <h1>📊 Relatório de Análise TRI</h1>
        <p>Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>

        <h2>Resumo Geral do Teste</h2>
        <div class="explanation">
            <p>Esta seção apresenta as métricas mais importantes que resumem a qualidade geral do teste e o desempenho dos participantes.</p>
        </div>
        <div class="metric-box">
            <div class="metric"><div class="label">Alunos Analisados</div><div class="value">{len(student_results)}</div></div>
            <div class="metric"><div class="label">Itens Analisados</div><div class="value">{len(item_results)}</div></div>
            <div class="metric"><div class="label">Proficiência Média (θ)</div><div class="value">{student_results['Proficiência (θ)'].mean():.3f}</div></div>
            <div class="metric"><div class="label">Confiabilidade (Alpha)</div><div class="value">{calculate_reliability(item_results):.3f}</div></div>
        </div>

        <h2>🎓 Análise da Proficiência dos Alunos</h2>
        <div class="explanation">
            <p>A <strong>proficiência</strong> é a estimativa do conhecimento de cada aluno. O gráfico abaixo mostra como as proficiências se distribuem. Uma curva em formato de "sino" (distribuição normal) é geralmente esperada.</p>
        </div>
        {theta_dist_html}
        
        <h3>Entendendo os Índices dos Alunos</h3>
        <div class="glossary">
            <dl>
                <dt>Nota Acertos (%)</dt>
                <dd>A contagem simples de acertos (ex: 80%). É a nota tradicional.</dd>
                <dt>Nota TRI (Proficiência θ)</dt>
                <dd>É a "nota" real do aluno segundo a TRI. Diferente da nota tradicional, ela considera <strong>quais</strong> questões foram acertadas (fáceis ou difíceis). Um valor maior indica maior conhecimento.</dd>
            </dl>
        </div>
        
        <h3>Tabela de Desempenho (Acertos vs TRI)</h3>
        {all_students_html}
        <h2>📝 Análise Psicométrica dos Itens</h2>
        <div class="explanation">
            <p>Aqui, avaliamos a qualidade de cada questão do teste. O gráfico de dispersão ajuda a visualizar rapidamente os itens mais e menos eficazes.</p>
        </div>
        {item_scatter_html}

        <h3>Entendendo os Parâmetros dos Itens</h3>
        <div class="glossary">
            <dl>
                <dt>Dificuldade (b)</dt>
                <dd>Mede o quão difícil é a questão. Valores positivos indicam questões difíceis; valores negativos, fáceis.</dd>
                <dt>Discriminação (a)</dt>
                <dd>Mede a capacidade da questão de diferenciar alunos com alta e baixa proficiência. Valores desejáveis são geralmente acima de 0.8.</dd>
                <dt>Correlação Bisserial</dt>
                <dd>Mede a consistência interna. Alunos com notas altas no geral devem acertar esta questão. Valores negativos são um alerta (pode haver erro no gabarito).</dd>
            </dl>
        </div>
        
        <h3>Tabela Completa de Parâmetros dos Itens</h3>
        {item_params_html}

        <h2>🔍 Diagnóstico do Teste e Recomendações</h2>
        <div class="explanation">
            <p>Com base nos dados, esta seção fornece um diagnóstico final sobre a qualidade do teste e aponta itens específicos que podem ser melhorados.</p>
        </div>
        {recommendations_html}

    </div>
    """
    full_html = f"<!DOCTYPE html><html><head><title>Relatório de Análise TRI</title>{html_style}</head><body>{html_body}</body></html>"
    return full_html


# --- PARTE 3: INTERFACE STREAMLIT (Lançamento Manual + Análise) ---

st.title("Corretor de Gabaritos e Análise TRI 📚")

# --- Inicialização do Banco de Dados na Sessão ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# --- Configuração da Prova na Sidebar ---
st.sidebar.header("⚙️ 1. Configurar Prova")
gabarito_correto_str = st.sidebar.text_input(
    "Gabarito Correto (ex: ABDEC...)", 
    "EBDEABDCABEEBADEABDCABEDCD" # Gabarito de 26 questões (preto)
).upper()

if not gabarito_correto_str:
    st.sidebar.error("Por favor, insira um gabarito correto para continuar.")
    st.stop()

# --- Deriva N° de Questões e Opções do Gabarito ---
n_items = len(gabarito_correto_str)
options_list = sorted(list(set(gabarito_correto_str)))
options_with_blank = options_list + ["BRANCO"]

st.sidebar.info(f"Prova configurada para **{n_items} questões** com **{len(options_list)} opções** ({', '.join(options_list)}).")


# --- Definir Abas ---
tab_entry, tab_analysis = st.tabs([" Lançamento de Respostas ", " Análise TRI e Relatório "])


# --- ABA 1: Lançamento de Respostas ---
with tab_entry:
    st.header("✏️ Lançamento Manual de Respostas")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Novo Lançamento")
        id_aluno = st.text_input("Nome/ID do Aluno:")
        
        # O formulário agrupa todos os botões de rádio
        with st.form("entry_form"):
            student_answers = {} # Dicionário para guardar as respostas
            
            # Cria os botões de rádio para cada questão
            for i in range(n_items):
                q_num = i + 1
                correct_answer = gabarito_correto_str[i]
                
                # key é essencial para o Streamlit resetar o botão
                student_answers[f'Q{q_num}'] = st.radio(
                    f"**Questão {q_num}** (Gabarito: {correct_answer})",
                    options_with_blank,
                    horizontal=True,
                    key=f"q_{id_aluno}_{q_num}" # Chave única para resetar
                )
            
            submitted = st.form_submit_button("Salvar Respostas deste Aluno")
        
        if submitted:
            if not id_aluno:
                st.error("Por favor, insira um Nome/ID do Aluno antes de salvar.")
            elif id_aluno in st.session_state.results_df.get('Aluno', []):
                 st.error(f"Erro: O aluno '{id_aluno}' já existe na matriz. Use um nome diferente.")
            else:
                # --- Lógica para salvar ---
                binary_responses = {}
                for q_key, answer in student_answers.items():
                    q_idx = int(q_key[1:]) - 1 # Pega o índice (Q1 -> 0)
                    if answer == gabarito_correto_str[q_idx]:
                        binary_responses[q_key] = 1 # Acerto
                    else:
                        binary_responses[q_key] = 0 # Erro
                
                # Cria a nova linha
                new_row_data = {"Aluno": id_aluno, **binary_responses}
                new_row_df = pd.DataFrame([new_row_data])
                
                # Adiciona à matriz na sessão
                st.session_state.results_df = pd.concat(
                    [st.session_state.results_df, new_row_df], 
                    ignore_index=True
                )
                st.success(f"Respostas de '{id_aluno}' salvas com sucesso!")
                st.balloons()

    # --- Coluna da Direita (Matriz de Resultados) ---
    with col2:
        st.subheader("📋 Matriz de Respostas (Banco de Dados)")
        if st.session_state.results_df.empty:
            st.info("Nenhum aluno foi lançado ainda. Salve o primeiro aluno à esquerda.")
        else:
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            # --- Botão de Download CSV ---
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(st.session_state.results_df)
            st.download_button(
                label="📥 Baixar Matriz de Respostas como CSV",
                data=csv_data,
                file_name="matriz_respostas_tri.csv",
                mime="text/csv",
                use_container_width=True
            )


# --- ABA 2: Análise TRI e Relatório ---
with tab_analysis:
    st.header("📊 Análise Psicométrica (TRI)")

    # Verifica se há dados suficientes para a análise
    if len(st.session_state.results_df) < 5: # TRI precisa de variância
        st.warning(f"É necessário lançar pelo menos 5 alunos para uma análise TRI minimamente estável. (Atualmente: {len(st.session_state.results_df)})")
    else:
        st.info(f"Pronto para analisar **{len(st.session_state.results_df)} alunos** e **{n_items} questões**.")
        
        if st.button("🚀 Executar Análise TRI Avançada", type="primary", use_container_width=True):
            with st.spinner('Realizando análise psicométrica...'):
                try:
                    # Roda a análise com os dados da sessão
                    df_to_analyze = st.session_state.results_df.copy()
                    
                    student_results, item_results, cci_df, model_params = run_advanced_tri_analysis(df_to_analyze)
                    
                    # Salva os resultados da análise na sessão
                    st.session_state['tri_student_results'] = student_results
                    st.session_state['tri_item_results'] = item_results
                    st.session_state['tri_cci_df'] = cci_df
                    
                    st.success('✅ Análise TRI concluída com sucesso!')
                except Exception as e:
                    st.error(f"❌ Erro durante a análise TRI: {e}")
                    st.exception(e)

    # --- Se a análise foi rodada, exibe os resultados ---
    if 'tri_student_results' in st.session_state:
        # Puxa os resultados da sessão para exibição
        student_results = st.session_state['tri_student_results']
        item_results = st.session_state['tri_item_results']
        cci_df = st.session_state['tri_cci_df']
        
        st.markdown("---")
        
        # Cria as abas de resultados (copiado de main_2.py)
        tab_alunos, tab_itens, tab_curvas, tab_diag, tab_report = st.tabs([
            "🎓 Análise dos Alunos", 
            "📝 Análise dos Itens", 
            "📈 Curvas TRI", 
            "🔍 Diagnóstico", 
            "📋 Relatório HTML"
        ])
        
        with tab_alunos:
            st.subheader("Análise da Proficiência dos Alunos")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_theta_distribution(student_results), use_container_width=True)
                st.dataframe(student_results.sort_values('Proficiência (θ)', ascending=False).head(10).reset_index(drop=True), use_container_width=True)
            with col2:
                fig = px.scatter(student_results, x='Pontuação Total', y='Proficiência (θ)', hover_data=['Aluno', 'Percentual de Acerto'], title='Relação: Proficiência vs Pontuação Total', trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("📊 Estatísticas Descritivas")
                col_stats = st.columns(4)
                with col_stats[0]: st.metric("Média θ", f"{student_results['Proficiência (θ)'].mean():.3f}")
                with col_stats[1]: st.metric("Desvio Padrão", f"{student_results['Proficiência (θ)'].std():.3f}")
                with col_stats[2]: st.metric("Mínimo", f"{student_results['Proficiência (θ)'].min():.3f}")
                with col_stats[3]: st.metric("Máximo", f"{student_results['Proficiência (θ)'].max():.3f}")
        
        with tab_itens:
            st.subheader("Análise Psicométrica dos Itens")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_difficulty_discrimination_scatter(item_results), use_container_width=True)
            with col2:
                st.plotly_chart(plot_difficulty_discrimination_heatmap(item_results), use_container_width=True)
            st.subheader("📋 Tabela de Parâmetros dos Itens")
            st.dataframe(item_results.sort_values('Dificuldade (b)').round(3), use_container_width=True)
        
        with tab_curvas:
            st.subheader("Curvas Características e Informação")
            col1, col2 = st.columns(2)
            with col1:
                default_selection = item_results['Questão'].head(3).tolist()
                selected_items_cci = st.multiselect("Selecione as questões para análise das CCIs:", options=item_results['Questão'].tolist(), default=default_selection)
            with col2:
                st.info("💡 **Interpretação das CCIs:**")
                st.markdown("- **Curva à esquerda**: Questão mais fácil\n- **Curva à direita**: Questão mais difícil\n- **Curva mais íngreme**: Melhor discriminação")
            if selected_items_cci:
                col_cci1, col_cci2 = st.columns(2)
                with col_cci1: st.plotly_chart(plot_cci(cci_df, selected_items_cci), use_container_width=True)
                with col_cci2: st.plotly_chart(plot_information_curve(cci_df, selected_items_cci), use_container_width=True)
        
        with tab_diag:
            st.subheader("Diagnóstico do Teste")
            col1, col2 = st.columns(2)
            with col1:
                reliability = calculate_reliability(item_results)
                st.metric("Confiabilidade (Alpha de Cronbach)", f"{reliability:.3f}")
                avg_discrimination = item_results['Índice de Discriminação'].mean()
                st.metric("Discriminação Média", f"{avg_discrimination:.3f}")
                difficulty_stats = item_results['Dificuldade (b)'].describe()
                st.write("**Estatísticas de Dificuldade:**")
                st.json({"Média": round(difficulty_stats['mean'], 3), "Desvio Padrão": round(difficulty_stats['std'], 3), "Range": round(difficulty_stats['max'] - difficulty_stats['min'], 3)})
            with col2:
                fig = px.histogram(item_results, x='Dificuldade (b)', title='Distribuição da Dificuldade dos Itens', nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab_report:
            st.subheader("📄 Relatório Completo para Download")
            st.success("✅ Relatório pronto! Clique abaixo para baixar.")
            
            # --- GERA O RELATÓRIO MODIFICADO ---
            html_report = generate_html_report(student_results, item_results)
            st.download_button(
                label="📥 Baixar Relatório Completo em HTML", 
                data=html_report, 
                file_name=f"relatorio_tri_{datetime.now().strftime('%Y%m%d')}.html", 
                mime="text/html", 
                use_container_width=True
            )