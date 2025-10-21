import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
from datetime import datetime
import hashlib
import hmac

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DE AUTENTICAÇÃO ---
def check_password():
    """Retorna `True` se o usuário tiver a senha correta."""

    def password_entered():
        """Verifica se a senha inserida está correta."""
        if st.session_state["username"] in st.secrets["passwords"]:
            correct_username = st.session_state["username"]
            correct_password = st.secrets["passwords"][correct_username]
            
            # Verifica a senha usando hash seguro
            input_password = st.session_state["password"]
            if hmac.compare_digest(input_password, correct_password):
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = correct_username
                del st.session_state["password"]  # Não armazena a senha
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # Primeiro acesso, mostrar inputs para login
    if "password_correct" not in st.session_state:
        st.title("🔒 Sistema de Análise TRI - Acesso Restrito")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Login de Acesso")
            
            with st.form("login_form"):
                st.text_input("Usuário", key="username", help="Digite seu nome de usuário")
                st.text_input("Senha", type="password", key="password", help="Digite sua senha")
                st.form_submit_button("Entrar", on_click=password_entered)
            
            if "password_correct" in st.session_state:
                if not st.session_state["password_correct"]:
                    st.error("😕 Usuário ou senha incorretos")
                    st.info("💡 Contate o administrador para obter acesso")
            else:
                st.info("👆 Por favor, faça login para acessar o sistema")
        
        st.stop()
    
    # Senha incorreta, mostrar erro
    elif not st.session_state["password_correct"]:
        st.title("🔒 Sistema de Análise TRI - Acesso Restrito")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("😕 Usuário ou senha incorretos")
            
            with st.form("login_retry"):
                st.text_input("Usuário", key="username", help="Digite seu nome de usuário")
                st.text_input("Senha", type="password", key="password", help="Digite sua senha")
                st.form_submit_button("Tentar Novamente", on_click=password_entered)
            
            st.info("💡 Contate o administrador para obter acesso")
        
        st.stop()
    
    # Senha correta, mostrar aplicação
    else:
        # Mostrar header com informações do usuário
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("🎓 Corretor e Analisador TRI")
        with col3:
            st.write(f"👤 **{st.session_state.current_user}**")
            if st.button("🚪 Sair"):
                for key in st.session_state.keys():
                    if key not in ['results_df']:  # Mantém os dados, remove apenas o login
                        del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        return True

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Corretor e Analisador CESB",
    page_icon="📚",
    layout="wide"
)

# --- VERIFICAR LOGIN ANTES DE CONTINUAR ---
if not check_password():
    st.stop()

# --- PARTE 1: LÓGICA E FUNÇÕES DE ANÁLISE TRI (MODELO 3PL SIMPLIFICADO) ---

def probability_3pl(theta, a, b, c):
    """Função de resposta do item - modelo 3PL"""
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def log_likelihood_3pl(params, response_matrix):
    """Função de verossimilhança para o modelo 3PL"""
    n_students, n_items = response_matrix.shape
    ability = params[:n_students]
    discrimination = params[n_students:n_students+n_items]
    difficulty = params[n_students+n_items:n_students+2*n_items]
    guessing = params[n_students+2*n_items:]
    
    log_lik = 0
    for i in range(n_students):
        for j in range(n_items):
            p = probability_3pl(ability[i], discrimination[j], difficulty[j], guessing[j])
            if response_matrix[i, j] == 1:
                log_lik += np.log(p)
            else:
                log_lik += np.log(1 - p)
    return -log_lik  # Negative for minimization

@st.cache_data
def run_advanced_tri_analysis(df):
    """Executa a análise TRI completa (3PL) com método simplificado"""
    if df.columns[0].lower() in ['aluno', 'user', 'nome', 'student']:
        student_names = df[df.columns[0]]
        df = df.set_index(df.columns[0])
    else:
        student_names = pd.Series([f'Aluno_{i + 1}' for i in range(len(df))])
    
    # Converte para numérico, tratando erros
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    response_matrix = df.to_numpy()
    
    n_students, n_items = response_matrix.shape
    ability_range = np.linspace(-4, 4, 100)
    
    # --- Estimação Simplificada dos Parâmetros TRI ---
    try:
        # Estimação inicial baseada na Teoria Clássica dos Testes
        p_values = response_matrix.mean(axis=0)  # Proporção de acerto por item
        total_scores = response_matrix.sum(axis=1)
        
        # Dificuldade inicial (transformação logística)
        difficulty_initial = -np.log(p_values / (1 - p_values + 1e-8))
        difficulty_initial = np.nan_to_num(difficulty_initial, nan=0.0)
        
        # Discriminação inicial (correlação bisserial)
        discrimination_initial = []
        for j in range(n_items):
            try:
                corr = np.corrcoef(response_matrix[:, j], total_scores)[0, 1]
                discrimination_initial.append(corr if not np.isnan(corr) else 0.8)
            except:
                discrimination_initial.append(0.8)
        discrimination_initial = np.array(discrimination_initial)
        
        # Adivinhação inicial (assume 1/num_opcoes para questões muito difíceis)
        guessing_initial = np.where(p_values < 0.3, 0.25, 0.1)
        
        # Habilidade inicial (padronização da pontuação total)
        ability_initial = (total_scores - total_scores.mean()) / total_scores.std()
        ability_initial = np.nan_to_num(ability_initial, nan=0.0)
        
        # Ajuste fino usando otimização (apenas para habilidade, mantendo outros parâmetros fixos)
        def optimize_ability_for_student(i, response_vector, discrimination, difficulty, guessing):
            def student_log_lik(theta):
                log_lik = 0
                for j in range(n_items):
                    p = probability_3pl(theta, discrimination[j], difficulty[j], guessing[j])
                    if response_vector[j] == 1:
                        log_lik += np.log(p)
                    else:
                        log_lik += np.log(1 - p)
                return -log_lik
            
            result = minimize(student_log_lik, ability_initial[i], method='BFGS', 
                            options={'gtol': 1e-4})
            return result.x[0] if result.success else ability_initial[i]
        
        # Otimiza habilidade para cada aluno
        ability_final = []
        for i in range(n_students):
            theta_opt = optimize_ability_for_student(
                i, response_matrix[i], discrimination_initial, difficulty_initial, guessing_initial
            )
            ability_final.append(theta_opt)
        
        ability_final = np.array(ability_final)
        
        model_params = {
            'difficulty': difficulty_initial,
            'discrimination': discrimination_initial,
            'guessing': guessing_initial,
            'ability': ability_final,
            'n_items': n_items,
            'n_students': n_students
        }

        # --- Resultados dos Alunos ---
        student_results = pd.DataFrame({
            'Aluno': student_names,
            'Proficiência (θ)': model_params['ability'],
            'Pontuação Total': response_matrix.sum(axis=1),
            'Percentual de Acerto': (response_matrix.sum(axis=1) / model_params['n_items'] * 100).round(2),
            'Z-Score': (model_params['ability'] - model_params['ability'].mean()) / model_params['ability'].std()
        })
        
        # --- Resultados dos Itens ---
        # Calcula métricas clássicas (TCT) para complementar a TRI
        corr_bisserial = []
        idx_discrimination = []
        
        # Separa grupos de alta e baixa proficiência
        median_score = np.median(total_scores)
        high_group = total_scores > median_score
        low_group = total_scores <= median_score

        for i in range(n_items):
            try:
                # Correlação Bisserial Pontual (item vs score total)
                corr = np.corrcoef(response_matrix[:, i], total_scores)[0, 1]
                corr_bisserial.append(corr if not np.isnan(corr) else 0)
            except:
                corr_bisserial.append(0)
            
            # Índice de Discriminação Clássico (Diferença entre grupos)
            idx_discrimination.append(
                response_matrix[high_group, i].mean() - response_matrix[low_group, i].mean()
            )

        item_results = pd.DataFrame({
            'Questão': df.columns,
            'Dificuldade': model_params['difficulty'],
            'Discriminação': model_params['discrimination'],
            'Adivinhação': model_params['guessing'],
            'Percentual Acerto': (response_matrix.mean(axis=0) * 100).round(2),
            'Índice Discriminação': idx_discrimination,
            'Correlação Bisserial': corr_bisserial
        })
        
        # --- Dados da Curva Característica do Item (CCI) ---
        cci_data = []
        for i, item in enumerate(df.columns):
            for theta in ability_range:
                prob = probability_3pl(theta,
                                       model_params['discrimination'][i],
                                       model_params['difficulty'][i],
                                       model_params['guessing'][i])
                cci_data.append({
                    'Questão': item, 'Theta': theta, 'Probabilidade': prob,
                    'Dificuldade': model_params['difficulty'][i],
                    'Discriminação': model_params['discrimination'][i],
                    'Adivinhação': model_params['guessing'][i]
                })
        cci_df = pd.DataFrame(cci_data)
        
        # --- Confiabilidade (Alfa de Cronbach) ---
        model_params['alpha'] = calculate_cronbach_alpha(response_matrix)
        
        return student_results, item_results, cci_df, model_params
        
    except Exception as e:
        st.error(f"Erro na análise TRI: {e}")
        return None, None, None, None

def calculate_cronbach_alpha(response_matrix):
    """ Calcula o Alfa de Cronbach, uma medida de confiabilidade (consistência interna). """
    try:
        n_items = response_matrix.shape[1]
        if n_items < 2:
            return 0.0
        
        # Variância da pontuação total dos alunos
        variance_total = response_matrix.sum(axis=1).var(ddof=1)
        if variance_total == 0:
            return 0.0
            
        # Soma da variância de cada item individual
        variance_items = response_matrix.var(axis=0, ddof=1).sum()
        
        alpha = (n_items / (n_items - 1)) * (1 - variance_items / variance_total)
        return max(0, min(1, alpha)) # Alpha está entre 0 e 1
    except:
        return 0.0 # Retorna 0 em caso de erro

# --- Funções de Gráfico (Atualizadas para 3PL) ---
def plot_cci(cci_df, selected_items):
    fig = go.Figure()
    for item in selected_items:
        item_data = cci_df[cci_df['Questão'] == item]
        b = item_data["Dificuldade"].iloc[0]
        c = item_data["Adivinhação"].iloc[0]
        fig.add_trace(go.Scatter(
            x=item_data['Theta'], y=item_data['Probabilidade'], mode='lines',
            name=f'{item} (b={b:.2f}, c={c:.2f})',
            hovertemplate=f'Questão: {item}<br>θ: %{{x:.2f}}<br>P(θ): %{{y:.3f}}<extra></extra>'
        ))
    fig.update_layout(title='Curvas Características dos Itens (CCI - 3PL)', xaxis_title='Habilidade (θ)', yaxis_title='Probabilidade de Acerto P(θ)', hovermode='closest', height=500)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="P(θ) = 0.5")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig

def plot_information_curve(cci_df, selected_items):
    """ Atualizado para a fórmula de informação do 3PL """
    fig = go.Figure()
    for item in selected_items:
        item_data = cci_df[cci_df['Questão'] == item]
        p = item_data['Probabilidade']
        a = item_data['Discriminação'].iloc[0]
        c = item_data['Adivinhação'].iloc[0]
        
        # Fórmula de Informação do 3PL
        p_minus_c = p - c
        one_minus_p = 1 - p
        one_minus_c_sq = (1 - c) ** 2
        
        numerator = (a**2) * (p_minus_c**2) * one_minus_p
        denominator = p * one_minus_c_sq
        
        # Evita divisão por zero e valores negativos
        information = (numerator / denominator).fillna(0).clip(0)

        fig.add_trace(go.Scatter(x=item_data['Theta'], y=information, mode='lines', name=f'{item}', hovertemplate=f'Questão: {item}<br>θ: %{{x:.2f}}<br>Info: %{{y:.3f}}<extra></extra>'))
    
    fig.update_layout(title='Função de Informação dos Itens (3PL)', xaxis_title='Habilidade (θ)', yaxis_title='Informação de Fisher', hovermode='closest', height=500)
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
    fig = px.scatter(item_results, x='Dificuldade', y='Discriminação', 
                     size='Percentual Acerto', color='Correlação Bisserial', 
                     hover_name='Questão', 
                     hover_data=['Adivinhação', 'Índice Discriminação'],
                     title='Análise Multidimensional dos Itens (TRI)', 
                     labels={'Dificuldade': 'Dificuldade (b)', 
                             'Discriminação': 'Discriminação (a)', 
                             'Percentual Acerto': '% de Acerto', 
                             'Correlação Bisserial': 'Corr. Bisserial'})
    return fig

def plot_difficulty_discrimination_heatmap(item_results):
    item_results['Dificuldade_Cat'] = pd.cut(item_results['Dificuldade'], bins=[-np.inf, -1, 0, 1, np.inf], labels=['Muito Fácil', 'Fácil', 'Difícil', 'Muito Difícil'])
    item_results['Discriminação_Cat'] = pd.cut(item_results['Discriminação'], bins=[-np.inf, 0.3, 0.6, 1, 1.7, np.inf], labels=['Baixa', 'Média', 'Alta', 'Muito Alta', 'Extrema'])
    heatmap_data = pd.crosstab(item_results['Dificuldade_Cat'], item_results['Discriminação_Cat'])
    fig = px.imshow(heatmap_data, title='Mapa de Calor: Dificuldade vs Discriminação', color_continuous_scale='Viridis', aspect='auto')
    fig.update_layout(xaxis_title='Poder de Discriminação', yaxis_title='Nível de Dificuldade')
    return fig

# --- PARTE 2: FUNÇÃO DE RELATÓRIO HTML SIMPLIFICADA ---

def generate_conclusion_text(student_results, item_results, model_params, problematic_items):
    """Gera texto de conclusão personalizado baseado na análise"""
    
    n_students = len(student_results)
    n_items = len(item_results)
    alpha = model_params['alpha']
    avg_proficiency = student_results['Proficiência (θ)'].mean()
    std_proficiency = student_results['Proficiência (θ)'].std()
    
    # Avaliação da confiabilidade
    if alpha >= 0.8:
        reliability_text = "excelente confiabilidade"
        reliability_emoji = "🎉"
    elif alpha >= 0.7:
        reliability_text = "boa confiabilidade"
        reliability_emoji = "✅"
    elif alpha >= 0.6:
        reliability_text = "confiabilidade moderada"
        reliability_emoji = "⚠️"
    else:
        reliability_text = "baixa confiabilidade - precisa de revisão"
        reliability_emoji = "🔴"
    
    # Avaliação da distribuição de proficiência
    if std_proficiency > 1.0:
        distribution_text = "boa variabilidade entre os alunos"
        distribution_emoji = "📊"
    elif std_proficiency > 0.5:
        distribution_text = "variabilidade moderada"
        distribution_emoji = "📈"
    else:
        distribution_text = "pouca variabilidade - alunos muito homogêneos"
        distribution_emoji = "📉"
    
    # Avaliação dos itens problemáticos
    n_problematic = len(problematic_items)
    if n_problematic == 0:
        items_text = "Todos os itens apresentam qualidade psicométrica adequada"
        items_emoji = "🏆"
    elif n_problematic <= 2:
        items_text = f"Apenas {n_problematic} itens precisam de atenção"
        items_emoji = "💡"
    else:
        items_text = f"{n_problematic} itens necessitam de revisão urgente"
        items_emoji = "🚨"
    
    # Texto de conclusão final
    conclusion = f"""
    <div class="conclusion-box">
        <h4>{reliability_emoji} CONCLUSÃO FINAL DA AVALIAÇÃO</h4>
        <p>Com base na análise completa do teste aplicado a <strong>{n_students} alunos</strong> com <strong>{n_items} questões</strong>, podemos concluir que:</p>
        
        <div class="conclusion-points">
            <p>📚 <strong>Qualidade do Teste:</strong> O instrumento apresenta <strong>{reliability_text}</strong> (α = {alpha:.3f}), indicando que mede consistentemente o construto avaliado.</p>
            
            <p>{distribution_emoji} <strong>Perfil dos Alunos:</strong> A turma mostra {distribution_text}, com proficiência média de {avg_proficiency:.2f} e desvio padrão de {std_proficiency:.2f}.</p>
            
            <p>{items_emoji} <strong>Qualidade dos Itens:</strong> {items_text}. Isso representa {n_problematic/n_items*100:.1f}% do total de questões.</p>
            
            <p>🎯 <strong>Recomendação Geral:</strong> {"O teste pode ser utilizado com confiança para avaliações futuras." if n_problematic == 0 else "Recomendamos revisar os itens problemáticos antes de reutilizar o teste."}</p>
        </div>
        
        <div class="final-thought">
            <p>💫 <strong>Insight Final:</strong> {"Parabéns! Seu teste está bem calibrado e é uma ferramenta confiável para avaliação." if n_problematic <= 2 else "Com pequenos ajustes, seu teste se tornará uma ferramenta ainda mais precisa!"}</p>
        </div>
    </div>
    """
    
    return conclusion

def generate_html_report(student_results, item_results, model_params):
    """
    Gera um relatório HTML simplificado focado nos resultados mais importantes para impressão.
    """
    # --- Estilo CSS Simplificado ---
    html_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        body { 
            font-family: 'Inter', sans-serif; 
            line-height: 1.6; 
            color: #2d3748; 
            background: white;
            margin: 0;
            padding: 20px;
        }
        
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
        }
        
        h1, h2, h3 { 
            color: #2d3748;
            margin-bottom: 15px;
        }
        
        h1 { 
            font-size: 2.5em; 
            font-weight: 700;
            color: #2d3748;
            text-align: center;
            margin-bottom: 10px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        
        h2 { 
            font-size: 1.8em; 
            font-weight: 600;
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin-top: 40px;
            color: #2d3748;
        }
        
        h3 { 
            font-size: 1.4em; 
            font-weight: 500;
            color: #4a5568;
            margin-top: 30px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }
        
        .header-subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #718096;
            margin-bottom: 30px;
            font-weight: 400;
        }
        
        .explanation { 
            background: #f7fafc;
            border-left: 4px solid #4299e1;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        
        .explanation h4 {
            color: #2b6cb0;
            margin-top: 0;
            font-size: 1.2em;
        }
        
        .compact-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .compact-table th, 
        .compact-table td { 
            padding: 10px 8px; 
            border: 1px solid #e2e8f0; 
            text-align: center;
        }
        
        .compact-table th { 
            background: #667eea;
            color: white;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .compact-table tr:nth-child(even) { 
            background-color: #f7fafc; 
        }
        
        .compact-table tr:hover {
            background-color: #edf2f7;
        }
        
        .ranking-table .position {
            font-weight: bold;
            background: #48bb78;
            color: white;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            font-size: 0.8em;
        }
        
        .ranking-table .top-3 {
            background: #d69e2e;
        }
        
        .metric-box { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-top: 25px;
        }
        
        .metric { 
            background: #667eea;
            padding: 25px 15px;
            border-radius: 12px;
            text-align: center;
            color: white;
        }
        
        .metric .label { 
            font-size: 0.9em; 
            opacity: 0.9;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .metric .value { 
            font-size: 2em; 
            font-weight: 700;
        }
        
        .warning { 
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e53e3e;
            margin: 20px 0;
            font-weight: 600;
        }
        
        .success { 
            background: #c6f6d5;
            color: #276749;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #38a169;
            margin: 20px 0;
            font-weight: 600;
        }
        
        .info-box {
            background: #bee3f8;
            color: #2c5282;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #3182ce;
        }
        
        .tip {
            background: #faf089;
            color: #744210;
            padding: 12px 15px;
            border-radius: 6px;
            margin: 12px 0;
            border-left: 4px solid #d69e2e;
        }
        
        .conclusion-box {
            background: #e9d8fd;
            color: #553c9a;
            padding: 20px;
            border-radius: 10px;
            margin: 25px 0;
            border-left: 4px solid #9f7aea;
        }
        
        .conclusion-points {
            margin: 15px 0;
        }
        
        .conclusion-points p {
            margin: 10px 0;
            padding: 8px;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 6px;
            border-left: 3px solid #9f7aea;
        }
        
        .final-thought {
            background: rgba(255, 255, 255, 0.9);
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #9f7aea;
            margin-top: 15px;
        }
        
        .section-icon {
            font-size: 1.3em;
            margin-right: 8px;
            vertical-align: middle;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 0.85em;
        }
        
        @media print {
            .container {
                box-shadow: none;
                padding: 15px;
            }
            .metric-box {
                break-inside: avoid;
            }
            .compact-table {
                break-inside: avoid;
            }
        }
    </style>
    """

    # --- Gráficos e Tabelas ---
    theta_dist_html = plot_theta_distribution(student_results).to_html(full_html=False, include_plotlyjs='cdn')
    
    # --- Tabela de Ranking Top 10 ---
    ranking_df = student_results[['Aluno', 'Percentual de Acerto', 'Proficiência (θ)']].copy()
    ranking_df.columns = ['Aluno', 'Nota Acertos', 'Nota TRI']
    ranking_df['Nota TRI'] = ranking_df['Nota TRI'].round(3)
    ranking_df = ranking_df.sort_values('Nota TRI', ascending=False).head(10).reset_index(drop=True)
    ranking_df['Posição'] = ranking_df.index + 1
    
    # Criar tabela de ranking compacta
    ranking_html = """
        <table class="compact-table ranking-table">
            <thead>
                <tr>
                    <th>🏆 Posição</th>
                    <th>👤 Aluno</th>
                    <th>📊 Nota Acertos</th>
                    <th>🎯 Nota TRI</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for idx, row in ranking_df.iterrows():
        position_class = "top-3" if idx < 3 else ""
        ranking_html += f"""
                <tr>
                    <td><div class="position {position_class}">{row['Posição']}º</div></td>
                    <td><strong>{row['Aluno']}</strong></td>
                    <td>{row['Nota Acertos']}%</td>
                    <td>{row['Nota TRI']:.3f}</td>
                </tr>
        """
    
    ranking_html += """
            </tbody>
        </table>
    """
    
    # --- Tabela de Itens Problemáticos ---
    problematic_items = item_results[
        (item_results['Discriminação'] < 0.3) | 
        (item_results['Percentual Acerto'] < 15) | 
        (item_results['Percentual Acerto'] > 95) |
        (item_results['Adivinhação'] > 0.4)
    ]
    
    # --- Cálculo de métricas adicionais para o relatório ---
    alpha_quality = "Excelente" if model_params['alpha'] > 0.8 else "Boa" if model_params['alpha'] > 0.7 else "Moderada" if model_params['alpha'] > 0.6 else "Baixa"
    avg_difficulty = item_results['Dificuldade'].mean()
    difficulty_level = "Equilibrada" if -0.5 <= avg_difficulty <= 0.5 else "Fácil" if avg_difficulty < -0.5 else "Difícil"
    
    # Tabela de Itens Problemáticos (se houver)
    items_problematic_html = ""
    if len(problematic_items) > 0:
        items_problematic_html = """
        <table class="compact-table">
            <thead>
                <tr>
                    <th>🔢 Questão</th>
                    <th>🎯 Dificuldade</th>
                    <th>⚡ Discriminação</th>
                    <th>🎲 Adivinhação</th>
                    <th>📊 Acerto</th>
                    <th>⚠️ Problema</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for _, row in problematic_items.iterrows():
            issues = []
            if row['Discriminação'] < 0.3: issues.append("Baixa discriminação")
            if row['Percentual Acerto'] < 15: issues.append("Muito difícil")
            if row['Percentual Acerto'] > 95: issues.append("Muito fácil")
            if row['Adivinhação'] > 0.4: issues.append("Alto chute")
            
            items_problematic_html += f"""
                <tr>
                    <td><strong>{row['Questão']}</strong></td>
                    <td>{row['Dificuldade']:.3f}</td>
                    <td>{row['Discriminação']:.3f}</td>
                    <td>{row['Adivinhação']:.3f}</td>
                    <td>{row['Percentual Acerto']:.1f}%</td>
                    <td>{', '.join(issues)}</td>
                </tr>
            """
        
        items_problematic_html += """
            </tbody>
        </table>
        """
    
    recommendations_html = ""
    if len(problematic_items) > 0:
        recommendations_html += f"""
        <div class="warning">
            <h4>⚠️ Atenção: {len(problematic_items)} Itens Problemáticos</h4>
            <p>Identificamos {len(problematic_items)} itens que podem estar com problemas e precisam de revisão.</p>
        </div>
        """
    else:
        recommendations_html = """
        <div class="success">
            <h4>🎉 Excelente Notícia!</h4>
            <p>Todos os itens do teste apresentam características psicométricas adequadas.</p>
        </div>
        """

    # Gerar texto de conclusão
    conclusion_html = generate_conclusion_text(student_results, item_results, model_params, problematic_items)

    # --- Montagem do Corpo do HTML ---
    html_body = f"""
    <div class="container">
        <h1>📊 Relatório de Análise TRI</h1>
        <div class="header-subtitle">Análise Psicométrica - Foco nos Resultados Principais</div>
        
        <div class="info-box">
            <h4>📅 Informações do Relatório</h4>
            <p><strong>Data de Geração:</strong> {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
            <p><strong>Metodologia:</strong> Modelo 3PL (Três Parâmetros)</p>
        </div>

        <h2><span class="section-icon">🎯</span> Resumo Executivo</h2>
        <div class="explanation">
            <p>Este relatório apresenta os <strong>resultados mais importantes</strong> da análise psicométrica usando Teoria de Resposta ao Item.</p>
        </div>
        
        <div class="metric-box">
            <div class="metric">
                <div class="label">👥 Alunos</div>
                <div class="value">{len(student_results)}</div>
            </div>
            <div class="metric">
                <div class="label">📝 Questões</div>
                <div class="value">{len(item_results)}</div>
            </div>
            <div class="metric">
                <div class="label">🎓 Proficiência Média</div>
                <div class="value">{student_results['Proficiência (θ)'].mean():.3f}</div>
            </div>
            <div class="metric">
                <div class="label">🛡️ Confiabilidade</div>
                <div class="value">{model_params['alpha']:.3f}</div>
            </div>
        </div>

        <h2><span class="section-icon">🎓</span> Desempenho dos Alunos</h2>
        
        {theta_dist_html}
        
        <h3>🏆 Top 10 - Ranking de Proficiência</h3>
        {ranking_html}

        <h2><span class="section-icon">🔍</span> Análise dos Itens</h2>

        {recommendations_html}

        {items_problematic_html if len(problematic_items) > 0 else ""}

        <div class="tip">
            <h4>💡 Dicas para Melhoria</h4>
            <p><strong>Itens ideais:</strong> Dificuldade entre -0.5 e 0.5 | Discriminação acima de 0.8 | Adivinhação próxima de 0.25</p>
        </div>

        <h2><span class="section-icon">📈</span> Estatísticas do Teste</h2>
        <div class="explanation">
            <p><strong>Confiabilidade (Alpha):</strong> {model_params['alpha']:.3f} - {alpha_quality}</p>
            <p><strong>Dificuldade Média:</strong> {avg_difficulty:.3f} - {difficulty_level}</p>
            <p><strong>Itens Problemáticos:</strong> {len(problematic_items)} de {len(item_results)} ({len(problematic_items)/len(item_results)*100:.1f}%)</p>
        </div>

        <h2><span class="section-icon">🎓</span> Conclusão da Avaliação</h2>
        {conclusion_html}

        <div class="footer">
            <p>📄 Relatório gerado automaticamente - Foco nos resultados principais para análise rápida</p>
            <p>✨ <strong>Versão Simplificada para Impressão</strong></p>
        </div>
    </div>
    """
    
    full_html = f"<!DOCTYPE html><html><head><title>Relatório TRI Simplificado</title>{html_style}</head><body>{html_body}</body></html>"
    return full_html

# --- PARTE 3: INTERFACE STREAMLIT (APÓS LOGIN) ---

# --- Inicialização do Banco de Dados na Sessão ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# --- Configuração da Prova na Sidebar ---
st.sidebar.header("⚙️ 1. Configurar Prova")
gabarito_correto_str = st.sidebar.text_input(
    "Gabarito Correto (ex: ABDEC...)", 
    "EBDEABDCABEEBADEABDCABEDCD"
).upper()

if not gabarito_correto_str:
    st.sidebar.error("Por favor, insira um gabarito correto para continuar.")
    st.stop()

# --- Deriva N° de Questões e Opções do Gabarito ---
n_items = len(gabarito_correto_str)
options_list = sorted(list(set(gabarito_correto_str)))
options_with_blank = options_list + ["BRANCO"]

st.sidebar.info(f"📝 Prova configurada para **{n_items} questões** com **{len(options_list)} opções** ({', '.join(options_list)}).")

# --- Gerador de Dados Aleatórios ---
st.sidebar.header("🎲 Gerar Dados de Teste")
if st.sidebar.button("✨ Gerar Planilha Aleatória (50 alunos)"):
    # Gera dados aleatórios realistas
    nomes_base = [
        "Ana Silva", "Carlos Oliveira", "Maria Santos", "João Pereira", "Luiza Costa",
        "Pedro Almeida", "Fernanda Lima", "Rafael Souza", "Juliana Rocha", "Marcos Santos",
        "Patrícia Ferreira", "Bruno Carvalho", "Amanda Dias", "Lucas Martins", "Carla Ribeiro"
    ]
    
    # Expande a lista para 50 alunos
    nomes_alunos = []
    for i in range(50):
        if i < len(nomes_base):
            nomes_alunos.append(nomes_base[i])
        else:
            nomes_alunos.append(f"Aluno_{i+1}")
    
    # Gera respostas aleatórias com padrão realista
    dados_aleatorios = {"Aluno": nomes_alunos}
    
    for i in range(n_items):
        questao = f"Q{i+1}"
        # Cria um padrão onde alunos melhores tendem a acertar mais
        respostas = []
        for aluno_idx in range(50):
            # Simula habilidade do aluno (normal distribuída)
            habilidade = np.random.normal(0, 1)
            # Dificuldade da questão (varia entre questões)
            dificuldade = np.random.normal(0, 1)
            # Probabilidade de acerto baseada na diferença habilidade-dificuldade
            prob_acerto = 1 / (1 + np.exp(-(habilidade - dificuldade)))
            
            if np.random.random() < prob_acerto:
                respostas.append(1)
            else:
                respostas.append(0)
        
        dados_aleatorios[questao] = respostas
    
    st.session_state.results_df = pd.DataFrame(dados_aleatorios)
    st.sidebar.success("✅ 50 alunos aleatórios gerados com sucesso!")

# --- Definir Abas ---
tab_entry, tab_analysis = st.tabs([" ✏️ Lançamento de Respostas ", " 📊 Análise TRI e Relatório "])

# --- ABA 1: Lançamento de Respostas ---
with tab_entry:
    st.header("✏️ Lançamento Manual de Respostas")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Novo Lançamento")
        id_aluno = st.text_input("👤 Nome/ID do Aluno:")
        
        with st.form("entry_form"):
            student_answers = {}
            
            for i in range(n_items):
                q_num = i + 1
                correct_answer = gabarito_correto_str[i]
                
                student_answers[f'Q{q_num}'] = st.radio(
                    f"**Questão {q_num}** 🎯 (Gabarito: {correct_answer})",
                    options_with_blank,
                    horizontal=True,
                    key=f"q_{id_aluno}_{q_num}"
                )
            
            submitted = st.form_submit_button("💾 Salvar Respostas deste Aluno")
        
        if submitted:
            if not id_aluno:
                st.error("❌ Por favor, insira um Nome/ID do Aluno antes de salvar.")
            elif id_aluno in st.session_state.results_df.get('Aluno', []):
                 st.error(f"❌ Erro: O aluno '{id_aluno}' já existe na matriz. Use um nome diferente.")
            else:
                binary_responses = {}
                for q_key, answer in student_answers.items():
                    q_idx = int(q_key[1:]) - 1
                    if answer == gabarito_correto_str[q_idx]:
                        binary_responses[q_key] = 1
                    else:
                        binary_responses[q_key] = 0
                
                new_row_data = {"Aluno": id_aluno, **binary_responses}
                new_row_df = pd.DataFrame([new_row_data])
                
                st.session_state.results_df = pd.concat(
                    [st.session_state.results_df, new_row_df], 
                    ignore_index=True
                )
                st.success(f"✅ Respostas de '{id_aluno}' salvas com sucesso!")
                st.balloons()

    with col2:
        st.subheader("📋 Matriz de Respostas (Banco de Dados)")
        if st.session_state.results_df.empty:
            st.info("💡 Nenhum aluno foi lançado ainda. Use o gerador na sidebar ou adicione alunos manualmente.")
        else:
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
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

    min_students = 5
    if len(st.session_state.results_df) < min_students: 
        st.warning(f"⚠️ É necessário lançar pelo menos {min_students} alunos para uma análise TRI. (Atualmente: {len(st.session_state.results_df)})")
    else:
        st.info(f"🎯 Pronto para analisar **{len(st.session_state.results_df)} alunos** e **{n_items} questões**.")
        
        if st.button("🚀 Executar Análise TRI Avançada (3PL)", type="primary", use_container_width=True):
            with st.spinner('🔍 Realizando análise psicométrica... (Isso pode levar alguns segundos)'):
                try:
                    df_to_analyze = st.session_state.results_df.copy()
                    
                    student_results, item_results, cci_df, model_params = run_advanced_tri_analysis(df_to_analyze)
                    
                    if student_results is not None:
                        st.session_state['tri_student_results'] = student_results
                        st.session_state['tri_item_results'] = item_results
                        st.session_state['tri_cci_df'] = cci_df
                        st.session_state['tri_model_params'] = model_params
                        
                        st.success('✅ Análise TRI (3PL) concluída com sucesso!')
                    else:
                        st.error("❌ Falha na análise TRI. Verifique os dados e tente novamente.")
                except Exception as e:
                    st.error(f"❌ Erro durante a análise TRI: {e}")

    if 'tri_student_results' in st.session_state:
        student_results = st.session_state['tri_student_results']
        item_results = st.session_state['tri_item_results']
        cci_df = st.session_state['tri_cci_df']
        model_params = st.session_state['tri_model_params']
        
        st.markdown("---")
        
        tab_alunos, tab_itens, tab_curvas, tab_diag, tab_report = st.tabs([
            " 🎓 Análise dos Alunos", 
            " 📝 Análise dos Itens", 
            " 📈 Curvas TRI", 
            " 🔍 Diagnóstico", 
            " 📋 Relatório HTML"
        ])
        
        with tab_alunos:
            st.subheader("🎓 Análise da Proficiência dos Alunos")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_theta_distribution(student_results), use_container_width=True)
                st.dataframe(student_results.sort_values('Proficiência (θ)', ascending=False).head(10).reset_index(drop=True), use_container_width=True)
            with col2:
                fig = px.scatter(student_results, x='Pontuação Total', y='Proficiência (θ)', hover_data=['Aluno', 'Percentual de Acerto'], title='Relação: Proficiência vs Pontuação Total', trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Estatísticas Descritivas")
            col_stats = st.columns(4)
            with col_stats[0]: st.metric("📈 Média θ", f"{student_results['Proficiência (θ)'].mean():.3f}")
            with col_stats[1]: st.metric("📊 Desvio Padrão", f"{student_results['Proficiência (θ)'].std():.3f}")
            with col_stats[2]: st.metric("📉 Mínimo", f"{student_results['Proficiência (θ)'].min():.3f}")
            with col_stats[3]: st.metric("📈 Máximo", f"{student_results['Proficiência (θ)'].max():.3f}")
        
        with tab_itens:
            st.subheader("📝 Análise Psicométrica dos Itens")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_difficulty_discrimination_scatter(item_results), use_container_width=True)
            with col2:
                st.plotly_chart(plot_difficulty_discrimination_heatmap(item_results), use_container_width=True)
            st.subheader("📋 Tabela de Parâmetros dos Itens (3PL)")
            st.dataframe(item_results.sort_values('Dificuldade').round(3), use_container_width=True)
        
        with tab_curvas:
            st.subheader("📈 Curvas Características e Informação")
            col1, col2 = st.columns(2)
            with col1:
                default_selection = item_results['Questão'].head(3).tolist()
                selected_items_cci = st.multiselect("🎯 Selecione as questões para análise:", options=item_results['Questão'].tolist(), default=default_selection)
            with col2:
                st.info("💡 **Interpretação das CCIs (3PL):**")
                st.markdown("- **📊 Curva à esquerda**: Questão mais fácil")
                st.markdown("- **🎯 Curva à direita**: Questão mais difícil") 
                st.markdown("- **⚡ Curva mais íngreme**: Melhor discriminação")
                st.markdown("- **🎲 Ponto inicial (esquerda)**: Prob. de chute (parâmetro 'c')")
            if selected_items_cci:
                col_cci1, col_cci2 = st.columns(2)
                with col_cci1: st.plotly_chart(plot_cci(cci_df, selected_items_cci), use_container_width=True)
                with col_cci2: st.plotly_chart(plot_information_curve(cci_df, selected_items_cci), use_container_width=True)
        
        with tab_diag:
            st.subheader("🔍 Diagnóstico do Teste")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🛡️ Confiabilidade (Alpha de Cronbach)", f"{model_params['alpha']:.3f}")
                avg_discrimination_tct = item_results['Índice Discriminação'].mean()
                st.metric("⚡ Discriminação Média (TCT)", f"{avg_discrimination_tct:.3f}")
                difficulty_stats = item_results['Dificuldade'].describe()
                st.write("**🎯 Estatísticas de Dificuldade (b):**")
                st.json({"📈 Média": round(difficulty_stats['mean'], 3), "📊 Desvio Padrão": round(difficulty_stats['std'], 3), "📐 Range": round(difficulty_stats['max'] - difficulty_stats['min'], 3)})
            with col2:
                fig = px.histogram(item_results, x='Dificuldade', title='Distribuição da Dificuldade dos Itens', nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab_report:
            st.subheader("📄 Relatório Completo para Download")
            st.success("✅ Relatório simplificado pronto para impressão! Clique abaixo para baixar.")
            
            html_report = generate_html_report(student_results, item_results, model_params)
            st.download_button(
                label="📥 Baixar Relatório Simplificado (HTML)", 
                data=html_report, 
                file_name=f"relatorio_tri_simplificado_{datetime.now().strftime('%Y%m%d_%H%M')}.html", 
                mime="text/html", 
                use_container_width=True
            )