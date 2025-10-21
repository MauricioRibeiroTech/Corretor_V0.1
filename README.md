# Corretor de Gabaritos e Analisador Psicrométrico (TRI)

Este projeto é um dashboard interativo construído com Streamlit que serve a dois propósitos principais:

1.  **Corretor Manual:** Permite o lançamento manual de respostas de alunos em uma prova de múltipla escolha e as corrige com base em um gabarito pré-definido.
2.  **Analisador Psicrométrico:** Realiza uma análise estatística avançada da prova e dos alunos usando os princípios da **Teoria de Resposta ao Item (TRI)**.

O objetivo é ir além da simples nota de "porcentagem de acerto", oferecendo uma visão mais profunda sobre a qualidade da prova, a eficácia de cada questão e a verdadeira proficiência de cada aluno.

## 🌟 Principais Funcionalidades

  * **Lançamento de Respostas:** Interface simples para inserir o gabarito correto e, em seguida, lançar as respostas de cada aluno, questão por questão.
  * **Banco de Dados em Sessão:** As respostas são salvas em tempo real (usando `st.session_state`) e exibidas em uma matriz de acertos (0 ou 1).
  * **Download de Dados:** Permite baixar a matriz de respostas completa em formato CSV.
  * **Análise TRI Simplificada:** Implementa um modelo logístico de 2 parâmetros (2PL) para estimar:
      * **Proficiência dos Alunos ($\theta$):** A "nota" real do aluno, que considera a dificuldade das questões que ele acertou.
      * **Parâmetros dos Itens (Questões):**
          * **Dificuldade (parâmetro $b$):** O quão "difícil" uma questão é.
          * **Discriminação (parâmetro $a$):** A capacidade da questão de diferenciar alunos com alta e baixa proficiência.
  * **Visualizações Interativas (Plotly):**
      * Distribuição das proficiências dos alunos (Histograma).
      * Relação entre a nota tradicional (acertos) e a nota TRI (proficiência).
      * Gráficos de Dispersão e Mapas de Calor para Dificuldade vs. Discriminação dos itens.
      * Curvas Características dos Itens (CCI) e Curvas de Informação.
  * **Relatório HTML Completo:** Gera um relatório HTML exportável que resume toda a análise, explica os conceitos para não-especialistas e inclui uma tabela comparativa (Nota de Acertos vs. Nota TRI) para todos os alunos.

## 🤖 Como Funciona: A Teoria de Resposta ao Item (TRI)

Diferente da Teoria Clássica dos Testes (TCT), que se baseia na pontuação total, a TRI é um modelo estatístico que analisa o padrão de resposta de cada aluno em cada item.

1.  **Proficiência ($\theta$):** Cada aluno possui um nível de "proficiência" (conhecimento latente) em uma escala, geralmente centrada em 0. Alunos com $\theta > 0$ estão acima da média, e alunos com $\theta < 0$ estão abaixo.
2.  **Dificuldade ($b$):** Cada questão tem um nível de dificuldade, medido na mesma escala $\theta$. Uma questão com $b = 1.5$ é difícil, exigindo um aluno com alta proficiência para ter 50% de chance de acerto.
3.  **Discriminação ($a$):** Mede o quão "bem" a questão funciona. Uma questão com alta discriminação é muito boa em separar alunos com proficiência ligeiramente abaixo da dificuldade da questão (que provavelmente errarão) daqueles com proficiência ligeiramente acima (que provavelmente acertarão).

Nesta aplicação, esses parâmetros são *estimados* de forma simplificada:

  * A **Dificuldade ($b$)** é estimada a partir do percentual de acerto total da questão.
  * A **Discriminação ($a$)** é estimada usando a correlação entre acertar aquela questão e a pontuação total no teste.
  * A **Proficiência ($\theta$)** do aluno é estimada a partir do seu percentil de acerto total.

## 🖥️ Demonstração (Recomendado)

*(Nesta seção, é altamente recomendado adicionar GIFs ou screenshots da aplicação em uso.)*

**Exemplo de GIF:**
`![Demonstração da Aplicação](link_para_seu_gif.gif)`

**Screenshots Sugeridos:**

  * Tela de Lançamento de Respostas.
  * Dashboard de Análise (com os gráficos de Dificuldade vs. Discriminação).
  * Exemplo do Relatório HTML final.

## 🛠️ Tecnologias Utilizadas

O projeto é construído inteiramente em Python, com as seguintes bibliotecas principais:

  * **Streamlit:** Para a criação do dashboard web interativo.
  * **Pandas:** Para manipulação e armazenamento dos dados (matriz de respostas).
  * **Numpy:** Para cálculos numéricos e matriciais.
  * **Plotly:** Para a geração de todos os gráficos interativos.
  * **Scipy:** Para cálculos estatísticos (distribuição normal, correlações).

## 🚀 Instalação e Uso

Para executar este projeto localmente, siga os passos abaixo.

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação Streamlit:**

    ```bash
    streamlit run 2_leitura.py
    ```

A aplicação será aberta automaticamente no seu navegador padrão.

## 🗺️ Estrutura da Aplicação

A interface é dividida em duas abas principais:

### 1\. Lançamento de Respostas

1.  **Sidebar (Configuração):** Primeiro, você deve inserir o gabarito correto da prova no campo "Gabarito Correto" na barra lateral. O número de questões e as opções de resposta são inferidos automaticamente.
2.  **Formulário de Lançamento:** Na coluna da esquerda, insira o nome ou ID do aluno e preencha os botões de rádio para cada questão. Clique em "Salvar Respostas".
3.  **Matriz de Respostas:** Na coluna da direita, a tabela é atualizada com os dados do aluno, mostrando `1` para acerto e `0` para erro. Você pode baixar esta matriz como CSV a qualquer momento.

### 2\. Análise TRI e Relatório

Esta aba só é ativada após o lançamento de um número mínimo de alunos (definido no código como 5, para estabilidade estatística).

1.  **Executar Análise:** Clique no botão "Executar Análise TRI Avançada".
2.  **Resultados:** A análise gera 5 sub-abas:
      * **🎓 Análise dos Alunos:** Mostra a distribuição das proficiências ($\theta$) e a correlação entre a pontuação total e a proficiência.
      * **📝 Análise dos Itens:** Exibe os gráficos de Dificuldade ($b$) vs. Discriminação ($a$), ajudando a identificar itens problemáticos.
      * **📈 Curvas TRI:** Permite selecionar itens específicos para visualizar suas Curvas Características (CCI) e Funções de Informação.
      * **🔍 Diagnóstico:** Apresenta métricas de qualidade do teste, como o Alfa de Cronbach (confiabilidade) e a distribuição da dificuldade.
      * **📋 Relatório HTML:** Fornece um botão para baixar o relatório completo.

## 📄 O Relatório HTML

Um dos principais produtos desta ferramenta é o relatório HTML portátil. Ele é projetado para ser compartilhado com coordenadores, professores ou outros stakeholders.

O relatório inclui:

  * Métricas de resumo do teste (Nº de alunos, Nº de itens, Confiabilidade).
  * Gráficos da distribuição de proficiência e análise de itens.
  * Glossários que explicam os termos técnicos (TRI, $\theta$, $a$, $b$) de forma simples.
  * **Diagnóstico de Itens:** Recomendações automáticas para itens que precisam de revisão (ex: baixa discriminação, muito fáceis ou muito difíceis).
  * **Tabela de Desempenho (Acertos vs TRI):** Uma tabela completa de todos os alunos, comparando sua "Nota Acertos (%)" tradicional com sua "Nota TRI ($\theta$)", ordenada pela proficiência.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
