import os
import yaml
from firecrawl import FirecrawlApp
from swarm import Agent
from swarm.repl import run_demo_loop
import dotenv
from openai import OpenAI

# Carrega variáveis de ambiente do .env
dotenv.load_dotenv()

# Inicializa FirecrawlApp e OpenAI
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carrega as configurações dos agentes do arquivo YAML
def load_agents_config(file_path="agents_config.yaml"):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)['agents']

agents_config = load_agents_config()

# Funções principais
def scrape_website(url):
    """Raspa conteúdo de um site usando Firecrawl."""
    return app.scrape_url(url, params={'formats': ['markdown']})

def analyze_website_content(content):
    """Analisa conteúdo do site e fornece insights de ASO."""
    analysis = generate_completion(
        "ASO specialist",
        "Analise o conteúdo e forneça insights-chave sobre ASO. "
        "Identifique os tópicos avaliados, como é feita a avaliação, "
        "o porquê da avaliação e anote métricas de referência encontradas "
        "(mínimos e máximos recomendados, % de keywords ideal, etc.).",
        content
    )
    
    # Extração de pontos específicos de otimização
    title_insights = analyze_title(content.get("title", ""))
    description_insights = analyze_description(content.get("description", ""))
    
    return {
        "analysis": analysis,
        "title_insights": title_insights,
        "description_insights": description_insights
    }

def extract_aso_metrics(content):
    """Extrai métricas de ASO do conteúdo analisado."""
    return generate_completion(
        "ASO specialist",
        "Liste as métricas de ASO relevantes, como mínimos e máximos recomendados.",
        content
    )

def generate_copy(brief):
    """Gera uma cópia de marketing com base no briefing focada em ASO."""
    return {
        "copy": generate_completion(
            "ASO copywriter",
            "Crie uma cópia otimizada para loja de aplicativos com base no briefing.",
            brief
        )
    }

def create_campaign_idea(target_audience, goals):
    """Cria uma ideia de campanha com foco em ASO com base no público-alvo e objetivos."""
    return {
        "campaign_idea": generate_completion(
            "ASO strategist",
            "Crie uma ideia de campanha de ASO inovadora.",
            f"Target Audience: {target_audience}\nGoals: {goals}"
        )
    }

def generate_completion(role, task, content, model="gpt-4o-mini", temperature=0.7):
    """Gera uma resposta usando o OpenAI."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a {role}. {task}"},
            {"role": "user", "content": content}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# Funções auxiliares de análise de ASO
def analyze_title(title):
    """Analisa o título do aplicativo com base nas práticas recomendadas de ASO."""
    # Exemplo de lógica de análise de título (pode ser aprimorado)
    if len(title) > 30:
        return "O título está muito longo. O ideal é ter menos de 30 caracteres."
    elif len(title) < 10:
        return "O título está muito curto. Considere adicionar palavras-chave relevantes."
    return "O título parece adequado em termos de comprimento."

def analyze_description(description):
    """Verifica a descrição do aplicativo, palavras-chave e chamadas para ação."""
    # Exemplo de lógica de análise de descrição
    keyword_count = description.lower().count("app")
    if keyword_count < 5:
        return "A descrição poderia ter mais palavras-chave relevantes."
    return "A descrição está bem otimizada em termos de palavras-chave."

# Funções de handoff
def handoff_to_website_scraper_agent():
    return website_scraper_agent

def handoff_to_analyst_agent():
    return analyst_agent

def handoff_to_campaign_idea_agent():
    return campaign_idea_agent

def handoff_to_copywriter_agent():
    return copywriter_agent

# Criação dos agentes com base nas configurações do YAML
user_interface_agent = Agent(
    name=agents_config['user_interface_agent']['name'],
    instructions=agents_config['user_interface_agent']['instructions'],
    functions=[scrape_website, handoff_to_website_scraper_agent],
)

website_scraper_agent = Agent(
    name=agents_config['website_scraper_agent']['name'],
    instructions=agents_config['website_scraper_agent']['instructions'],
    functions=[scrape_website, handoff_to_analyst_agent],
)

analyst_agent = Agent(
    name=agents_config['analyst_agent']['name'],
    instructions=agents_config['analyst_agent']['instructions'],
    functions=[analyze_website_content, extract_aso_metrics, handoff_to_campaign_idea_agent],
)

campaign_idea_agent = Agent(
    name=agents_config['campaign_idea_agent']['name'],
    instructions=agents_config['campaign_idea_agent']['instructions'],
    functions=[create_campaign_idea, handoff_to_copywriter_agent],
)

copywriter_agent = Agent(
    name=agents_config['copywriter_agent']['name'],
    instructions=agents_config['copywriter_agent']['instructions'],
    functions=[generate_copy],
)

if __name__ == "__main__":
    # Executa o loop de demonstração com o agente de interface
    run_demo_loop(user_interface_agent, stream=True)
