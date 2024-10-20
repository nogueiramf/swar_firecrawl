# Swarm Firecrawl Marketing Agent

Um multi-agent usando [OpenAI Swarm](https://github.com/openai/swarm) para estratégias de marketing impulsionadas por IA usando [Firecrawl](https://firecrawl.dev) para web scraping.

## Agents

1. Interface do Usuário: Gerencia as interações dos usuários
2. Website Scraper: Extrai conteúdo limpo pronto para LLM através da API Firecrawl
3. Analista: Fornece insights de marketing
4. Ideia de Campanha: Gera conceitos de campanhas de marketing
5. Copywriter: Cria textos de marketing envolventes

## Requirements

- [Firecrawl](https://firecrawl.dev) API key
- [OpenAI](https://platform.openai.com/api-keys) API key

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure suas variáveis de ambiente em um arquivo`.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   ```

## Utilização

Execute o script principal para iniciar a demonstração interativa.:

```
python main.py
````

