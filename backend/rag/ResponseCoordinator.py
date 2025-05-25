from typing import Dict
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseCoordinator:
    def __init__(self, config,domain_agent, brand_agent, table_agent):
        self.config = config
        self.domain_agent = domain_agent
        self.brand_agent = brand_agent
        self.table_agent = table_agent
        
        self.synth_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
You are a professional architecture consultant synthesizing information from various sources.
Answer the user query below by combining insights from domain expertise, brand-specific knowledge, and data insights.
Ensure the response is clear, professional, and reflects deep understanding.

Query: {query}

Available Information:
{context}

Instructions:
- Provide a comprehensive answer that integrates all available information
- Highlight key insights and data points
- Make specific recommendations where appropriate
- Maintain a professional and authoritative tone
- Mention some key actionable insights that can be taken according to the domain expertise

Professional Answer:"""
        )
        
        self.llm = OpenAI(temperature=0,api_key=self.config.openai_api_key)
        self.synth_chain = self.synth_prompt | self.llm
    
    def respond(self, query: str) -> str:
        """Generate coordinated response from all agents"""
        logger.info(f"Processing query: {query}")
        
        responses = {}
        
        # Get domain insights
        if self.domain_agent:
            try:
                responses['domain'] = self.domain_agent.invoke(query)
                logger.info("Domain agent response obtained")
            except Exception as e:
                logger.error(f"Domain agent error: {e}")
                responses['domain'] = "Domain information unavailable"
        
        # Get brand insights
        if self.brand_agent:
            try:
                responses['brand'] = self.brand_agent.invoke(query)
                logger.info("Brand agent response obtained")
            except Exception as e:
                logger.error(f"Brand agent error: {e}")
                responses['brand'] = "Brand information unavailable"
        
        # Get data insights
        if self.table_agent:
            try:
                responses['data'] = self.table_agent.invoke(query)
                logger.info("Table agent response obtained")
            except Exception as e:
                logger.error(f"Table agent error: {e}")
                responses['data'] = "Data analysis unavailable"
        
        # Synthesize responses
        context = self._format_context(responses)
        
        try:
            input_data = {
            'query': query,
            'context': context,
            }

            final_response = self.synth_chain.invoke(input=input_data)
            return final_response
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._fallback_response(query, responses)
    
    def _format_context(self, responses: Dict[str, str]) -> str:
        """Format responses into context string"""
        context_parts = []
        
        if 'domain' in responses:
            context_parts.append(f"DOMAIN EXPERTISE:\n{responses['domain']}\n")
        
        if 'brand' in responses:
            context_parts.append(f"BRAND-SPECIFIC KNOWLEDGE:\n{responses['brand']}\n")
        
        if 'data' in responses:
            context_parts.append(f"DATA ANALYSIS:\n{responses['data']}\n")
        
        return "\n".join(context_parts)
    
    def _fallback_response(self, query: str, responses: Dict[str, str]) -> str:
        """Provide fallback response if synthesis fails"""
        response_parts = [f"Query: {query}\n"]
        
        for source, content in responses.items():
            response_parts.append(f"{source.upper()} INSIGHTS:\n{content}\n")
        
        return "\n".join(response_parts)
