"""
RAG Pipeline Module
Orchestrates retrieval-augmented generation for answering user queries.
Uses Euron API for chat completions.
"""

import numpy as np
import requests
from typing import List, Dict, Any, Optional
import re


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for answering questions about banking data.
    Uses vector similarity search to find relevant data and Euron API for generating answers.
    """
    
    def __init__(self, embedding_generator, vector_db, summarizer,
                 api_key: str = "euri-5af13587821689d1c1c8c50ab9fab3e6d04e4800a8489e6bb87b2df2cd408a75"):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_db: VectorDBClient instance
            summarizer: Summarizer instance
            api_key: Euron API key
        """
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.summarizer = summarizer
        self.api_key = api_key
        self.chat_url = "https://api.euron.one/api/v1/euri/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Query patterns for different types of questions
        self.query_patterns = {
            'aggregate': ['total', 'sum', 'average', 'mean', 'count', 'how many', 'how much'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'between'],
            'trend': ['trend', 'over time', 'increase', 'decrease', 'growing', 'declining'],
            'anomaly': ['unusual', 'anomaly', 'outlier', 'strange', 'unexpected'],
            'filter': ['show me', 'find', 'where', 'which', 'what are'],
            'top': ['top', 'highest', 'largest', 'most', 'best'],
            'bottom': ['bottom', 'lowest', 'smallest', 'least', 'worst']
        }
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call Euron chat API for generating responses."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "model": "gpt-4.1-nano",
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.chat_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Euron chat API: {e}")
            return None
    
    def retrieve(self, query: str, top_k: int = 10, 
                 filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant data points for a query.
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant data points with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector database
        results = self.vector_db.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def generate_answer(self, query: str, data: Any = None, 
                        context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate an answer to the user's query.
        
        Args:
            query: User's natural language query
            data: Full DataFrame (optional, for aggregate queries)
            context: Retrieved context from vector search
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        # Classify query type
        query_type = self._classify_query(query)
        
        # Retrieve relevant context if not provided
        if context is None:
            context = self.retrieve(query, top_k=15)
        
        # Generate response based on query type
        if query_type == 'aggregate' and data is not None:
            response = self._handle_aggregate_query(query, data, context)
        elif query_type == 'trend' and data is not None:
            response = self._handle_trend_query(query, data, context)
        elif query_type == 'anomaly' and data is not None:
            response = self._handle_anomaly_query(query, data, context)
        elif query_type in ['top', 'bottom'] and data is not None:
            response = self._handle_ranking_query(query, data, context, query_type)
        else:
            response = self._handle_general_query_with_llm(query, context, data)
        
        return {
            'query': query,
            'query_type': query_type,
            'answer': response['answer'],
            'confidence': response.get('confidence', 0.8),
            'relevant_data': context[:5],
            'sources_count': len(context)
        }
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        return 'general'
    
    def _handle_general_query_with_llm(self, query: str, context: List[Dict], data: Any = None) -> Dict[str, Any]:
        """Handle general queries using LLM with retrieved context."""
        
        # Build context string from retrieved data
        context_str = ""
        if context:
            context_str = "Relevant data points from the dataset:\n"
            for i, item in enumerate(context[:10], 1):
                metadata = item.get('metadata', {})
                score = item.get('score', 0)
                row_info = " | ".join([f"{k}: {v}" for k, v in metadata.items() 
                                       if k not in ['text', 'row_index'] and v is not None][:6])
                context_str += f"{i}. (relevance: {score:.2f}) {row_info}\n"
        
        # Add data summary if available
        data_summary = ""
        if data is not None:
            try:
                summary = self.summarizer.summarize_trends(data)
                data_summary = f"\nDataset overview: {summary['overview']['total_rows']} rows, {summary['overview']['total_columns']} columns.\n"
                
                # Add numeric column info
                if summary['numeric_trends']:
                    data_summary += "Numeric columns: "
                    for col, stats in list(summary['numeric_trends'].items())[:3]:
                        data_summary += f"{col} (avg: {stats['mean']:.2f}, range: {stats['min']:.2f}-{stats['max']:.2f}), "
                    data_summary = data_summary.rstrip(", ") + "\n"
            except:
                pass
        
        system_prompt = """You are a helpful data analyst assistant. Analyze the provided banking data context and answer the user's question accurately and concisely. 
        
Guidelines:
- Use the provided data context to inform your answer
- If specific data points are relevant, reference them
- Provide clear, actionable insights
- Format numbers appropriately (use $ for currency, commas for large numbers)
- If you cannot answer from the provided context, say so clearly
- Keep responses concise but informative"""
        
        user_prompt = f"""Question: {query}

{data_summary}
{context_str}

Please provide a helpful answer based on the data above."""
        
        # Call LLM
        llm_response = self._call_llm(user_prompt, system_prompt)
        
        if llm_response:
            return {'answer': llm_response, 'confidence': 0.85}
        else:
            # Fallback to basic response if LLM fails
            return self._handle_general_query(query, context)
    
    def _handle_aggregate_query(self, query: str, data: Any, 
                                context: List[Dict]) -> Dict[str, Any]:
        """Handle aggregate queries (sum, average, count, etc.)."""
        query_lower = query.lower()
        
        # Identify the target column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_col = None
        
        for col in numeric_cols:
            if col.lower() in query_lower:
                target_col = col
                break
        
        # If no specific column mentioned, look for amount/balance
        if target_col is None:
            for col in numeric_cols:
                if 'amount' in col.lower() or 'balance' in col.lower():
                    target_col = col
                    break
        
        if target_col is None and len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        
        # Calculate aggregates
        if target_col:
            col_data = data[target_col].dropna()
            
            if 'total' in query_lower or 'sum' in query_lower:
                result = col_data.sum()
                answer = f"The total {target_col} is **${result:,.2f}**."
            elif 'average' in query_lower or 'mean' in query_lower:
                result = col_data.mean()
                answer = f"The average {target_col} is **${result:,.2f}**."
            elif 'count' in query_lower or 'how many' in query_lower:
                result = len(col_data)
                answer = f"There are **{result:,}** records with {target_col} data."
            elif 'maximum' in query_lower or 'highest' in query_lower or 'max' in query_lower:
                result = col_data.max()
                answer = f"The maximum {target_col} is **${result:,.2f}**."
            elif 'minimum' in query_lower or 'lowest' in query_lower or 'min' in query_lower:
                result = col_data.min()
                answer = f"The minimum {target_col} is **${result:,.2f}**."
            else:
                # Default to summary statistics
                answer = f"**{target_col} Statistics:**\n"
                answer += f"- Count: {len(col_data):,}\n"
                answer += f"- Sum: ${col_data.sum():,.2f}\n"
                answer += f"- Average: ${col_data.mean():,.2f}\n"
                answer += f"- Min: ${col_data.min():,.2f}\n"
                answer += f"- Max: ${col_data.max():,.2f}"
        else:
            answer = f"There are **{len(data):,}** total records in the dataset."
        
        return {'answer': answer, 'confidence': 0.95}
    
    def _handle_trend_query(self, query: str, data: Any, 
                           context: List[Dict]) -> Dict[str, Any]:
        """Handle trend-related queries."""
        # Get trend analysis from summarizer
        trends = self.summarizer.summarize_trends(data)
        
        # Build answer from numeric trends
        answer_parts = ["**Trend Analysis:**\n"]
        
        for col, stats in trends['numeric_trends'].items():
            direction = stats.get('trend_direction', 'stable')
            if direction == 'increasing':
                emoji = "📈"
                desc = "increasing"
            elif direction == 'decreasing':
                emoji = "📉"
                desc = "decreasing"
            else:
                emoji = "➡️"
                desc = "stable"
            
            answer_parts.append(
                f"- **{col}**: {emoji} {desc.capitalize()} (avg: ${stats['mean']:,.2f})"
            )
        
        # Add temporal analysis if available
        if trends['temporal_trends']:
            answer_parts.append("\n**Time Period:**")
            for col, temporal in trends['temporal_trends'].items():
                date_range = temporal['date_range']
                answer_parts.append(
                    f"- Data spans from {date_range['start'][:10]} to {date_range['end'][:10]} ({temporal['span_days']} days)"
                )
        
        return {'answer': '\n'.join(answer_parts), 'confidence': 0.85}
    
    def _handle_anomaly_query(self, query: str, data: Any, 
                              context: List[Dict]) -> Dict[str, Any]:
        """Handle anomaly-related queries."""
        anomalies = self.summarizer.summarize_anomalies(data)
        
        answer_parts = ["**Anomaly Detection Results:**\n"]
        
        # Numeric outliers
        if anomalies['numeric_outliers']:
            answer_parts.append("**Outliers Found:**")
            for col, outlier_info in anomalies['numeric_outliers'].items():
                answer_parts.append(
                    f"- **{col}**: {outlier_info['count']} outliers ({outlier_info['percentage']}%) "
                    f"outside range [{outlier_info['lower_bound']:,.2f}, {outlier_info['upper_bound']:,.2f}]"
                )
                if outlier_info['sample_outliers']:
                    samples = ', '.join([f"${x:,.2f}" for x in outlier_info['sample_outliers'][:3]])
                    answer_parts.append(f"  Examples: {samples}")
        else:
            answer_parts.append("✅ No significant numeric outliers detected.")
        
        # Unusual patterns
        if anomalies['unusual_patterns']:
            answer_parts.append("\n**Unusual Patterns:**")
            for pattern in anomalies['unusual_patterns']:
                severity_emoji = "🔴" if pattern['severity'] == 'high' else "🟡" if pattern['severity'] == 'medium' else "🟢"
                answer_parts.append(f"- {severity_emoji} {pattern['description']}")
        
        return {'answer': '\n'.join(answer_parts), 'confidence': 0.9}
    
    def _handle_ranking_query(self, query: str, data: Any, 
                              context: List[Dict], query_type: str) -> Dict[str, Any]:
        """Handle top/bottom ranking queries."""
        query_lower = query.lower()
        
        # Extract number if specified
        num_match = re.search(r'\b(\d+)\b', query_lower)
        n = int(num_match.group(1)) if num_match else 5
        n = min(n, 20)  # Cap at 20
        
        # Find target column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_col = None
        
        for col in numeric_cols:
            if col.lower() in query_lower:
                target_col = col
                break
        
        if target_col is None:
            for col in numeric_cols:
                if 'amount' in col.lower() or 'balance' in col.lower():
                    target_col = col
                    break
        
        if target_col is None and len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        
        if target_col:
            ascending = query_type == 'bottom'
            top_data = data.nlargest(n, target_col) if not ascending else data.nsmallest(n, target_col)
            
            direction = "Lowest" if ascending else "Highest"
            answer_parts = [f"**{direction} {n} by {target_col}:**\n"]
            
            for idx, (_, row) in enumerate(top_data.iterrows(), 1):
                value = row[target_col]
                # Include other relevant info
                row_info = f"{idx}. ${value:,.2f}"
                
                # Add category/type if available
                for cat_col in ['category', 'type', 'description']:
                    if cat_col in row.index and row[cat_col]:
                        row_info += f" - {row[cat_col]}"
                        break
                
                answer_parts.append(row_info)
            
            return {'answer': '\n'.join(answer_parts), 'confidence': 0.95}
        
        return {'answer': "Unable to identify a numeric column for ranking.", 'confidence': 0.5}
    
    def _handle_general_query(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """Handle general queries using retrieved context (fallback without LLM)."""
        if not context:
            return {
                'answer': "I couldn't find relevant data for your query. Try rephrasing or asking about specific columns in your dataset.",
                'confidence': 0.3
            }
        
        # Build answer from context
        answer_parts = [f"**Found {len(context)} relevant records:**\n"]
        
        for i, result in enumerate(context[:5], 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            # Format metadata for display
            info_parts = []
            for key, value in metadata.items():
                if key not in ['row_index', 'text'] and value is not None:
                    if isinstance(value, (int, float)):
                        if 'amount' in key.lower() or 'balance' in key.lower():
                            info_parts.append(f"{key}: ${value:,.2f}")
                        else:
                            info_parts.append(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value}")
                    else:
                        info_parts.append(f"{key}: {value}")
            
            answer_parts.append(f"{i}. " + " | ".join(info_parts[:5]))
        
        if len(context) > 5:
            answer_parts.append(f"\n*...and {len(context) - 5} more matching records*")
        
        return {'answer': '\n'.join(answer_parts), 'confidence': 0.75}
    
    def get_suggested_questions(self, data: Any) -> List[str]:
        """Generate suggested questions based on the data schema."""
        suggestions = []
        
        # Get column info
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Generate suggestions based on columns
        if numeric_cols:
            col = numeric_cols[0]
            suggestions.append(f"What is the average {col}?")
            suggestions.append(f"Show me the top 10 highest {col}")
            suggestions.append(f"Are there any outliers in {col}?")
        
        if cat_cols:
            col = cat_cols[0]
            suggestions.append(f"What are the most common {col} values?")
        
        if datetime_cols:
            suggestions.append("What are the trends over time?")
        
        # General suggestions
        suggestions.extend([
            "Give me a summary of this data",
            "What anomalies exist in the data?",
            "How many records are in the dataset?"
        ])
        
        return suggestions[:8]
