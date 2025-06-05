    async def generate_response_suggestion(self,
                                          others_statement: str,
                                          conversation_context: List[Dict],
                                          rag_context: List[Dict] = None,
                                          situation_analysis: Dict = None) -> AIResponse:
        """
        Genera sugerencia de cómo responder cuando otros hablan
        NUEVO MÉTODO CLAVE para el enfoque proactivo
        """
        
        start_time = time.time()
        
        try:
            # Construir prompt especializado para generar respuestas
            response_template = """
Alguien acaba de decir en la reunión: "{others_statement}"

Tu trabajo es sugerir cómo responder usando el framework AEIOU de comunicación no-violenta:

A (Acknowledge): Reconoce y valida lo que dijeron
E (Express): Expresa tu perspectiva usando "yo" 
I (Identify): Propone una solución específica
O (Outcome): Define el resultado que buscas
U (Understanding): Haz una pregunta para generar diálogo

Contexto de la conversación: {context_summary}
Análisis de la situación: {situation_info}

Genera una respuesta AEIOU completa, clara y diplomática:"""

            # Construir contexto
            context_summary = self._build_context_summary(conversation_context, rag_context)
            
            # Información de situación
            situation_info = ""
            if situation_analysis:
                situation_info = f"Tipo: {situation_analysis.get('statement_type', 'neutral')}, "
                situation_info += f"Tensión: {situation_analysis.get('tension_level', 0):.0%}, "
                situation_info += f"Tono: {situation_analysis.get('emotional_tone', 'neutral')}"
            
            # Generar prompt
            prompt = response_template.format(
                others_statement=others_statement,
                context_summary=context_summary,
                situation_info=situation_info
            )
            
            # Llamar al LLM
            response_text = await self._call_ollama(prompt)
            
            # Calcular métricas
            processing_time = time.time() - start_time
            confidence = self._calculate_response_confidence(response_text, others_statement, situation_analysis)
            
            return AIResponse(
                text=response_text,
                confidence=confidence,
                processing_time=processing_time,
                trigger_type=self._map_situation_to_trigger(situation_analysis),
                aeiou_components={}
            )
            
        except Exception as e:
            logger.error(f"❌ Error generando sugerencia de respuesta: {e}")
            return self._create_response_fallback(others_statement, start_time)
    
    def _calculate_response_confidence(self, response: str, others_statement: str, situation_analysis: Dict = None) -> float:
        """
        Calcula confianza específica para sugerencias de respuesta
        """
        confidence = 0.5
        
        # Factor 1: Estructura AEIOU presente
        aeiou_indicators = ['entiendo', 'reconozco', 'yo', 'propongo', 'objetivo', 'qué']
        found_indicators = sum(1 for word in aeiou_indicators if word.lower() in response.lower())
        confidence += min(0.3, found_indicators * 0.05)
        
        # Factor 2: Relevancia al statement original
        others_words = set(others_statement.lower().split())
        response_words = set(response.lower().split())
        
        # Debe haber algo de overlap pero no demasiado (evitar repetición)
        overlap_ratio = len(others_words.intersection(response_words)) / max(len(others_words), 1)
        if 0.1 <= overlap_ratio <= 0.4:  # Sweet spot
            confidence += 0.2
        
        # Factor 3: Adecuación al tipo de situación
        if situation_analysis:
            statement_type = situation_analysis.get('statement_type', 'neutral')
            response_lower = response.lower()
            
            if statement_type == "question" and "?" in response:
                confidence += 0.1  # Bueno responder pregunta con pregunta
            
            if statement_type == "problem_statement" and any(word in response_lower for word in ['solución', 'alternativa', 'podríamos']):
                confidence += 0.15  # Bueno ofrecer soluciones a problemas
            
            if statement_type == "disagreement" and any(word in response_lower for word in ['entiendo', 'perspectiva', 'ambos']):
                confidence += 0.15  # Bueno reconocer desacuerdos diplomáticamente
        
        # Factor 4: Longitud apropiada para respuesta
        word_count = len(response.split())
        if 15 <= word_count <= 50:  # Ni muy corta ni muy larga
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _map_situation_to_trigger(self, situation_analysis: Dict = None) -> SuggestionTrigger:
        """
        Mapea análisis de situación a tipo de trigger
        """
        if not situation_analysis:
            return SuggestionTrigger.CONFLICT
        
        statement_type = situation_analysis.get('statement_type', 'neutral')
        
        mapping = {
            'question': SuggestionTrigger.DIFFICULT_QUESTION,
            'problem_statement': SuggestionTrigger.TENSION,
            'disagreement': SuggestionTrigger.DISAGREEMENT,
            'help_request': SuggestionTrigger.DIFFICULT_QUESTION
        }
        
        return mapping.get(statement_type, SuggestionTrigger.CONFLICT)
    
    def _create_response_fallback(self, others_statement: str, start_time: float) -> AIResponse:
        """
        Crea respuesta de fallback para sugerencias de respuesta
        """
        
        # Analizar tipo básico de statement para fallback apropiado
        statement_lower = others_statement.lower()
        
        if '?' in others_statement:
            # Es una pregunta
            fallback_text = "Entiendo tu pregunta (A). Yo también quiero asegurarme de darte una respuesta clara (E). ¿Podrías darme un momento para considerar esto? (I) Mi objetivo es darte una respuesta útil (O). ¿Hay algún aspecto específico que te interesa más? (U)"
        
        elif any(word in statement_lower for word in ['problema', 'error', 'no funciona']):
            # Es un problema
            fallback_text = "Entiendo que hay un desafío aquí (A). Yo también quiero encontrar una solución (E). ¿Podríamos analizar las opciones disponibles? (I) Mi objetivo es resolver esto efectivamente (O). ¿Qué has intentado hasta ahora? (U)"
        
        elif any(word in statement_lower for word in ['no estoy de acuerdo', 'pero', 'realmente no']):
            # Es desacuerdo
            fallback_text = "Entiendo que tienes una perspectiva diferente (A). Yo valoro escuchar puntos de vista diversos (E). ¿Podríamos explorar ambas perspectivas? (I) Mi objetivo es que lleguemos a una buena decisión juntos (O). ¿Qué aspectos son más importantes para ti? (U)"
        
        else:
            # Neutral/general
            fallback_text = "Entiendo tu punto (A). Yo también quiero asegurarme de que consideremos esto bien (E). ¿Podríamos discutir las implicaciones? (I) Mi objetivo es que tomemos la mejor decisión (O). ¿Qué opinas de este enfoque? (U)"
        
        return AIResponse(
            text=fallback_text,
            confidence=0.7,  # Fallbacks son bastante sólidos
            processing_time=time.time() - start_time,
            trigger_type=SuggestionTrigger.CONFLICT,
            aeiou_components={}
        )
