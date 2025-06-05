    async def _process_audio_chunk(self, audio_data: np.ndarray):
        """
        LÓGICA CORREGIDA: Sugerencias ANTES de que hables
        Cuando otros hablan → Generar sugerencia de cómo responder
        Cuando TÚ hablas → Solo contexto 
        """
        processing_start = time.time()
        
        try:
            # 1. Identificación de voz
            speaker_info = await self.speaker_id.identify_speaker(audio_data)
            is_user_voice = speaker_info['is_user_voice'] and speaker_info['confidence'] > self.config.voice_threshold
            
            # 2. Solo procesar si hay voz clara
            if speaker_info.get('confidence', 0) > 0.4:
                
                # STT para ambos casos
                transcription = await self.stt.transcribe(audio_data)
                
                if not transcription or len(transcription.strip()) < 5:
                    return
                
                if is_user_voice:
                    # TÚ estás hablando → Solo agregar al contexto
                    logger.debug(f"🎤 Tu respuesta: {transcription[:50]}...")
                    self._add_to_conversation_buffer(transcription, "user")
                    
                    # Opcional: verificar si usaste la sugerencia anterior
                    await self._analyze_user_response(transcription)
                    
                else:
                    # OTROS están hablando → GENERAR SUGERENCIA de cómo responder
                    logger.debug(f"👥 Otros hablan: {transcription[:50]}...")
                    self._add_to_conversation_buffer(transcription, "other")
                    
                    # 🎯 AQUÍ está la magia: sugerencia ANTES de que hables
                    await self._suggest_response_to_others(transcription, processing_start)
        
        except Exception as e:
            logger.error(f"❌ Error procesando audio: {e}")
    
    async def _suggest_response_to_others(self, others_speech: str, processing_start: float):
        """
        Genera sugerencia de cómo responder cuando otros hablan
        ESTE ES EL MÉTODO CLAVE
        """
        
        # Control de frecuencia (no spamear sugerencias)
        current_time = time.time()
        if current_time - self.last_suggestion_time < self.config.min_suggestion_interval:
            return
        
        try:
            # 1. Analizar lo que dijeron otros
            conversation_context = self.conversation_buffer[-10:]
            situation_analysis = self._analyze_others_statement(others_speech, conversation_context)
            
            # 2. Detectar si necesitas una sugerencia para responder
            needs_response_suggestion = self._detect_response_opportunity(
                others_speech, 
                situation_analysis
            )
            
            if not needs_response_suggestion:
                logger.debug("ℹ️ No requiere respuesta estructurada")
                return
            
            # 3. Query RAG para situaciones similares
            rag_context = []
            if self.rag:
                try:
                    # Buscar cómo responder a este tipo de situaciones
                    rag_query = f"Cómo responder cuando alguien dice: {others_speech}"
                    rag_context = await self._query_rag_context(rag_query)
                except Exception as e:
                    logger.warning(f"⚠️ Error en RAG: {e}")
            
            # 4. Generar sugerencia de respuesta AEIOU
            logger.debug("💡 Generando sugerencia de respuesta...")
            response_suggestion = await self.llm.generate_response_suggestion(
                others_statement=others_speech,
                conversation_context=conversation_context,
                rag_context=rag_context,
                situation_analysis=situation_analysis
            )
            
            # 5. Mostrar sugerencia ANTES de que hables
            await self._display_response_suggestion(
                response_suggestion, 
                others_speech, 
                situation_analysis,
                rag_context
            )
            
            # 6. Update estado
            self.last_suggestion_time = current_time
            
            processing_time = time.time() - processing_start
            logger.info(f"⚡ Sugerencia de respuesta en {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error generando sugerencia de respuesta: {e}")
    
    def _analyze_others_statement(self, others_speech: str, context: List[Dict]) -> Dict[str, Any]:
        """
        Analiza lo que dijeron otros para determinar tipo de respuesta necesaria
        """
        analysis = {
            "statement_type": "neutral",
            "emotional_tone": "neutral", 
            "requires_response": False,
            "tension_level": 0.0,
            "response_urgency": "low"
        }
        
        speech_lower = others_speech.lower()
        
        # Detectar tipo de statement
        if '?' in others_speech:
            analysis["statement_type"] = "question"
            analysis["requires_response"] = True
            analysis["response_urgency"] = "high"
        
        # Detectar tensión/problemas
        problem_keywords = [
            'problema', 'error', 'mal', 'no funciona', 'imposible', 
            'difícil', 'preocupa', 'bloqueado', 'deadline'
        ]
        
        tension_count = sum(1 for keyword in problem_keywords if keyword in speech_lower)
        analysis["tension_level"] = min(1.0, tension_count / 3)
        
        if tension_count > 0:
            analysis["statement_type"] = "problem_statement"
            analysis["requires_response"] = True
            analysis["emotional_tone"] = "concerned"
        
        # Detectar desacuerdo/conflicto
        disagreement_keywords = [
            'no estoy de acuerdo', 'pero', 'sin embargo', 'realmente no',
            'definitivamente no', 'creo que no', 'no funciona'
        ]
        
        if any(phrase in speech_lower for phrase in disagreement_keywords):
            analysis["statement_type"] = "disagreement"
            analysis["requires_response"] = True
            analysis["emotional_tone"] = "challenging"
            analysis["response_urgency"] = "high"
        
        # Detectar solicitudes de ayuda
        help_keywords = ['ayuda', 'apoyo', 'qué hacemos', 'cómo', 'ideas']
        if any(keyword in speech_lower for keyword in help_keywords):
            analysis["statement_type"] = "help_request"
            analysis["requires_response"] = True
            analysis["response_urgency"] = "medium"
        
        return analysis
    
    def _detect_response_opportunity(self, others_speech: str, analysis: Dict) -> bool:
        """
        Detecta si vale la pena generar una sugerencia de respuesta
        """
        
        # Siempre sugerir para preguntas directas
        if analysis["statement_type"] == "question":
            return True
        
        # Sugerir para problemas o desacuerdos
        if analysis["statement_type"] in ["problem_statement", "disagreement"]:
            return True
        
        # Sugerir si hay tensión alta
        if analysis["tension_level"] > 0.3:
            return True
        
        # Sugerir para solicitudes de ayuda
        if analysis["statement_type"] == "help_request":
            return True
        
        # No sugerir para statements neutros/informativos
        return False
    
    async def _display_response_suggestion(self, 
                                         suggestion, 
                                         others_speech: str,
                                         analysis: Dict,
                                         rag_context: List[Dict]):
        """
        Muestra sugerencia de cómo responder ANTES de que hables
        """
        
        print("\n" + "🔄"*80)
        print("💬 ALGUIEN ESTÁ HABLANDO:")
        print(f"   \"{others_speech}\"")
        print("\n" + "💡 CÓMO RESPONDER (AEIOU):")
        print("🔄"*80)
        print(f"📝 {suggestion.text}")
        
        # Mostrar análisis de la situación
        if analysis["tension_level"] > 0.2:
            print(f"\n🌡️ Tensión detectada: {analysis['tension_level']:.0%}")
        
        print(f"📊 Tipo: {analysis['statement_type']} | Urgencia: {analysis['response_urgency']}")
        
        if rag_context:
            print(f"📚 Basado en {len(rag_context)} situaciones similares")
        
        print("\n⏳ Esperando tu respuesta...")
        print("🔄"*80 + "\n")
    
    async def _analyze_user_response(self, user_speech: str):
        """
        Opcional: Analiza si el usuario siguió la sugerencia AEIOU
        """
        # Detectar componentes AEIOU en la respuesta
        aeiou_indicators = {
            'acknowledge': ['entiendo', 'comprendo', 'veo', 'reconozco'],
            'express': ['yo', 'mi perspectiva', 'pienso', 'siento'],
            'identify': ['podríamos', 'qué tal si', 'propongo', 'sugiero'],
            'outcome': ['objetivo', 'meta', 'busco', 'queremos lograr'],
            'understanding': ['qué piensas', 'cómo lo ves', 'tu opinión']
        }
        
        user_lower = user_speech.lower()
        components_used = []
        
        for component, indicators in aeiou_indicators.items():
            if any(indicator in user_lower for indicator in indicators):
                components_used.append(component.upper())
        
        if len(components_used) >= 3:
            logger.info(f"✅ Excelente respuesta AEIOU! Componentes: {', '.join(components_used)}")
        elif len(components_used) >= 1:
            logger.info(f"👍 Buena respuesta! Usaste: {', '.join(components_used)}")
        else:
            logger.debug("ℹ️ Respuesta directa (sin estructura AEIOU)")
