    async def _process_audio_chunk(self, audio_data: np.ndarray):
        """
        Procesamiento lean mejorado: STT para todos, sugerencias solo para usuario
        SOLUCIÓN: Contexto completo + trigger selectivo
        """
        processing_start = time.time()
        
        try:
            # 1. Identificación de voz (siempre)
            speaker_info = await self.speaker_id.identify_speaker(audio_data)
            is_user_voice = speaker_info['is_user_voice'] and speaker_info['confidence'] > self.config.voice_threshold
            
            # 2. STT para TODOS los hablantes (optimizado por tipo)
            if speaker_info.get('confidence', 0) > 0.4:  # Solo si hay voz detectada
                
                if is_user_voice:
                    # Tu voz → STT de alta calidad
                    logger.debug(f"🎯 Tu voz detectada (confianza: {speaker_info['confidence']:.2f})")
                    transcription = await self.stt.transcribe(audio_data)
                    speaker_type = "user"
                    
                else:
                    # Otra voz → STT rápido pero suficiente para contexto
                    logger.debug(f"👥 Otra voz detectada (confianza: {speaker_info['confidence']:.2f})")
                    transcription = await self._transcribe_for_context(audio_data)
                    speaker_type = "other"
                
                # 3. Agregar SIEMPRE al buffer de conversación (contexto completo)
                if transcription and len(transcription.strip()) > 5:
                    self._add_to_conversation_buffer(transcription, speaker_type)
                    
                    # 4. Generar sugerencia SOLO cuando hablas TÚ
                    if is_user_voice and len(transcription.strip()) > 10:
                        await self._analyze_and_suggest_with_context(
                            user_speech=transcription,
                            processing_start=processing_start
                        )
            
            else:
                # Silencio o audio muy bajo
                logger.debug("🔇 No se detectó voz clara")
        
        except Exception as e:
            logger.error(f"❌ Error procesando audio: {e}")
    
    async def _transcribe_for_context(self, audio_data: np.ndarray) -> Optional[str]:
        """
        STT optimizado para contexto (otros hablantes)
        Más rápido pero suficiente para entender la conversación
        """
        try:
            # Usar configuración más rápida para otros
            old_options = self.stt.transcribe_options.copy()
            
            # Configuración lean para contexto
            self.stt.transcribe_options.update({
                "temperature": 0.0,  # Más determinístico
                "no_speech_threshold": 0.8,  # Más estricto
                "compression_ratio_threshold": 3.0,  # Menos exigente
                "condition_on_previous_text": False  # Más rápido
            })
            
            transcription = await self.stt.transcribe(audio_data)
            
            # Restaurar configuración original
            self.stt.transcribe_options = old_options
            
            return transcription
            
        except Exception as e:
            logger.warning(f"⚠️ Error en STT para contexto: {e}")
            return None
    
    async def _analyze_and_suggest_with_context(self, user_speech: str, processing_start: float):
        """
        Análisis mejorado usando contexto completo de la conversación
        """
        
        # Control de frecuencia de sugerencias
        current_time = time.time()
        if current_time - self.last_suggestion_time < self.config.min_suggestion_interval:
            return
        
        try:
            # 1. Analizar contexto completo para detectar tensión
            conversation_context = self.conversation_buffer[-10:]  # Últimas 10 intervenciones
            situation_analysis = self._analyze_conversation_situation(conversation_context, user_speech)
            
            # 2. Detectar si necesita sugerencia basado en contexto completo
            needs_suggestion = await self._detect_suggestion_trigger_with_context(
                user_speech, 
                conversation_context,
                situation_analysis
            )
            
            if not needs_suggestion:
                logger.debug("ℹ️ No se detectó necesidad de sugerencia")
                return
            
            # 3. Query RAG con contexto enriquecido
            rag_context = []
            if self.rag:
                try:
                    logger.debug("🔍 Buscando contexto en base de conocimiento...")
                    # Usar tanto tu speech como el contexto para query RAG
                    context_summary = self._build_context_summary_for_rag(conversation_context, user_speech)
                    rag_context = await self._query_rag_context(context_summary)
                except Exception as e:
                    logger.warning(f"⚠️ Error en RAG: {e}")
            
            # 4. Generar sugerencia AEIOU con contexto completo
            logger.debug("💡 Generando sugerencia AEIOU con contexto...")
            suggestion = await self.llm.generate_aeiou_suggestion(
                user_speech=user_speech,
                conversation_context=conversation_context,  # Contexto completo
                rag_context=rag_context
            )
            
            # 5. Mostrar sugerencia enriquecida
            await self._display_suggestion_with_context(suggestion, rag_context, situation_analysis)
            
            # 6. Update estado
            self.last_suggestion_time = current_time
            
            processing_time = time.time() - processing_start
            logger.info(f"⚡ Sugerencia contextual generada en {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error generando sugerencia: {e}")
    
    def _analyze_conversation_situation(self, conversation_context: List[Dict], user_speech: str) -> Dict[str, Any]:
        """
        Analiza la situación completa de la conversación
        """
        analysis = {
            "tension_level": 0.0,
            "participants_count": len(set(msg['speaker'] for msg in conversation_context)),
            "recent_topics": [],
            "emotional_indicators": [],
            "conflict_signs": []
        }
        
        # Analizar tensión en la conversación
        tension_keywords = ['problema', 'error', 'mal', 'no funciona', 'imposible', 'difícil']
        conflict_indicators = ['no estoy de acuerdo', 'pero', 'sin embargo', 'realmente', 'definitivamente no']
        
        recent_messages = [msg['text'].lower() for msg in conversation_context[-5:]]
        all_text = ' '.join(recent_messages + [user_speech.lower()])
        
        # Calcular nivel de tensión
        tension_count = sum(1 for keyword in tension_keywords if keyword in all_text)
        conflict_count = sum(1 for indicator in conflict_indicators if indicator in all_text)
        
        analysis['tension_level'] = min(1.0, (tension_count + conflict_count * 2) / 10)
        analysis['conflict_signs'] = [word for word in conflict_indicators if word in all_text]
        
        # Detectar patrones de conversación
        if len(conversation_context) >= 3:
            last_three = conversation_context[-3:]
            speakers = [msg['speaker'] for msg in last_three]
            
            if len(set(speakers)) >= 2:  # Intercambio entre múltiples personas
                analysis['interaction_pattern'] = 'active_discussion'
            else:
                analysis['interaction_pattern'] = 'monologue'
        
        return analysis
    
    async def _detect_suggestion_trigger_with_context(self, 
                                                    user_speech: str, 
                                                    conversation_context: List[Dict],
                                                    situation_analysis: Dict) -> bool:
        """
        Detección mejorada usando contexto completo
        """
        
        # Trigger basado en tu speech (original)
        user_trigger = await self._detect_suggestion_trigger(user_speech)
        
        # Trigger basado en nivel de tensión en la conversación
        tension_trigger = situation_analysis['tension_level'] > 0.3
        
        # Trigger basado en patrones de conflicto
        conflict_trigger = len(situation_analysis['conflict_signs']) > 0
        
        # Trigger basado en cambio de dinámicas
        if len(conversation_context) >= 5:
            recent_speakers = [msg['speaker'] for msg in conversation_context[-5:]]
            if 'other' in recent_speakers and recent_speakers[-1] == 'user':
                # Acabas de responder después de que otros hablaran
                response_after_others = True
            else:
                response_after_others = False
        else:
            response_after_others = False
        
        final_trigger = user_trigger or tension_trigger or conflict_trigger or response_after_others
        
        if final_trigger:
            trigger_reasons = []
            if user_trigger: trigger_reasons.append("user_speech")
            if tension_trigger: trigger_reasons.append("conversation_tension")
            if conflict_trigger: trigger_reasons.append("conflict_detected")
            if response_after_others: trigger_reasons.append("response_after_discussion")
            
            logger.debug(f"🎯 Trigger detectado: {', '.join(trigger_reasons)}")
        
        return final_trigger
    
    def _build_context_summary_for_rag(self, conversation_context: List[Dict], user_speech: str) -> str:
        """
        Construye resumen inteligente para query RAG
        """
        if not conversation_context:
            return user_speech
        
        # Incluir últimas intervenciones de otros + tu speech actual
        other_statements = [
            msg['text'] for msg in conversation_context[-5:] 
            if msg['speaker'] == 'other' and len(msg['text']) > 10
        ]
        
        if other_statements:
            context_summary = f"Contexto: {' '.join(other_statements[-2:])}. Tu respuesta: {user_speech}"
        else:
            context_summary = user_speech
        
        return context_summary
    
    async def _display_suggestion_with_context(self, suggestion, rag_context: List[Dict], situation_analysis: Dict):
        """
        Display mejorado con información contextual
        """
        
        print("\n" + "="*80)
        print("💡 SUGERENCIA AEIOU CONTEXTUAL")
        print("="*80)
        print(f"📝 {suggestion.text}")
        
        # Mostrar análisis de situación
        if situation_analysis['tension_level'] > 0.2:
            print(f"\n🌡️ Nivel de tensión detectado: {situation_analysis['tension_level']:.1%}")
        
        if situation_analysis['conflict_signs']:
            print(f"⚠️ Señales de conflicto: {', '.join(situation_analysis['conflict_signs'])}")
        
        # Info RAG
        if rag_context:
            print(f"\n📊 Basado en {len(rag_context)} situaciones similares")
            print(f"🎯 Confianza: {suggestion.confidence:.0%}")
        
        # Contexto conversacional
        participants = situation_analysis.get('participants_count', 1)
        if participants > 1:
            print(f"👥 Conversación con {participants} participantes")
        
        print("="*80 + "\n")
    
    def _add_to_conversation_buffer(self, text: str, speaker_type: str):
        """
        Buffer de conversación mejorado con metadatos
        """
        self.conversation_buffer.append({
            "text": text,
            "speaker": speaker_type,
            "timestamp": time.time(),
            "length": len(text),
            "word_count": len(text.split())
        })
        
        # Limitar tamaño del buffer de manera inteligente
        if len(self.conversation_buffer) > self.max_buffer_size:
            # Mantener mensajes importantes (largos o recientes)
            important_messages = [
                msg for msg in self.conversation_buffer 
                if msg['word_count'] > 5 or (time.time() - msg['timestamp']) < 60
            ]
            
            if len(important_messages) > self.max_buffer_size:
                # Si aún hay muchos, mantener los más recientes
                self.conversation_buffer = important_messages[-self.max_buffer_size:]
            else:
                self.conversation_buffer = important_messages
