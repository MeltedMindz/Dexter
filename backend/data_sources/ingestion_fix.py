            try:
                # Convert to insight format
                insight = self.position_to_insight(position)
                
                # Filter out low-quality data
                if insight['data_quality_score'] < 50:
                    logger.warning(f"Skipping low-quality position {position.position_id} (score: {insight['data_quality_score']:.1f})")
                    continue
                
                # Include metadata in insight directly
                insight.update({
                    "source": "historical_position_fetcher",
                    "position_id_meta": position.position_id,
                    "quality_score_meta": insight["data_quality_score"]
                })
                
                # Store in knowledge base
                await self.knowledge_base.store_insight(
                    category=category,
                    insight=insight
                )
                
                batch_stats['insights_created'] += 1
                logger.debug(f"Stored insight for position {position.position_id}")
                batch_stats['processed'] += 1
                
            except Exception as e:
                batch_stats['errors'] += 1
                logger.error(f"Error processing position {position.position_id}: {e}")