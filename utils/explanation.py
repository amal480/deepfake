def generate_explanation(classification: str, aggregated: bool):
    if classification == "AI_GENERATED":
        if aggregated:
            return (
                "Multiple segments of the audio exhibited spectral consistency and "
                "reduced micro-prosodic variation commonly associated with AI-generated speech."
            )
        else:
            return (
                "The audio exhibits spectral consistency and reduced micro-prosodic "
                "variation commonly associated with AI-generated speech."
            )
    else:
        if aggregated:
            return (
                "Across multiple segments, the audio contained natural pitch variation "
                "and timing irregularities typically present in human speech."
            )
        else:
            return (
                "The audio contains natural pitch variation and timing irregularities "
                "typically present in human speech."
            )
