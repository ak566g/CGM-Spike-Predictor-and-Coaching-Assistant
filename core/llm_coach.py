import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMCoachingAssistant:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "moonshotai/kimi-k2-instruct-0905"

    async def get_explanation(self, risk_score, glc, vel, cob, meal_type):
        """
        Analyzes the interplay between meal type and metabolic response.
        """
        prompt = f"""
        You are a Professional Metabolic Health Assistant. 
        Explain this glucose spike prediction:
        
        - Prediction: {'Risk of Spike' if risk_score > 0.5 else 'Stable'}
        - Current Level: {glc} mg/dL
        - Meal Logged: {meal_type}
        - Carbs: {cob}g
        - Rise Velocity: {vel:.2f} mg/dL/min
        
        INSTRUCTIONS:
        1. Specifically mention how the '{meal_type}' is impacting the current trend.
        2. If it is a 'Snack', highlight the risk of a quick spike.
        3. If it is 'Lunch/Dinner', explain the sustained impact of the carb load.
        4. Provide one actionable tip (e.g., 'A short walk' or 'Add protein next time').
        5. Limit to 2 concise sentences.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Objective Fallback Logic
            if risk_score > 0.5:
                return f"A glucose spike is likely due to the {cob}g of carbohydrates consumed and a current rising trend of {vel:.1f} mg/dL/min. Light physical activity may help moderate this rise."
            return "Glucose levels are currently within a stable physiological range."