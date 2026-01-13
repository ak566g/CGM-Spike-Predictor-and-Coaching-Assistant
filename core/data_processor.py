import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

class GlucoseProcessor:
    def __init__(self, threshold=180):
        self.threshold = threshold

    def parse_ohio_xml(self, file_path):
        """
        Parses OhioT1DM XML structure. 
        Aligns asynchronous meal events with 5-minute CGM intervals.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        glc_data = []
        for event in root.find('glucose_level').findall('event'):
            glc_data.append({
                'ts': pd.to_datetime(event.get('ts'), dayfirst=True),
                'glucose': float(event.get('value'))
            })
        df = pd.DataFrame(glc_data).set_index('ts').sort_index()
        
        meal_node = root.find('meal')
        if meal_node is not None:
            meals = []
            for event in meal_node.findall('event'):
                meals.append({
                    'ts': pd.to_datetime(event.get('ts'), dayfirst=True),
                    'carbs': float(event.get('carbs'))
                })
            df_meals = pd.DataFrame(meals).set_index('ts').resample('5min').sum()
            df = df.join(df_meals).fillna({'carbs': 0})
        else:
            df['carbs'] = 0

        return df.resample('5min').mean().interpolate(limit=2)

    def engineer_features(self, df, is_training=True):
        """
        Derived metabolic signals.
        - slope_15: Immediate momentum
        - slope_60: Sustained trend
        - cob_2h: Carbs on Board (cumulative impact)
        """
        df = df.copy()
        
        # Velocity calculations (mg/dL per minute)
        df['slope_15'] = df['glucose'].diff(3) / 15
        df['slope_60'] = df['glucose'].diff(12) / 60
        
        # Carbs on Board (Rolling 2-hour window)
        df['cob_2h'] = df['carbs'].rolling(window=24, min_periods=1).sum()
        
        if is_training:
            # TARGET: Max glucose in the NEXT 2 hours (24 intervals) > threshold
            df['future_max'] = df['glucose'].shift(-24).rolling(window=24).max()
            df['target'] = (df['future_max'] > self.threshold).astype(int)
            
            return df.dropna(subset=['target', 'slope_60', 'cob_2h'])
        
        return df.fillna(0)