import os
from flask import Flask
from flask_cors import CORS
from fastapi import FastAPI, HTTPException
from typing import List
from db_connector import db_connector
from data_processor import N_RECOMMENDATIONS, DataProcessor # <-- Importamos el procesador

# app = FastAPI()
app = Flask(__name__)
CORS(app)  

# Inicialización del procesador de datos (se realiza al inicio del servicio)
data_processor = DataProcessor(db_connector)


@app.on_event("startup")
def startup_event():
    print("--- Iniciando servicio de recomendación ---")
    
    # 1. Cargamos los juegos (ESTO ES LO IMPORTANTE)
    # No le pasamos ratings_df aquí, o le pasamos uno vacío
    try:
        data_processor.train_model(force_retrain=False) 
        print("✅ Motor de IA cargado y listo para recibir usuarios.")
    except Exception as e:
        print(f"❌ Error al cargar los juegos: {e}")


@app.get("/")
def read_root():
    return {"message": "Motor de Recomendación (FastAPI) activo."}


@app.get("/recommend/cold-start", response_model=List[int])
def get_cold_start_games():
    """
    Devuelve un pool de juegos diversificados para que el usuario nuevo 
    comience a calificar.
    """
    try:
        # 1. Obtener el pool diversificado
        diverse_pool = db_connector.get_diverse_onboarding_pool(games_per_genre=5)
        
        if diverse_pool is None or diverse_pool.empty:
            # Si falla, devolvemos un top 20 global por seguridad
            query_fallback = 'SELECT steam_appid FROM steam_app_details ORDER BY metacritic_score DESC LIMIT 20'
            fallback = db_connector.fetch_data_to_df(query_fallback)
            return fallback['steam_appid'].tolist()

        # 2. Mezclamos los juegos para que no aparezcan agrupados por género 
        # (Así la experiencia es más variada visualmente)
        shuffled_pool = diverse_pool.sample(frac=1).reset_index(drop=True)
        
        # Devolvemos los IDs de Steam
        return shuffled_pool['steam_appid'].unique().tolist()

    except Exception as e:
        print(f"Error en Cold Start: {e}")
        raise HTTPException(status_code=500, detail="Error al generar pool inicial.")


@app.get("/recommend/{user_id}", response_model=List[int])
def get_recommendations(user_id: int):
    """
    Genera las recomendaciones basadas en contenido para el usuario.
    """
    
    ratings_df = db_connector.get_user_ratings()
    if ratings_df is None or ratings_df.empty:
        raise HTTPException(status_code=503, detail="El motor no tiene datos para generar recomendaciones.")
    
    if not data_processor.is_ready:
        raise HTTPException(status_code=500, detail="El modelo aún no está listo. Intente recargar el servicio.")
        
    # 1. Verificar Cold Start
    if not user_id in ratings_df['idUser'].unique():
        # Lógica de Cold Start: devolver los 10 IDs de juego más populares/comunes
        # Para el ejemplo:
        print(f"Usuario {user_id} sin ratings. Aplicando Cold Start.")
        return ratings_df['idSteamGame'].value_counts().head(N_RECOMMENDATIONS).index.tolist()
        
    # 2. Generar Recomendaciones
    recommendations = data_processor.generate_recommendations(user_id, ratings_df)
    
    if not recommendations:
        # Esto sucede si el usuario solo tiene valoraciones negativas (gameValue=0)
        return []

    return recommendations


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)