import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, List

# Reemplaza con tus credenciales reales
DATABASE_URL = "postgresql://postgres:1234@localhost:5432/RecGameDB"

class DBConnector:
    """Clase para manejar la conexión y extracción de datos desde PostgreSQL."""

    def __init__(self, db_url: str = DATABASE_URL):
        try:
            # Crear el motor de conexión SQLAlchemy
            self.engine = create_engine(db_url)
        except Exception as e:
            print(f"Error al conectar con la base de datos: {e}")
            self.engine = None

    def fetch_data_to_df(self, query:str) -> Optional[pd.DataFrame]:
        """Ejecuta una consulta SQL y devuelve los resultados como un DataFrame"""
        if not self.engine:
            return None
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
            return None
        
    def get_user_ratings(self) -> Optional[pd.DataFrame]:
        """Obtiene todos los registros de valoración de juegos."""
        query = "SELECT \"idUser\", \"idSteamGame\", \"gameValue\" FROM \"Info_games\";"
        return self.fetch_data_to_df(query)

    def get_relevant_metadata_tokens(self) -> List[str]:
        """Obtiene solo los tokens de metadatos marcados como relevantes por el administrador."""
        query = "SELECT \"tokenName\" FROM \"MetadataControl\" WHERE \"isRelevant\" = TRUE;"
        df = self.fetch_data_to_df(query)
        if df is not None:
            return df['tokenName'].tolist()
        return []
    
    def get_all_game_metadata(self) -> Optional[pd.DataFrame]:
        # 1. Traer detalles, descripciones y métricas de calidad
        query = """
        SELECT 
            d.steam_appid, d.name, d.developers, d.publishers, 
            d.metacritic_score, d.short_description, d.detailed_description,
            s.rating, s.peak_players, s.reviews
        FROM "steam_app_details" d
        LEFT JOIN "steamids" s ON d.steam_appid = s.app_id
        WHERE d.type = 'game';
        """
        details_df = self.fetch_data_to_df(query)

        if details_df is None or details_df.empty:
            return None
        
        # 2. Unir Categorías y Géneros (como hicimos antes)
        categories_df = self.fetch_data_to_df("SELECT steam_appid, category_description FROM steam_categories")
        genres_df = self.fetch_data_to_df("SELECT steam_appid, genre_description FROM steam_genres")

        if categories_df is not None:
             cat_grouped = categories_df.groupby('steam_appid')['category_description'].apply(list).reset_index()
             details_df = pd.merge(details_df, cat_grouped, on='steam_appid', how='left')
        
        if genres_df is not None:
             gen_grouped = genres_df.groupby('steam_appid')['genre_description'].apply(list).reset_index()
             details_df = pd.merge(details_df, gen_grouped, on='steam_appid', how='left')

        # Limpieza inicial de nulos numéricos
        details_df['metacritic_score'] = details_df['metacritic_score'].fillna(0)
        details_df['rating'] = details_df['rating'].fillna(0)
        details_df['peak_players'] = details_df['peak_players'].fillna(0)

        return details_df
    
    def get_diverse_onboarding_pool(self, games_per_genre: int = 5) -> Optional[pd.DataFrame]:
        """
        Obtiene los mejores juegos de cada género relevante para usuarios nuevos.
        """
        query = f"""
        WITH RankedGames AS (
            SELECT 
                d.steam_appid, 
                d.name, 
                g.genre_description,
                d.metacritic_score,
                s.rating,
                ROW_NUMBER() OVER(
                    PARTITION BY g.genre_description 
                    ORDER BY d.metacritic_score DESC, s.rating DESC
                ) as rank
            FROM "steam_app_details" d
            JOIN "steam_genres" g ON d.steam_appid = g.steam_appid
            JOIN "steamids" s ON d.steam_appid = s.app_id
            JOIN "MetadataControl" mc ON g.genre_description = mc."tokenName"
            WHERE d.type = 'game' 
              AND mc."isRelevant" = TRUE
              AND d.metacritic_score > 70
        )
        SELECT steam_appid, name, genre_description
        FROM RankedGames
        WHERE rank <= {games_per_genre}
        ORDER BY genre_description, metacritic_score DESC;
        """
        return self.fetch_data_to_df(query)

# Inicialización (ejemplo de uso)
db_connector = DBConnector()