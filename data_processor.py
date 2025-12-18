import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import re
import pickle  # Para guardar y cargar el caché
import os      # Para manejar rutas de archivos
from sklearn.preprocessing import normalize

# --- CONSTANTES DE VALORACIÓN ---
# Usaremos estas constantes para ponderar las valoraciones del usuario.
WEIGHTS = {
    0: -1.0,  # "No me interesa": Penaliza el juego/género
    1: 0.5,   # "Me gusta": Puntuación positiva suave
    2: 2.0,   # "Me encanta": Fuerte preferencia
    3: 0.8    # "Me interesa/No jugado": Puntuación positiva baja (intención)
}
N_RECOMMENDATIONS = 10

class DataProcessor:
    def __init__(self, db_connector):
        self.db = db_connector
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metadata_matrix = None
        self.game_ids = None
        self.full_metadata_df = None
        self.is_ready = False
        # Ruta del archivo de caché
        self.cache_path = "model_cache.pkl"

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def save_cache(self):
        """Guarda el estado actual del modelo en un archivo."""
        cache_data = {
            'matrix': self.metadata_matrix,
            'ids': self.game_ids,
            'df': self.full_metadata_df
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ Caché guardado en {self.cache_path}")

    def load_cache(self) -> bool:
        """Intenta cargar el caché desde el disco."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.metadata_matrix = data['matrix']
                self.game_ids = data['ids']
                self.full_metadata_df = data['df']
                self.is_ready = True
                print("🚀 Modelo cargado instantáneamente desde el caché.")
                return True
            except Exception as e:
                print(f"⚠️ Error al cargar caché: {e}")
        return False

    def _get_metadata_for_vectorization(self) -> pd.DataFrame:
        df = self.db.get_all_game_metadata()
        if df is None or df.empty:
            raise ValueError("La base de datos de juegos está vacía.")

        relevant_tokens = set(self.db.get_relevant_metadata_tokens())

        def create_context(row):
            # --- SOLUCIÓN AL ERROR DE CONCATENACIÓN ---
            # Nos aseguramos de que sean listas. Si es NaN o None, usamos []
            cats = row.get('category_description')
            gens = row.get('genre_description')
            
            list_cats = cats if isinstance(cats, list) else []
            list_gens = gens if isinstance(gens, list) else []
            
            # Ahora la concatenación es segura: lista + lista
            tags_list = list_cats + list_gens
            
            # Filtrado por relevancia
            filtered_tags = [str(t) for t in tags_list if t in relevant_tokens]
            
            tags_str = ", ".join(filtered_tags)
            name = row['name']
            desc = self._clean_text(row.get('short_description', ''))
            
            return f"Game: {name}. {name}. {name}. Tags: {tags_str}. {tags_str}. Summary: {desc}"

        df['combined_context'] = df.apply(create_context, axis=1)
        
        # Aseguramos que los valores numéricos no sean NaN antes del cálculo
        df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce').fillna(0)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        
        df['quality_boost'] = (df['metacritic_score'] / 100) + (df['rating'] / 100)
        return df

    def train_model(self, force_retrain: bool = False): # Quitamos ratings_df de aquí
        if not force_retrain and self.load_cache():
            return

        print("🧠 Analizando catálogo de juegos...")
        df = self._get_metadata_for_vectorization() # Esto lee steam_app_details
        
        # Generar los embeddings de los juegos (Global y compartido)
        sentences = df['combined_context'].tolist()
        self.metadata_matrix = self.model.encode(sentences, show_progress_bar=True)
        self.game_ids = df['steam_appid'].astype(int).tolist()
        self.full_metadata_df = df
        
        self.is_ready = True
        self.save_cache()
        print(f"✨ Entrenamiento completado con {len(self.game_ids)} juegos.")

    def _create_user_profile(self, user_ratings: pd.DataFrame) -> Optional[np.ndarray]:
        user_ratings = user_ratings[user_ratings['idSteamGame'].isin(self.game_ids)]
        if user_ratings.empty: return None

        weights_map = {0: -2.0, 1: 5, 2: 10.0, 3: 1}
        user_ratings['weight'] = user_ratings['gameValue'].map(weights_map)

        indices = [self.game_ids.index(gid) for gid in user_ratings['idSteamGame']]
        rated_game_vectors = self.metadata_matrix[indices]

        weighted_vectors = rated_game_vectors * user_ratings['weight'].values.reshape(-1, 1)
        user_profile = np.mean(weighted_vectors, axis=0)
        # NORMALIZACIÓN: Esto evita que el perfil se degrade a 0.03
        user_profile = normalize(user_profile.reshape(1, -1), norm='l2')
    
        return user_profile # Ya viene con el reshape hecho

    def _create_user_profile_bubbles(self, user_ratings: pd.DataFrame):
        """
        Crea múltiples vectores (burbujas) en lugar de uno solo.
        Separa los gustos por géneros o tipos de juego.
        """
        # Filtramos solo lo que le gusta (Me gusta=1, Me encanta=2)
        liked_games = user_ratings[user_ratings['gameValue'] > 0].copy()
        if liked_games.empty: return None

        # Obtenemos los vectores de cada juego que le gustó
        indices = [self.game_ids.index(int(gid)) for gid in liked_games['idSteamGame']]
        game_vectors = self.metadata_matrix[indices]
        
        # Aplicamos pesos (Me encanta pesa más)
        weights = liked_games['gameValue'].map({1: 1.0, 2: 2.5, 3: 0.5}).values.reshape(-1, 1)
        weighted_vectors = game_vectors * weights

        # En lugar de promediar todos, devolvemos la lista de vectores pesados
        # Cada juego calificado positivamente actúa como el centro de una burbuja
        return weighted_vectors
    
    def explain_profile(self, user_bubbles, user_ratings, final_scores, sim_scores, genre_multiplier):
        """Muestra las entrañas del cálculo para el juego #1 recomendado."""
        scored_series = pd.Series(final_scores, index=self.game_ids)
        
        played_ids = [int(i) for i in user_ratings['idSteamGame'].tolist()]
        valid_scores = scored_series.drop(index=played_ids, errors='ignore')
        
        if valid_scores.empty: return
        
        winner_id = valid_scores.idxmax()
        winner_idx = self.game_ids.index(winner_id)
        
        game_row = self.full_metadata_df.iloc[winner_idx]
        game_name = game_row['name']
        pure_sim = sim_scores[winner_idx]
        g_mult = genre_multiplier[winner_idx]
        
        reviews = game_row.get('reviews', 0)
        pop_boost = (np.log10(reviews + 1) / 6) * 0.20
        q_boost = (game_row.get('quality_boost', 0)) * 0.15
        
        # --- CORRECCIÓN AQUÍ ---
        # Accedemos a la FILA (juego) de la matriz de embeddings
        winner_vector = self.metadata_matrix[winner_idx].reshape(1, -1)
        all_sims = cosine_similarity(user_bubbles, winner_vector)
        # -----------------------
        
        best_bubble_idx = all_sims.argmax()
        
        liked_names = self.full_metadata_df[
            self.full_metadata_df['steam_appid'].isin(user_ratings[user_ratings['gameValue'] > 0]['idSteamGame'])
        ]['name'].tolist()

        print("\n" + "🚀" + "="*55)
        print(f" GANADOR: {game_name}")
        print(f" Atraído por tu burbuja de: {liked_names[best_bubble_idx]}")
        print("-" * 57)
        print(f" 📊 DESGLOSE DE PUNTUACIÓN:")
        print(f" > Similitud Semántica (IA): {pure_sim:.4f}")
        print(f" > Multiplicador Género:     x{g_mult:.1f}")
        print(f" > Bonus Calidad:           +{q_boost:.4f}")
        print(f" > Bonus Popularidad:       +{pop_boost:.4f}")
        print(f" > SCORE FINAL:             {valid_scores[winner_id]:.4f}")
        print("="*57 + "\n")

    def generate_recommendations(self, user_id: int, ratings_df: pd.DataFrame) -> List[int]:
        if not self.is_ready: return []
        
        user_ratings = ratings_df[ratings_df['idUser'] == user_id]
        user_bubbles = self._create_user_profile_bubbles(user_ratings)
        
        if user_bubbles is None: return []

        # 1. SIMILITUD POR BURBUJAS (Contrast Stretching)
        all_sims = cosine_similarity(user_bubbles, self.metadata_matrix)
        sim_scores = np.max(all_sims, axis=0) ** 2 

        # 2. FILTRO DE GÉNERO DURO
        # Extraer géneros de los juegos que le gustan al usuario
        liked_ids = user_ratings[user_ratings['gameValue'] > 0]['idSteamGame'].tolist()
        user_genres = set()
        for gid in liked_ids:
            game_data = self.full_metadata_df[self.full_metadata_df['steam_appid'] == gid]
            if not game_data.empty:
                g = game_data.iloc[0].get('genre_description', [])
                if isinstance(g, list): user_genres.update(g)

        # Aplicar penalización si el juego no comparte ningún género con lo que te gusta
        genre_multiplier = []
        for _, row in self.full_metadata_df.iterrows():
            g_list = row.get('genre_description', [])
            game_genres = set(g_list if isinstance(g_list, list) else [])
            
            # Contamos cuántos géneros coinciden
            intersection_size = len(user_genres.intersection(game_genres))
            
            if intersection_size >= 2:
                multiplier = 1.3  # ¡PREMIO! Coincide mucho con tu perfil
            elif intersection_size == 1:
                multiplier = 1.0  # Normal
            else:
                multiplier = 0.2  # PENALIZACIÓN: No es tu estilo
                
            genre_multiplier.append(multiplier)
        
        genre_multiplier = np.array(genre_multiplier)

        # 3. CÁLCULO DE SCORE CON POPULARIDAD Y CALIDAD
        reviews = self.full_metadata_df['reviews'].fillna(0).values
        pop_boost = np.log10(reviews + 1) / 6
        quality = self.full_metadata_df['quality_boost'].values
        
        # Fórmula: (Similitud * Género) + Bonus
        final_scores = (sim_scores * genre_multiplier) * (1 + (quality * 0.05) + (pop_boost * 0.05))
        
        # 4. PENALIZACIÓN DE SAGAS (HERMANOS)
        played_names = self.full_metadata_df[
            self.full_metadata_df['steam_appid'].isin(user_ratings['idSteamGame'])
        ]['name'].tolist()
        
        scored_series = pd.Series(final_scores, index=self.game_ids)
        
        for played_name in played_names:
            words = played_name.split()[:2]
            if len(words) >= 2:
                pattern = " ".join(words)
                mask = self.full_metadata_df['name'].str.contains(pattern, case=False, regex=False)
                # Bajamos el score a los juegos de la misma franquicia para dejar entrar a los "Primos"
                scored_series[mask.values] *= 0.6 

        # --- LLAMADA AL DEBUG ---
        self.explain_profile(user_bubbles, user_ratings, scored_series.values, sim_scores, genre_multiplier)

        # 5. EXCLUSIÓN Y RESULTADO FINAL
        played_ids = [int(i) for i in user_ratings['idSteamGame'].tolist()]
        final_recommendations = scored_series.drop(index=played_ids, errors='ignore')

        return final_recommendations.sort_values(ascending=False).head(10).index.tolist()