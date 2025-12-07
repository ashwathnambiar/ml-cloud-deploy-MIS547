# edcpvsw3.py
# -*- coding: utf-8 -*-
"""
Import-safe ML analysis script converted from the notebook.
Top-level only defines functions and helpers.
Runtime actions happen only under `if __name__ == "__main__":`.
"""

# --- Imports ---
import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ----------------------
# Data cleaning function
# ----------------------
def clean_data(df_game, english_only=True, winsorize=True):
    df = df_game.copy()

    # Language Filter English (if present)
    if english_only and 'author_language' in df.columns:
        df = df[df['author_language'].str.lower().str.contains('en', na=False)]

    # Normalize boolean-like columns
    bool_like = {
        'voted_up', 'steam_purchase', 'received_for_free',
        'primarily_steam_deck', 'written_during_early_access'
    }
    for col in bool_like:
        if col in df.columns:
            df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # Numeric coercion
    numeric_cols = [
        'author_num_games_owned', 'author_num_reviews',
        'author_playtime_forever', 'author_playtime_last_two_weeks',
        'author_playtime_at_review', 'author_deck_playtime_at_review',
        'votes_up', 'votes_funny', 'comment_count',
        'weighted_vote_score'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['author_playtime_last_two_weeks', 'author_playtime_at_review', 'author_deck_playtime_at_review']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Sentiment encoding (safe)
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['bert_sentiment_num'] = df.get('en_bert_label_from_partial', pd.Series(index=df.index)).map(sentiment_map).fillna(0)
    df['roberta_sentiment_num'] = df.get('en_roberta_label_from_partial', pd.Series(index=df.index)).map(sentiment_map).fillna(0)
    df['avg_sentiment'] = (pd.to_numeric(df['bert_sentiment_num'], errors='coerce').fillna(0) +
                           pd.to_numeric(df['roberta_sentiment_num'], errors='coerce').fillna(0)) / 2

    # Try to parse timestamps if present
    for col in ['timestamp_created', 'timestamp_updated', 'timestamp_dev_responded']:
        if col in df.columns:
            try:
                df[f'{col}_dt'] = pd.to_datetime(df[col], unit='s', errors='coerce')
            except Exception:
                df[f'{col}_dt'] = pd.to_datetime(df[col], errors='coerce')

    # De-duplicate
    if 'recommendationid' in df.columns:
        try:
            df = df.sort_values(by=['timestamp_updated'] if 'timestamp_updated' in df.columns else None, ascending=False)
        except Exception:
            pass
        df = df.drop_duplicates(subset=['recommendationid'], keep='first')
    elif {'author_steamid', 'review'}.issubset(df.columns):
        df = df.drop_duplicates(subset=['author_steamid', 'review'], keep='last')

    # Clip negatives
    for col in ['author_playtime_forever', 'author_playtime_last_two_weeks',
                'author_playtime_at_review', 'author_deck_playtime_at_review',
                'author_num_reviews', 'author_num_games_owned',
                'votes_up', 'votes_funny', 'comment_count',
                'community_participation', 'weighted_vote_score']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Winsorize heavy tails (optional)
    if winsorize:
        heavy = [c for c in ['author_playtime_forever', 'author_playtime_last_two_weeks',
                             'author_num_games_owned', 'author_num_reviews', 'weighted_vote_score'] if c in df.columns]
        for c in heavy:
            try:
                lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
                df[c] = df[c].clip(lower=lo, upper=hi)
            except Exception:
                pass

    # Ensure IDs as strings
    if 'author_steamid' in df.columns:
        df['author_steamid'] = df['author_steamid'].astype(str)

    # Fill remaining NAs used later
    for c in ['author_num_games_owned', 'author_num_reviews', 'author_playtime_forever',
              'author_playtime_last_two_weeks', 'author_playtime_at_review',
              'author_deck_playtime_at_review', 'weighted_vote_score', 'avg_sentiment']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df

# ----------------------
# Feature engineering
# ----------------------
def create_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()

    # Ensure required columns exist
    for c in ['votes_up','votes_funny','comment_count','author_playtime_forever',
              'author_playtime_last_two_weeks','author_playtime_at_review','author_num_games_owned',
              'received_for_free','weighted_vote_score']:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # community participation
    df['community_participation'] = (df['votes_up'] + df['votes_funny'] + df['comment_count']).astype(float)

    # recent_activity_ratio
    df['recent_activity_ratio'] = np.divide(
        df['author_playtime_last_two_weeks'], df['author_playtime_forever'],
        out=np.zeros_like(df['author_playtime_last_two_weeks'], dtype=float),
        where=(df['author_playtime_forever'] > 0)
    )

    # purchase_ratio
    df['purchase_ratio'] = np.where(df['received_for_free'] == 1, 0.0, 1.0)

    # helpfulness_ratio
    denom = df['votes_up'] + df['votes_funny'] + df['comment_count'] + 1.0
    df['helpfulness_ratio'] = df['votes_up'] / denom

    # recent_playtime_share
    base_play = df['author_playtime_at_review'].where(df['author_playtime_at_review'] > 0,
                                                      df['author_playtime_forever']).replace(0, np.nan)
    df['recent_playtime_share'] = np.divide(
        df['author_playtime_last_two_weeks'], base_play,
        out=np.zeros_like(df['author_playtime_last_two_weeks'], dtype=float),
        where=base_play.notna()
    )

    # sentiment agreement
    if {'bert_sentiment_num','roberta_sentiment_num'}.issubset(df.columns):
        df['sentiment_agreement'] = (df['bert_sentiment_num'] == df['roberta_sentiment_num']).astype(int)
    else:
        df['sentiment_agreement'] = 0

    return df

# ----------------------
# Helper clustering functions
# ----------------------
def _fit_kmeans_with_guard(X, desired_k, label_for_print):
    if isinstance(X, pd.DataFrame):
        features = X.columns.tolist()
        X = X.fillna(0).values
    else:
        features = None
    n = X.shape[0]

    if n == 0:
        return dict(labels=None, k=None, silhouette=None, features=features, skipped=True, skip_reason="No samples")

    k = 1 if n == 1 else min(desired_k, n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k == 1:
        labels = np.zeros(n, dtype=int)
        sil = None
        return dict(labels=labels, k=1, silhouette=sil, features=features, skipped=False, skip_reason=None)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else None
    return dict(labels=labels, k=k, silhouette=sil, features=features, skipped=False, skip_reason=None)

# ----------------------
# Clustering feature functions
# ----------------------
def cluster_engagement(df_game):
    df = df_game.copy()
    df['log_games_owned'] = np.log1p(df['author_num_games_owned'].fillna(0))
    df['log_num_reviews'] = np.log1p(df['author_num_reviews'].fillna(0))
    play = df['author_playtime_forever'].fillna(0).astype(float)
    mu, sigma = play.mean(), play.std(ddof=0)
    df['playtime_z'] = (play - mu) / (sigma if sigma > 0 else 1.0)
    features = ['log_games_owned', 'log_num_reviews', 'playtime_z', 'community_participation']
    X = df[features].fillna(0)
    results = _fit_kmeans_with_guard(X, desired_k=4, label_for_print="engagement")
    return {
        'labels': results['labels'],
        'silhouette': results['silhouette'],
        'features': features,
        'skipped': results['skipped'],
        'skip_reason': results['skip_reason'],
        'k': results['k']
    }

def cluster_reviewer_sentiment(df_game):
    df = df_game.copy()
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['bert_sentiment_num'] = df.get('en_bert_label_from_partial', pd.Series(index=df.index)).map(sentiment_map).fillna(0)
    df['roberta_sentiment_num'] = df.get('en_roberta_label_from_partial', pd.Series(index=df.index)).map(sentiment_map).fillna(0)
    df['avg_sentiment'] = (df['bert_sentiment_num'] + df['roberta_sentiment_num']) / 2

    player_stats = df.groupby('author_steamid', as_index=False).agg({
        'avg_sentiment': 'mean',
        'author_num_reviews': 'first',
        'weighted_vote_score': 'mean'
    }).rename(columns={'author_num_reviews': 'total_reviews', 'weighted_vote_score': 'avg_vote_score'})

    features = ['avg_sentiment','total_reviews','avg_vote_score']
    X = player_stats[features].fillna(0)
    results = _fit_kmeans_with_guard(X, desired_k=3, label_for_print="reviewer")
    return {
        'labels': results['labels'],
        'silhouette': results['silhouette'],
        'features': features,
        'player_stats': player_stats,
        'skipped': results['skipped'],
        'skip_reason': results['skip_reason'],
        'k': results['k']
    }

def cluster_playtime_patterns(df_game):
    df = df_game.copy()
    play = df['author_playtime_forever'].fillna(0).astype(float)
    mu, sigma = play.mean(), play.std(ddof=0)
    df['playtime_z'] = (play - mu) / (sigma if sigma > 0 else 1.0)
    df['playtime_pct'] = play.rank(method='average', pct=True)

    base_play = df['author_playtime_at_review'].where(
        df['author_playtime_at_review'].fillna(0) > 0,
        df['author_playtime_forever'].fillna(0)
    ).replace(0, np.nan)
    df['recent_playtime_share'] = np.divide(
        df['author_playtime_last_two_weeks'].fillna(0),
        base_play,
        out=np.zeros_like(df['author_playtime_last_two_weeks'].fillna(0), dtype=float),
        where=base_play.notna()
    )

    features = [
        'author_playtime_forever', 'author_playtime_last_two_weeks',
        'recent_activity_ratio','playtime_z','playtime_pct','recent_playtime_share'
    ]
    X = df[features].fillna(0)
    results = _fit_kmeans_with_guard(X, desired_k=4, label_for_print="playtime")
    return {
        'labels': results['labels'],
        'silhouette': results['silhouette'],
        'features': features,
        'skipped': results['skipped'],
        'skip_reason': results['skip_reason'],
        'k': results['k']
    }

def cluster_composite_profile(df_game):
    df = df_game.copy()
    denom = (df['votes_up'].fillna(0) + df['votes_funny'].fillna(0) + df['comment_count'].fillna(0) + 1.0)
    df['helpfulness_ratio'] = df['votes_up'].fillna(0) / denom
    df['log_games_owned'] = np.log1p(df['author_num_games_owned'].fillna(0))
    df['log_num_reviews'] = np.log1p(df['author_num_reviews'].fillna(0))

    player_composite = df.groupby('author_steamid', as_index=False).agg({
        'log_games_owned':'first',
        'log_num_reviews':'first',
        'author_playtime_forever':'first',
        'community_participation':'sum',
        'avg_sentiment':'mean',
        'helpfulness_ratio':'mean',
        'voted_up':'mean',
        'steam_purchase':'mean',
        'received_for_free':'mean',
        'recent_activity_ratio':'mean'
    })

    features = [
        'log_games_owned','log_num_reviews','author_playtime_forever',
        'community_participation','avg_sentiment','helpfulness_ratio',
        'voted_up','steam_purchase','received_for_free','recent_activity_ratio'
    ]
    X = player_composite[features].fillna(0)
    results = _fit_kmeans_with_guard(X, desired_k=5, label_for_print="composite")
    return {
        'labels': results['labels'],
        'silhouette': results['silhouette'],
        'features': features,
        'player_composite': player_composite,
        'skipped': results['skipped'],
        'skip_reason': results['skip_reason'],
        'k': results['k']
    }

# ----------------------
# Combine runner
# ----------------------
def perform_clustering_analysis(df_game, game_name):
    results = {
        'engagement': cluster_engagement(df_game),
        'reviewer': cluster_reviewer_sentiment(df_game),
        'playtime': cluster_playtime_patterns(df_game),
        'composite': cluster_composite_profile(df_game),
        'game_name': game_name,
        'original_df': df_game
    }
    return results

# ----------------------
# Build modeling table
# ----------------------
def build_modeling_table_for_game(game_key: str, all_results: dict) -> pd.DataFrame:
    if game_key not in all_results:
        raise ValueError(f"{game_key} not found in all_results keys: {list(all_results.keys())}")
    results_game = all_results[game_key]
    df = results_game['original_df'].copy()
    if 'author_steamid' not in df.columns:
        raise ValueError("author_steamid is missing in original_df for " + game_key)
    eng_res = results_game.get('engagement', {})
    play_res = results_game.get('playtime', {})

    if eng_res and eng_res.get('labels') is not None:
        df['engagement_cluster'] = eng_res['labels']
    else:
        df['engagement_cluster'] = -1
    if play_res and play_res.get('labels') is not None:
        df['playtime_cluster'] = play_res['labels']
    else:
        df['playtime_cluster'] = -1

    rev_res = results_game.get('reviewer', {})
    comp_res = results_game.get('composite', {})
    if rev_res and 'player_stats' in rev_res and rev_res.get('labels') is not None:
        player_stats = rev_res['player_stats'].copy()
        player_stats['reviewer_cluster'] = rev_res['labels']
        df = df.merge(player_stats[['author_steamid', 'reviewer_cluster']], on='author_steamid', how='left')
    else:
        df['reviewer_cluster'] = -1
    if comp_res and 'player_composite' in comp_res and comp_res.get('labels') is not None:
        player_comp = comp_res['player_composite'].copy()
        player_comp['composite_cluster'] = comp_res['labels']
        df = df.merge(player_comp[['author_steamid', 'composite_cluster']], on='author_steamid', how='left')
    else:
        df['composite_cluster'] = -1

    # Target label
    if 'voted_up' in df.columns:
        df['y'] = df['voted_up'].astype(int)
    else:
        if 'en_bert_label_from_partial' in df.columns:
            df['y'] = (df['en_bert_label_from_partial'] == 'positive').astype(int)
        else:
            raise ValueError(f"No voted_up or sentiment label found for {game_key}")

    numeric_cols = [
        'author_num_games_owned','author_num_reviews','community_participation',
        'author_playtime_forever','author_playtime_last_two_weeks','author_playtime_at_review',
        'recent_activity_ratio','recent_playtime_share','avg_sentiment','weighted_vote_score',
        'helpfulness_ratio','purchase_ratio'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    cluster_cols = ['engagement_cluster','playtime_cluster','reviewer_cluster','composite_cluster']
    meta_cols = [c for c in ['author_steamid','recommendationid'] if c in df.columns]
    keep_cols = meta_cols + cluster_cols + numeric_cols + ['y']
    modeling_df = df[keep_cols].copy()
    for c in numeric_cols:
        modeling_df[c] = pd.to_numeric(modeling_df[c], errors='coerce').fillna(0.0)
    for c in cluster_cols:
        modeling_df[c] = modeling_df[c].fillna(-1).astype(int)
    return modeling_df

def _fmt_sil(s):
    return "n/a" if (s is None or (isinstance(s, float) and np.isnan(s))) else f"{s:.3f}"

# ----------------------
# Main runtime (local CSVs first, Colab fallback)
# ----------------------
if __name__ == "__main__":
    import os
    print("\n=== Starting Local Data Load ===")

    # Local CSV filenames (the exact files you uploaded)
    local_paths = {
        'cyberpunk': 'dataset/steam_reviews_1091500.part000 (1).csv',
        'rdr2':      'dataset/steam_reviews_1174180.part000 (1).csv',
        'witcher':   'dataset/steam_reviews_292030.part000 (1).csv'
    }

    dfs = {}
    for key, path in local_paths.items():
        if os.path.exists(path):
            try:
                dfs[key] = pd.read_csv(path)
                print(f"Loaded {key} from local CSV: {path} (rows={len(dfs[key])})")
            except Exception as e:
                print(f"Error loading {key} from {path}: {e}")
        else:
            print(f"File not found for {key}: {path}")

    # If nothing loaded, try Colab Excel paths as fallback
    if not dfs:
        print("\nNo local datasets found. Trying Google Colab Excel paths (fallback)...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            colab_paths = {
                'cyberpunk': '/content/drive/MyDrive/Cyberpunk 2077_steam_reviews_1091500.part000_partial.xlsx',
                'rdr2': '/content/drive/MyDrive/Red Dead Redemption2_steam_reviews_1174180.part000_with_sent.xlsx',
                'witcher': '/content/drive/MyDrive/Witcher 3_steam_reviews_292030.part000_with_sent.xlsx'
            }
            for key, path in colab_paths.items():
                try:
                    dfs[key] = pd.read_excel(path)
                    print(f"Loaded {key} from Colab Excel: {path} (rows={len(dfs[key])})")
                except Exception:
                    print(f"Colab Excel not found for {key}: {path}")
        except Exception:
            print("Google Colab not available or mount failed. No datasets loaded. Exiting.")
            exit()

    # Proceed with analysis for loaded datasets
    if not dfs:
        print("No data available to process. Place your CSV files under ./dataset/ or run in Colab.")
        exit()

    print("\n=== Cleaning Data ===")
    cleaned = {k: clean_data(df) for k, df in dfs.items()}

    print("\n=== Creating Features ===")
    features = {k: create_features(df.copy()) for k, df in cleaned.items()}

    print("\n=== Running Clustering Analyses ===")
    all_results = {}
    game_names = {
        'cyberpunk': 'Cyberpunk 2077',
        'rdr2': 'Red Dead Redemption 2',
        'witcher': 'The Witcher 3'
    }
    for key, df in features.items():
        name = game_names.get(key, key)
        print(f"\n--- {name} ---")
        all_results[name] = perform_clustering_analysis(df, name)

    print("\n=== Clustering Summary ===")
    for g, r in all_results.items():
        print(
            f"{g}: "
            f"Eng={_fmt_sil(r['engagement']['silhouette'])}, "
            f"Rev={_fmt_sil(r['reviewer']['silhouette'])}, "
            f"Play={_fmt_sil(r['playtime']['silhouette'])}, "
            f"Comp={_fmt_sil(r['composite']['silhouette'])}"
        )

    # Optional small logistic model for Cyberpunk if dataset large enough
    if 'Cyberpunk 2077' in all_results:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, roc_auc_score

            cp_model_df = build_modeling_table_for_game("Cyberpunk 2077", all_results)
            unique_players = cp_model_df['author_steamid'].unique()
            if len(unique_players) > 10:
                train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
                train_df = cp_model_df[cp_model_df['author_steamid'].isin(train_players)].copy()
                test_df = cp_model_df[cp_model_df['author_steamid'].isin(test_players)].copy()

                drop_cols = [c for c in ['author_steamid', 'recommendationid', 'y'] if c in cp_model_df.columns]
                feature_cols = [c for c in cp_model_df.columns if c not in drop_cols]

                X_train = train_df[feature_cols]
                y_train = train_df['y'].astype(int)
                X_test = test_df[feature_cols]
                y_test = test_df['y'].astype(int)

                log_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
                log_clf.fit(X_train, y_train)
                y_pred = log_clf.predict(X_test)
                y_proba = log_clf.predict_proba(X_test)[:, 1]
                print("\n=== Logistic Regression Evaluation (Cyberpunk) ===")
                print(classification_report(y_test, y_pred, digits=3))
                print("ROC-AUC:", roc_auc_score(y_test, y_proba))
            else:
                print("Not enough unique players in Cyberpunk to train logistic model.")
        except Exception as e:
            print("Model training step failed:", e)

    print("\nMain script run complete.")
