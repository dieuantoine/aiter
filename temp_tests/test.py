from config import DATA_DIR
import pandas as pd

hyp_1 = pd.read_csv(DATA_DIR / "interim" / "mkqa" / "hypotheses_1.csv")
hyp_0 = pd.read_csv(DATA_DIR / "interim" / "mkqa" / "hypotheses_0.csv")

merged = hyp_1.merge(
    hyp_0[['request_id', 'model_id', 'hypothesis']],
    on=['request_id', 'model_id'],
    how='left',
    suffixes=('', '_from_0')
)

# Remplacer la colonne hypothesis si une valeur existe dans hyp_0
merged['hypothesis'] = merged['hypothesis_from_0'].combine_first(merged['hypothesis'])

# Supprimer la colonne temporaire
merged = merged.drop(columns=['hypothesis_from_0'])

# Sauvegarder le résultat
merged.to_csv(DATA_DIR / 'interim/mkqa/hypotheses_1_updated.csv', index=False)