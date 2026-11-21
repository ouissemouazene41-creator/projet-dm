# Projet 19 : Efficacité Énergétique des Centres de Données (Datacenters)

## 📌 Architecture
1. **LSTM** : Modélise et prédit la charge CPU des serveurs à court terme.
2. **CNN** : Analyse spatiale des matrices thermiques 2D (baies de serveurs 8x8) pour détecter les points chauds (hotspots).
3. **Random Forest** : Ajuste dynamiquement la consigne de température des climatiseurs (HVAC) en croisant la prévision de charge et la sévérité thermique.

---

## 🚀 Installation & Exécution

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Entraîner le pipeline complet (LSTM + CNN + Random Forest)
```bash
python pipeline.py
```

### 3. Lancer la Démonstration Interactive (Streamlit)
```bash
streamlit run app.py
```
Accédez ensuite à l'interface dans votre navigateur (`http://localhost:8501`).

