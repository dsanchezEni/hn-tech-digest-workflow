# hn-tech-digest-workflow

Workflow Mistral Cloud « hn-tech-digest » : à chaque déclenchement manuel (jamais planifié), il lit la page d'accueil de Hacker News, ne retient que les actualités du secteur tech (pas les blogs personnels ni les sujets hors tech), puis envoie dans la boîte Gmail de l'utilisateur qui déclenche le workflow :

- un email en français résumant les principales actualités tech ;
- une pièce jointe audio MP3 (1 à 2 min, voix française Voxtral) présentant les titres comme un mini-podcast.

Le workflow tourne en `on_behalf_of` (identité de l'utilisateur qui le lance, connecteur Gmail + appels API Mistral avec ses credentials), déployé en déploiement managé « hardened » sur Mistral Cloud.

Voir `apps/worker/src/worker/workflows/hn_tech_digest.py`.
