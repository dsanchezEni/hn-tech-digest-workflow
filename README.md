# hn-tech-digest-workflow

Workflow Mistral Cloud « hn-tech-digest », **planifié tous les jours à 08:00 (Europe/Paris)** et lançable manuellement à la demande. À chaque exécution, il lit la page d'accueil de Hacker News, ne retient que les actualités du secteur tech (pas les blogs personnels ni les sujets hors tech), puis envoie par Gmail :

- un email en français résumant les principales actualités tech (destinataire par défaut : `dsanchez@campus-eni.fr`) ;
- une pièce jointe audio MP3 (1 à 2 min, voix française Voxtral `voice_592`) présentant les titres comme un mini-podcast.

## Architecture (mode planifié, non interactif)

Une exécution planifiée n'a pas d'utilisateur déclencheur : le workflow n'est donc **pas** `on_behalf_of` (les schedules sont interdits en OBO) et tourne avec l'identité de service du déploiement :

- appels Mistral (chat + TTS Voxtral) via `get_mistral_client()` avec les credentials du worker (`workflow_executor`) ;
- envoi du mail via l'**API REST Gmail** directement, avec un refresh token fourni par des **secrets du workspace** liés au déploiement : `gmail_client_id`, `gmail_client_secret`, `gmail_refresh_token` (exposés aux workers comme `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`). Aucune valeur secrète n'est dans le code.

Voir `apps/worker/src/worker/workflows/hn_tech_digest.py`.
