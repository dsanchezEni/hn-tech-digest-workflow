# hn-tech-digest-workflow

Workflow Mistral Cloud « hn-tech-digest », **déclenché manuellement uniquement (jamais planifié)**. À chaque exécution, il lit la page d'accueil de Hacker News **et** les flux RSS de l'actualité tech (cybersécurité, IA, produits tech : TechCrunch, Ars Technica, The Verge, Wired, VentureBeat, ZDNet, BleepingComputer, Krebs on Security, The Hacker News, Numerama), ne retient que les actualités du secteur tech (pas les blogs personnels ni les sujets hors tech), puis envoie dans la boîte Gmail de l'utilisateur qui déclenche le workflow :

- un email en français résumant les principales actualités tech (destinataire par défaut : **l'adresse Gmail propre de l'utilisateur qui déclenche le workflow**, résolue à l'exécution ; surchargeable via l'entrée `to`) ;
- une pièce jointe audio MP3 (1 à 2 min, voix française Voxtral `voice_592`) présentant les titres comme un mini-podcast.

## Architecture (on-behalf-of, exécution ponctuelle)

Le workflow est `on_behalf_of=True` : il s'exécute avec l'identité de l'utilisateur qui le lance. La plateforme interdit les schedules sur un workflow OBO, ce qui garantit le caractère « ponctuel, sur déclenchement manuel ».

- envoi du mail via le **connecteur `gmail`** (outil `send_email`, pièce jointe MP3 en base64, ≤ 3 Mo) avec les credentials Gmail de l'utilisateur déclencheur ;
- appels Mistral (chat + TTS Voxtral `voxtral-mini-tts-3-0`) via `get_mistral_client()` avec l'identité de l'utilisateur déclencheur. Aucun secret n'est dans le code ni lié au déploiement.

Le déploiement managé « hardened » (obligatoire pour OBO) est `hn-tech-digest-worker` sur Mistral Cloud.

Voir `apps/worker/src/worker/workflows/hn_tech_digest.py`.

## Déclencher une exécution

Depuis Studio (Workflows → `hn-tech-digest` → Start workflow, entrée `{}`), ou via l'API/MCP `execute_workflow` avec `workflow_identifier="hn-tech-digest"` et `input={}`. Entrées optionnelles : `to` (vide = sa propre adresse Gmail), `top_n` (défaut 10), `voice_id` (défaut `voice_592`).
