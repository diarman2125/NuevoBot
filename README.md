# Bot Errores de Cuotas (NuevoBot.py)

## Despliegue en Railway
1. Sube estos archivos: `NuevoBot.py`, `Procfile`, `requirements.txt`.
2. En **Variables** añade:
   - `ODDS_API_KEY`
   - `TELEGRAM_BOT_TOKEN` (opcional, para alertas)
   - `TELEGRAM_CHAT_ID` (opcional, para alertas)
3. Asegúrate de eliminar cualquier **Custom Start Command** en Settings → Deploy.
4. Redeploy.

### Procfile
```
worker: python NuevoBot.py
```

### requirements.txt
```
requests
python-dateutil
pytz
```
