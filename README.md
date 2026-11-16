# PhishGuard IA · Hito 2

Clasificación defensiva de correos en español. Integrantes: Angel Van Gerven Romero y Camilo Junco Duque. Python 3.12.

Desde la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
python data/descargar.py
python src/phishguard.py entrenar data/spaphish.csv
python src/phishguard.py clasificar experiments/modelo.joblib
```

La descarga verifica la huella SHA-256 del CSV publicado. Fuente: [SpaPhish v5](https://data.mendeley.com/datasets/hz2d6gz7pc/5), Bustio-Martínez et al., DOI 10.17632/hz2d6gz7pc.5, CC BY 4.0. El ZIP incluye el descargador. El comando de clasificación solicita el asunto y el cuerpo en una línea, muestra la clase y hasta cinco términos influyentes, y no guarda el texto introducido. Carga solo el archivo de modelo generado localmente con el comando anterior.

`experiments/resultados.json` incluye los conteos por clase, configuración, métricas, matrices y registros con error. La matriz usa filas para etiquetas reales y columnas para predicciones, en orden legítimo (0), phishing (1). Los registros se numeran desde 1 en el CSV, excluyendo la cabecera. El umbral de éxito corresponde al Hito 1 y quedó fijado antes de ejecutar la prueba.

Los modelos usan solo asunto y cuerpo. El resultado sirve como apoyo a la revisión. Usa datos anonimizados en la demostración.
