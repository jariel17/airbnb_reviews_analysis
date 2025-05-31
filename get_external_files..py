import requests

## DESCARGA Y DESCOMPRIME LOS ARCHIVOS CSV PARA EL PROYECTO


## DESCARGA EL MODELO DE DETECCIÓN DE IDIOMA
url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
output_file = "lid.176.bin"

# Descargar el archivo
response = requests.get(url, stream=True)
with open(output_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"Descarga completa: {output_file}")
