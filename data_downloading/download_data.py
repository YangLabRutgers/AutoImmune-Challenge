import os
import requests

os.makedirs("./data",exist_ok=True)

address = "https://s3.embl.de/spatialdata/spatialdata-sandbox/xenium_rep1_io.zip"

response = requests.get(address,stream=True)

output_path = "xenium_rep1.zarr"

with open(output_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)