import os

from Verfahren.Polarabstandsprojektion import PolarDistanceProjection
from Verfahren.Hauptkomponentenanalyse import PCA


img_name = "Tasse1.jpg"
img_path = os.path.join("Bilder", img_name)


variant = "PCA"

if variant == "PDP":
    pdp = PolarDistanceProjection()
    pdp.run(img_path)
    
elif variant == "PCA":
    pca = PCA()
    pca.run(img_path)