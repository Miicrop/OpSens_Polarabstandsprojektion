import os

from Verfahren.Polarabstandsprojektion import PolarDistanceProjection
from Verfahren.Hauptkomponentenanalyse import PCA
from Verfahren.CircleIntersection import CircleIntersection


img_name = "Ei.png"
img_path = os.path.join("Bilder", img_name)


variant = "PDP"

if variant == "PDP":
    pdp = PolarDistanceProjection()
    pdp.run(img_path)
    
elif variant == "PCA":
    pca = PCA()
    pca.run(img_path)
    
elif variant == "CIRC":
    circle = CircleIntersection()
    circle.run(img_path)