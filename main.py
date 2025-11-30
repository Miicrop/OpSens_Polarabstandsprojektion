import os

from Verfahren.Polarabstandsprojektion import PolarDistanceProjection
from Verfahren.Hauptkomponentenanalyse import PCA


img_name = "Tasse1.jpg"
img_path = os.path.join("Bilder", img_name)


variant = "PDP"

if variant == "PDP":
    pdp = PolarDistanceProjection()
    pdp.load_image(img_path)
    pdp.img_to_gray()
    pdp.img_to_binary()
    pdp.calculate_distance_transform()
    pdp.find_center()
    pdp.polar_projection()
    pdp.smooth_radii()
    pdp.analyze_frequencies("peak")
    pdp.print_results()
    pdp.visualize()
    pdp.animate_live()
    
elif variant == "PCA":
    pca = PCA()
    pca.load_image(img_path)
    pca.img_to_gray()
    pca.img_to_binary()
    pca.find_largest_contour()
    pca.principal_component_analysis()
    pca.visualize()