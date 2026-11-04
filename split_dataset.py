import splitfolders

# 9 to 1 ratio because it's a tiny dataset
splitfolders.ratio("AnnotatedWasteDataset", "AnnotatedWasteDataset-Split", ratio=(.9, .1))
