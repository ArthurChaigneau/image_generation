from StyleTransfer import StyleTransferNetwork

#Chemin de l'image source
image_content = "images/arthur.jpg"

#Chemin du style à appliquer à l'image
image_style = "images/van.jpg"

#Chemin de l'image de sortie
file_output = "result.png"

model = StyleTransferNetwork(image_content, image_style, file_output)

model.fit(epochs=10, steps_per_epoch=100)