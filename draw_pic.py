from PIL import Image
def draw_grey(matrix,width,height):
    image=Image.new('L',(width,height))
    # print(width,height)
    for i in range(width):
        for j in range(height):
            value=matrix[i][j]*255
            image.putpixel((j,i),int(value))
    # image.save("grep_image.png")
