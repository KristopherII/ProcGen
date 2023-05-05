import tkinter as tk
from ProceduralGenerationAlghorithms import *
from Pathfinding import *
from PIL import Image, ImageTk
# import networkx as nx


# Gui functions
if True:
    def draw_next_frame(frame_to_delete_id, x, y, passed_image):
        canvas.delete(frame_to_delete_id)
        img = canvas.create_image(x, y, image=passed_image)
        return img


    def draw_previous_frame(frame_to_delete_id, x, y, passed_image):
        canvas.delete(frame_to_delete_id)
        img = canvas.create_image(x, y, image=passed_image)
        return img


    def generate_from_new_seed(width, height, passed_data):
        global spritesheet
        if not (len(passed_data[2]) == width * height):
            spritesheet = []
            for n in range(len(passed_data[2])):
                temp = Image.new('L', (width, height))
                temp.putdata(passed_data[2][n])
                spritesheet.append(ImageTk.PhotoImage(temp))

            img = canvas.create_image(width / 2, height / 2, image=spritesheet[0])
        elif len(passed_data[2]) == width * height:
            img = Image.new('L', (width, height))
            img.putdata(passed_data[2])
            spritesheet = ImageTk.PhotoImage(img)
            img = canvas.create_image(width / 2, height / 2, image=spritesheet)
        return img


# Global variables and Tkinter init
if True:
    SCALE = 0.5
    root = tk.Tk()
    WIDTH = int(root.winfo_screenwidth() * SCALE)
    HEIGHT = int(root.winfo_screenheight() * SCALE)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT)
    canvas.pack()
    # root.eval('tk::PlaceWindow . center')
    # root.overrideredirect(True)

    pressleft = pressright = pressdown = pressup = r_key = save_flag = False
    console_flag = True
    targeted_function = 0
    data = next_img = prev_img = img = spritesheet = None
    center_x, center_y = WIDTH / 2, HEIGHT / 2
    frame = 0
    SEED = 0

# Tkinter buttons
if True:
    def close(event):
        global running
        running = False


    def right(event):
        global pressright
        pressright = True


    def left(event):
        global pressleft
        pressleft = True


    def down(event):
        global pressdown, targeted_function
        targeted_function += 1
        if targeted_function > 10:
            targeted_function = 1
        pressdown = True


    def up(event):
        global pressup, targeted_function
        targeted_function -= 1
        if targeted_function < 1:
            targeted_function = 10
        pressup = True

    def r(event):
        global r_key
        r_key = True

    def save_on_off(event):
        global save_flag
        save_flag = not save_flag
        if save_flag:
            print('Save images: on')
        else:
            print('Save images: off')

    def console_on_off(event):
        global console_flag
        console_flag = not console_flag
        if console_flag:
            print('Console comments: on')
        else:
            print('Console comments: off')

#Tkinter bindings
if True:
    root.bind('<Escape>', close)
    root.bind('<Left>', left)
    root.bind('<Right>', right)
    root.bind('<Down>', down)
    root.bind('<Up>', up)
    root.bind('<r>', r)
    root.bind('<s>', save_on_off)
    root.bind('<c>', console_on_off)

# map_to_pass = simple_room_placement(width=WIDTH, height=HEIGHT, save=False, console=False)

running = True
while running:
    root.update()

    if pressleft:
        pressleft = False
        frame -= 1
        try:
            next_img = img
            if frame < 0:
                frame += 1
            img = draw_previous_frame(x=center_x, y=center_y, frame_to_delete_id=next_img,
                                      passed_image=spritesheet[frame])
        except:
            if spritesheet is not None:
                print('Selected method generates a single image')
            else:
                print('Spritesheet not initialized')
            if next_img is None:
                print('Generate image first')

    if pressright:
        pressright = False
        frame += 1
        try:
            prev_img = img
            if frame >= len(spritesheet):
                frame -= 1
            img = draw_next_frame(x=center_x, y=center_y, frame_to_delete_id=prev_img, passed_image=spritesheet[frame])
        except:
            if spritesheet is not None:
                print('Selected method generates a single image')
            else:
                print('Spritesheet not initialized')
            if next_img is None:
                print('Generate image first')

    if r_key:
        r_key = False
        SEED += 1
        try:
            img = generate_from_new_seed(width=WIDTH, height=HEIGHT,
                                         passed_data=data(width=WIDTH, height=HEIGHT, seed=SEED, save=save_flag,
                                                          console=console_flag))  # , passed_map=map_to_pass[1]))
        except:
            print('Select method first')
    if pressup or pressdown:
        pressup = pressdown = False

        if targeted_function == 1:
            print('Selected method: Binary space partition')
            data = binary_space_partition
        elif targeted_function == 2:
            print('Selected method: Cellular Automata')
            data = cellular_automata
        elif targeted_function == 3:
            print('Selected method: Diamond-Square')
            data = diamond_square
        elif targeted_function == 4:
            print('Selected method: Diffusion Limited Aggregation')
            data = diffusion_limited_aggregation
        elif targeted_function == 5:
            print('Selected method: Drunkard Walk')
            data = drunkard_walk
        elif targeted_function == 6:
            print('Selected method: Lazy Flood Fill')
            data = lazy_flood_fill
        elif targeted_function == 7:
            print('Selected method: Perlin Noise')
            data = perlin_noise
        elif targeted_function == 8:
            print('Selected method: Seeded Aggregation')
            data = seeded_aggregation
        elif targeted_function == 9:
            print('Selected method: Simple Room Placement')
            data = simple_room_placement
        elif targeted_function == 10:
            print('Selected method: Voronoi Diagrams')
            data = voroi_diagrams
