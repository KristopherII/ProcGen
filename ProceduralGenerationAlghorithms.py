from PIL import Image, ImageDraw
import opensimplex as simplex
import numpy as np


# work in progress
def binary_space_partition(width=100, height=100, seed=0, save=True, console=True):
    np.random.seed(seed)
    bsp_array = np.zeros((width, height), dtype=int)
    bsp_array_flattened = np.zeros((width * height), dtype=int)

    bsp_array_flattened = np.reshape(bsp_array * 255, (width * height), order='F')
    print('Saving 2D image...')
    im = Image.new('L', (width, height))
    im.putdata(bsp_array_flattened)
    if save:
        im.save('BinarySpacePartition/BSP.png')
    print('Algorithm not yet working, this is a template and work in progress')
    return im, bsp_array, bsp_array_flattened


def cellular_automata(iteration_count=7, width=100, height=100, density=50, seed=0, passed_map=None, save=True, console=True):
    np.random.seed(seed)

    def generate_random_noise():
        random_array = np.random.choice([0, 1], size=(width, height), p=[density / 100, (100 - density) / 100])
        if console:
            print('Generated image containing {} pixels, with density coefficient = {}%.'.format(width * height, density))
        return random_array

    def ca(iteration):
        kernel = np.ones((3, 3), dtype=int)
        kernel[1][1] = 0

        neighbour_count = np.zeros_like(map_data[0])
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbour_count += np.roll(map_data[iteration], (i, j), axis=(0, 1)) * kernel[i + 1, j + 1]

        tmp_grid = np.zeros_like(map_data[0])
        tmp_grid[neighbour_count > 4] = 1
        tmp_grid[neighbour_count < 4] = 0
        tmp_grid[neighbour_count == 4] = map_data[iteration][neighbour_count == 4]

        return tmp_grid

    map_data = np.zeros((iteration_count + 1, width, height), dtype=int)
    map_data_flattened = np.zeros((iteration_count + 1, width * height), dtype=int)

    if passed_map is None:
        tmp_grid = generate_random_noise()
    else:
        if console:
            print('Importing map...')
        tmp_grid = passed_map
    map_data[0] = tmp_grid

    for n in range(0, iteration_count):
        if console:
            print('Automata started shifting walls for the {} time...'.format(n + 1))
        map_data[n + 1] = ca(n)
    if console:
        print('Automata shifted {} walls, in {} generations.'.format(width * height * iteration_count, n + 1))

    for s in range(len(map_data)):
        map_data_flattened[s] = np.reshape(map_data[s] * 255, (width * height), order='F')
        im = Image.new('L', (width, height))
        im.putdata(map_data_flattened[s])
        if save:
            im.save('CellularAutomata/CA{}.png'.format(s))
            if not s and console:
                print('Saving {} 2D images...'.format(iteration_count + 1))
    return im, map_data, map_data_flattened


def diamond_square(width=100, height=100, roughness=8, seed=0, max_exponent=12, save=True, console=True):
    np.random.seed(seed)
    if console:
        print('Generating 2D image...')

    def square_step(map_array):
        for x in range(0, fit_size-1, chunk_size):
            for y in range(0, fit_size-1, chunk_size):
                map_array[x + half_chunk_size][y + half_chunk_size] = \
                    (map_array[x][y] + map_array[x + chunk_size][y] + map_array[x][y + chunk_size] +
                     map_array[x + chunk_size][y + chunk_size]) / 4 + np.random.uniform(-roughness, roughness)
        return map_array

    def diamond_step(map_array):
        half_chunk_size = chunk_size // 2
        for x in range(0, fit_size - 1, half_chunk_size):
            for y in range((x + half_chunk_size) % chunk_size, fit_size - 1, chunk_size):
                avg = (map_array[x][y - half_chunk_size] + map_array[x][y + half_chunk_size] +
                       map_array[x - half_chunk_size][y] + map_array[x + half_chunk_size][y]) / 4
                map_array[x][y] = avg + np.random.uniform(-roughness, roughness)
        return map_array


    # Fitting resolution into n^2 + 1
    fit_size = 0
    for n in range(max_exponent):
        if width <= 2 ** n and height <= 2 ** n:
            fit_size = 2 ** n +1
            break
    ds_array = np.zeros((fit_size, fit_size), dtype=float)
    ds_array_flattened = np.zeros((width * height), dtype=int)

    # Marking the corners
    ds_array[0][0] = 1
    ds_array[fit_size-1][fit_size-1] = 1
    ds_array[fit_size-1][0] = 1
    ds_array[0][fit_size-1] = 1

    # Map generation
    chunk_size = fit_size - 1
    while chunk_size > 1:
        half_chunk_size = chunk_size // 2
        ds_array = square_step(ds_array)
        ds_array = diamond_step(ds_array)
        chunk_size //= 2
        roughness /= 2

    # Cutting array into normalized shape
    ds_array = ds_array[1:width+1, 1:height+1]

    # Grey Scaling
    ds_array *= int(255 / np.max(ds_array))
    ds_array_flattened = np.reshape(ds_array.astype(int), (width * height), order='F')

    if console:
        print(f'Calculated grayscale map containing {width * height} pixels')
    im = Image.new('L', (width, height))
    im.putdata(ds_array_flattened)
    if save:
        if console:
            print('Saving 2D image...')
        im.save('DiamondSquare/DS.png')
    return im, ds_array, ds_array_flattened


# work in progress
def diffusion_limited_aggregation(width=100, height=100, seed=0, particle_count=1000,
                                  particle_steps=10000, seed_coords=None, save=True, console=True):
    np.random.seed(seed)
    dla_array = np.zeros((width, height), dtype=int)
    dla_array_flattened = np.zeros((width * height), dtype=int)
    if seed_coords is None:
        dla_array[int(width/2)][int(height/2)] = 1
    else:
        for sc in range(len(seed_coords)):
            dla_array[seed_coords[0]][seed_coords[1]] = 1

    while particle_count > 0:
        # Spawn particle
        particle_wall_spawn = np.random.randint(0, 2, dtype=int, size=2)
        if particle_wall_spawn[0]:
            particle_coords = np.array([np.random.randint(1, width+1, dtype=int), height*particle_wall_spawn[1]], dtype=int)
        elif not particle_wall_spawn[0]:
            particle_coords = np.array([width * particle_wall_spawn[1], np.random.randint(1, height + 1, dtype=int)], dtype=int)

        # Aggregate particle
        tries = particle_steps
        while tries > 0:
            # Generate random vector
            tries -= 1
        particle_count -= 1

    dla_array_flat = np.reshape(dla_array * 255, (width * height), order='F')
    print('Saving 2D image...')
    im = Image.new('L', (width, height))
    im.putdata(dla_array_flat)
    if save:
        im.save('DiffusionLimitedAggregation/DLA.png')
    print('Algorithm not yet working, this is a template and work in progress')
    return im, dla_array, dla_array_flattened


def drunkard_walk(width=100, height=100, seed=1, steps=1000, drunkenness=0.5, iteration=100, passed_map=None, save=True, console=True):
    start = np.array([int(width / 2), int(height / 2)], dtype=int)
    demolished_map = np.zeros((iteration, width, height), dtype=int)
    if passed_map is not None:
        demolished_map[-1] = passed_map
    demolished_map_flattened = np.reshape(demolished_map, (iteration, width * height), order='F')

    if console:
        print('Drunkard has started demolishing walls...')

    def drunkard(starting_point, temp_track=None, random_starting_point=False, random_start_tries=500):
        counter1 = 0
        previous_step = np.array([0, 0], dtype=int)
        track = temp_track
        location = starting_point
        tries = 0
        while random_starting_point:
            random_coords = np.array([np.random.randint(0, width - 1), np.random.randint(0, height - 1)], dtype=int)
            tries += 1
            if track[random_coords[0]][random_coords[1]]:
                location = random_coords
                break
            elif tries > random_start_tries:
                location = starting_point
                break
        track[location[0]][location[1]] = 1
        for n in range(steps):
            direction = np.random.randint(-1, 1, size=2)
            desire = np.random.randint(-100, 100, size=2)
            next_step = np.array([int((previous_step[0] * drunkenness) + (direction[0] * desire[0])),
                                  int((previous_step[1] * drunkenness) + (direction[1] * desire[1]))], dtype=int)

            if not next_step[0] or not next_step[1]:
                if not next_step[0] and not next_step[1]:
                    next_step = np.array([0, 0], dtype=int)
                elif not next_step[0] and next_step[1]:
                    next_step = np.array([0, next_step[1] / abs(next_step[1])], dtype=int)
                elif next_step[0] and not next_step[1]:
                    next_step = np.array([next_step[0] / abs(next_step[0]), 0], dtype=int)
            if next_step[0] and next_step[1]:
                next_step = np.array([next_step[0] / abs(next_step[0]), next_step[1] / abs(next_step[1])], dtype=int)

            border_check = np.add(location, next_step)

            if not 0 > border_check[0] and not border_check[0] >= width and \
                    not 0 > border_check[0] and not border_check[1] >= height:
                location = border_check

            if not track[location[0]][location[1]]:
                track[location[0]][location[1]] = 1
                counter1 += 1
        if console:
            print('Drunkard stumbled through {} walls'.format(counter1))
        return location, track

    for i in range(iteration):
        np.random.seed(seed * i)
        temp = drunkard(temp_track=demolished_map[-1], random_starting_point=True, starting_point=start)
        start = temp[0]
        demolished_map[i] = temp[1]

    for s in range(len(demolished_map)):
        demolished_map_flattened[s] = np.reshape(demolished_map[s] * 255, (width * height), order='F')
        im = Image.new('L', (width, height))
        im.putdata(demolished_map_flattened[s])
        if save:
            if not s and console:
                print('Saving {} 2D images...'.format(iteration))
            im.save('DrunkardWalk/Drunken_Walk{}.png'.format(s + 1))
    return im, demolished_map, demolished_map_flattened


def lazy_flood_fill(width=100, height=100, seed=0, iterations=50, blob_size=90,  save=True, console=True):
    np.random.seed(seed)
    lff_array = np.zeros((width, height), dtype=int)

    if console:
        print(f'Making {iterations} blobs, blob size {blob_size}')

    def make_blob():

        def handle_neighbours(x, y, col):

            # Up
            if y - 1 > 0 and not lff_array[x][y-1] and not lff_array[x][y-1] == 1:
                deque.append([x, y - 1, col])
                lff_array[x][y-1] = visited
            # Right
            if x + 1 < width and  not lff_array[x+1][y] and not lff_array[x+1][y] == 1:
                deque.append([x + 1, y, col])
                lff_array[x+1][y] = visited
            # Down
            if y + 1 < height and not lff_array[x][y+1] and not lff_array[x][y+1] == 1:
                deque.append([x, y + 1, col])
                lff_array[x][y+1] = visited
            # Left
            if x - 1 > 0 and not lff_array[x-1][y] and not lff_array[x-1][y] == 1:
                deque.append([x - 1, y, col])
                lff_array[x-1][y] = visited

        color = np.random.randint(50, 256, dtype=int)
        start = np.array([np.random.randint(0, width-1, dtype=int), np.random.randint(0, height-1, dtype=int), color], dtype=int)

        visited = 1
        chance = 100
        decay = ((blob_size*100)-1)/(blob_size*100)
        deque = []
        deque.insert(0, start)

        while deque:
            coords = deque.pop(0)
            lff_array[coords[0]][coords[1]] = coords[2]
            if chance >= np.random.randint(1, 101):
                handle_neighbours(x=coords[0], y=coords[1], col=coords[2])
            chance *= decay

    for n in range(iterations):
        make_blob()

    lff_array_flattened = np.reshape(lff_array, (width * height), order='F')
    im = Image.new('L', (width, height))
    im.putdata(lff_array_flattened)
    if save:
        if console:
            print('Saving 2D image...')
        im.save('LazyFloodFill/LFF.png')
    return im, lff_array, lff_array_flattened


def perlin_noise(width=100, height=100, seed=0, octaves=5, amplitude=255,
                 frequency=0.05, lacunarity=2, persistence=0.5, save=True, console=True):
    simplex.seed(seed)
    grid_flattened = np.zeros((octaves + 1, width * height), dtype=int)
    xy = 0
    if console:
        print(f'Generating {octaves} 2D images...')
    for y in range(0, height):
        for x in range(0, width):
            elevation = amplitude
            t_frequency = frequency
            t_amplitude = amplitude
            for z in range(octaves):
                sample_x = x * t_frequency
                sample_y = y * t_frequency
                elevation += simplex.noise2(sample_x, sample_y) * t_amplitude
                t_frequency *= lacunarity
                t_amplitude *= persistence
                grid_flattened[z][xy] = int(round(elevation / 2))
            xy += 1
    if console:
        print('Generated image containing {} pixels, using {} octaves.'.format(width * height, octaves))

    for n in range(octaves):
        im = Image.new('L', (width, height))
        im.putdata(grid_flattened[n])
        if save:
            im.save(f'PerlinNoise/PerlinNoiseOctave{n}.png')
            if not n and console:
                print(f'Saving {octaves} 2D images...')
    grid_flattened = np.delete(grid_flattened, octaves, 0)
    grid = np.reshape(grid_flattened, (octaves, width, height))
    return im, grid, grid_flattened


# work in progress
def seeded_aggregation(width=100, height=100, destruction_seeds=100, destruction=1000, seed=0, save=True, console=True):
    np.random.seed(seed)
    destruction_count = 0
    if console:
        print('Sowing seeds of destruction...')

    def sow_seeds(number_of_seeds=1):
        sa = np.zeros((width, height), dtype=int)
        seed_coord_x = np.random.randint(0, width, dtype=int, size=number_of_seeds)
        seed_coord_y = np.random.randint(0, height, dtype=int, size=number_of_seeds)
        for n in range(number_of_seeds):
            sa[seed_coord_x[n]][seed_coord_y[n]] = 1
        return sa

    sa_array = sow_seeds(destruction_seeds)
    while destruction_count < destruction:
        rand_coord_x = np.random.randint(0, width, dtype=int)
        rand_coord_y = np.random.randint(0, height, dtype=int)
        if sa_array[rand_coord_x][rand_coord_y]:
            aim = np.random.randint(1, 4, dtype=int)
            if aim == 1 and rand_coord_y < height - 1:
                sa_array[rand_coord_x][rand_coord_y + 1] = 1
                destruction_count += 1
            elif aim == 2 and rand_coord_x < width - 1:
                sa_array[rand_coord_x + 1][rand_coord_y] = 1
                destruction_count += 1
            elif aim == 3 and 1 < rand_coord_y:
                sa_array[rand_coord_x][rand_coord_y - 1] = 1
                destruction_count += 1
            elif aim == 4 and 1 < rand_coord_x:
                sa_array[rand_coord_x - 1][rand_coord_y] = 1
                destruction_count += 1
    if console:
        print('Sew {} seeds that destroyed {} pixels'.format(destruction_seeds, destruction_seeds + destruction))

    sa_array_flattened = np.reshape(sa_array * 255, (width * height), order='F')

    im = Image.new('L', (width, height))
    im.putdata(sa_array_flattened)
    if save:
        if console:
            print('Saving 2D image...')
        im.save('SeededAggregation/SA.png')
    return im, sa_array, sa_array_flattened


# work in progress
def simple_room_placement(width=100, height=100, seed=0, rooms_count=10, min_room_size=5, room_distance=5,
                          retry_limit=1000, passed_map=None, connection=False, save=True, console=True):
    np.random.seed(seed)

    def place_room(tmp_map_array):
        if console:
            print('Started placing rooms...')
        if tmp_map_array is None:
            tmp_map_array = np.zeros((width, height), dtype=int)
        discarded_rooms_counter = 0
        rooms_placed_counter = 0
        room_coord_list = np.zeros((rooms_count, 2), dtype=int)

        while rooms_placed_counter < rooms_count:

            # Room generation
            sx = np.random.randint(min_room_size, int(width / 8)+min_room_size, dtype=int)
            sy = np.random.randint(min_room_size, int(height / 8)+min_room_size, dtype=int)
            room = np.ones(((sx * 2) - 1, (sy * 2) - 1), dtype=int)
            room_geometry = np.array([len(room), len(room[0])], dtype=int)

            # Room coordinates
            r_coords = np.array([np.random.randint(1, width - 1, dtype=int),
                                 np.random.randint(1, height - 1, dtype=int)], dtype=int)

            # Border check
            if ((room_geometry[0] - 1) / 2) + room_distance < r_coords[0] < \
                    width - ((room_geometry[0] - 1) / 2 + room_distance) and \
                    ((room_geometry[1] - 1) / 2) + room_distance < r_coords[1] < \
                    height - ((room_geometry[1] - 1) / 2 + room_distance):

                # Check collision with other rooms
                collision_check = 0
                for i in range(int(r_coords[0] - room_distance - (room_geometry[0] - 1) / 2),
                               int(r_coords[0] + room_distance + (room_geometry[0] - 1) / 2)):
                    for j in range(int(r_coords[1] - room_distance - (room_geometry[1] - 1) / 2),
                                   int(r_coords[1] + room_distance + (room_geometry[1] - 1) / 2)):
                        collision_check += tmp_map_array[i][j]

                # Room placement
                if not collision_check:
                    for x in range(int(r_coords[0] - (room_geometry[0] - 1) / 2),
                                   int(r_coords[0] + ((room_geometry[0] + 1) / 2))):
                        for y in range(int(r_coords[1] - (room_geometry[1] - 1) / 2),
                                       int(r_coords[1] + ((room_geometry[1] + 1) / 2))):
                            tmp_map_array[x][y] = 1
                    room_coord_list[rooms_placed_counter] = r_coords
                    rooms_placed_counter += 1
                else:
                    discarded_rooms_counter += 1
            else:
                discarded_rooms_counter += 1
            if discarded_rooms_counter >= retry_limit:
                if console:
                    print('Retry limit exceeded')
                break
        if console:
            print('Rooms discarded: {}'.format(discarded_rooms_counter))
        return tmp_map_array, room_coord_list

    # repairs to the path placing needed - too regular
    # sort by distance
    def connect_rooms(rooms_map, rooms_to_connect_list):
        if console:
            print('Started connecting rooms...')
        for cnt in range(1, len(rooms_to_connect_list)):
            start = rooms_to_connect_list[cnt - 1]
            stop = rooms_to_connect_list[cnt]

            option1 = np.array([stop[0], start[1]], dtype=int)
            option2 = np.array([start[0], stop[1]], dtype=int)
            if cnt > rooms_count / 2:
                path = option1
            else:
                path = option2

            if start[0] > stop[0]:
                tmp = start[0]
                start[0] = stop[0]
                stop[0] = tmp
                # print('swapped x')
            if start[1] > stop[1]:
                tmp = start[1]
                start[1] = stop[1]
                stop[1] = tmp
                # print('swapped y')

            for x in range(start[0], stop[0]):
                rooms_map[x][path[1]] = 1
            for y in range(start[1], stop[1]):
                rooms_map[path[0]][y] = 1
        if console:
            print('Connected {} rooms'.format(cnt + 1))
        return rooms_map

    srp_array = place_room(passed_map)
    if connection:
        srp_array = connect_rooms(rooms_map=srp_array[0], rooms_to_connect_list=srp_array[1])
    else:
        srp_array = srp_array[0]

    srp_array_flattened = np.reshape(srp_array * 255, (width * height), order='F')

    im = Image.new('L', (width, height))
    im.putdata(srp_array_flattened)
    if save:
        if console:
            print('Saving 2D image...')
        im.save('SimpleRoomPlacement/SMP.png')
    return im, srp_array, srp_array_flattened


# work in progress
def voroi_diagrams(width=100, height=100, seed=0, save=True, console=True):
    np.random.seed(seed)

    # Generate random points
    points = np.array([[np.random.randint(0, width, size=50)], [np.random.randint(0, height, size=50)]], dtype=int)



    # Compute Voronoi diagram
    from scipy.spatial import Voronoi
    vor = Voronoi(points)

    # Create image and draw Voronoi diagram
    im = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(im)

    for i, region in enumerate(vor.regions):
        if not region:
            continue
        polygon = [vor.vertices[j] for j in region]
        draw.polygon(polygon, fill=tuple(np.random.randint(0, 255)))

    # Convert image to array
    vd_array = np.array(im, dtype=int)

    # Flatten image array
    vd_array_flattened = vd_array.flatten()

    return im, vd_array, vd_array_flattened






    # np.random.seed(seed)
    # vd_array = np.zeros((width, height), dtype=int)
    # vd_array_flattened = np.zeros((width * height), dtype=int)
    #
    # vd_array_flattened = np.reshape(vd_array * 255, (width * height), order='F')
    # print('Saving 2D image...')
    # im = Image.new('L', (width, height))
    # im.putdata(vd_array_flattened)
    # if save_flag:
    #     im.save_flag('VoroiDiagram/VD.png')
    # print('Algorithm not yet working, this is a template and work in progress')
    # return im, vd_array, vd_array_flattened


