import numpy as np 



def region_growing(img, seed_point, threshold):
    seed_value = img[seed_point]
    height, width, channels = img.shape
    output = np.zeros_like(img)

    def get_neighbours(point):
        x, y = point
        neighbours = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [n for n in neighbours if n[0]>=0 and n[0]<width and n[1]>=0 and n[1]<height]

    def similarity(pixel1, pixel2, threshold):
        return np.sqrt(np.sum((pixel1 - pixel2) ** 2)) < threshold

    queue = [seed_point]
    while queue:
        current_point = queue.pop(0)
        output[current_point] = 255
        for neighbour in get_neighbours(current_point):
           if (output[neighbour] == 0).all() and similarity(img[neighbour], seed_value, threshold):
                output[neighbour] = 255
                queue.append(neighbour)

    return output

    