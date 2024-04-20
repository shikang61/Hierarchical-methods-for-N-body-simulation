def get_neighbours_child(box, curr_child, i, other):

    parent_neighbours_child = curr_child ^ ((other+1)%2+1)
    if box.side_neighbours[i] is not None:
        return box.side_neighbours[i].get_Child_At_Index(parent_neighbours_child)
    else:
        return None

def get_corner_neighbours(box):
    corner_neighbour = []
    DISREGARD = [1,2,0,3]
    CORNER_CHILDREN = (3, 2, 0, 1)
    for i, side_neigh in enumerate(box.side_neighbours):
        # Find remaining corner neighours at lower levels
        if side_neigh is not None:
            corner = side_neigh.side_neighbours[(i+1)%4]
            if corner is not None:
                if side_neigh.level == box.level:
                    # Find corner neighbours if side neighbour is same level as current box
                    """
                        | 1 >   | 
                    ^^^^---------
                      0 | * | 2 | 
                    ---------⌄⌄⌄⌄
                        < 3 |   | 
                    NW corner neighbour is side neighbour 1 of current box's neighbour 0
                    NE corner neighbour is side neighbour 2 of current box's neighbour 1
                    SE corner neighbour is side neighbour 3 of current box's neighbour 2
                    SW corner neighbour is side neighbour 0 of current box's neighbour 3
                    """
                    corner_neighbour.append(corner)
                elif side_neigh.level < box.level and i != DISREGARD[box.c_index]:
                    # Find corner neighbours if side neighbour is lower level than current box
                    """
                        ---------
                        |       |  
                        |   1   |  
                        |       | 
                -------------------------
                |       | 0 | 1 |       |
                |   0   ---------   2   |   
                |       | 2 | 3 |       |
                -------------------------
                        |       |   
                        |   3   | 
                        |       | 
                        ---------
                    If cindex == 0, disregard side neighbour 1
                    If cindex == 1, disregard side neighbour 2
                    If cindex == 2, disregard side neighbour 0
                    If cindex == 3, disregard side neighbour 3

                    NW corner neighbour is child 3 of side neighbour 0
                    NE corner neighbour is child 2 of side neighbour 1
                    SE corner neighbour is child 0 of side neighbour 2
                    SW corner neighbour is child 1 of side neighbour 3
                    """
                    corner_neighbour.append(corner.get_Child_At_Index(CORNER_CHILDREN[i]))
    return corner_neighbour