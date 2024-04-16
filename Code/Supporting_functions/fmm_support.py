def get_neighbours_child(box, curr_child, i, other):
    parent_neighbours_child = curr_child ^ ((other+1)%2+1)
    if box.side_neighbours[i] is not None:
        return box.side_neighbours[i].get_Child_At_Index(parent_neighbours_child)
    else:
        return None
    
