

def can_place(r, c, room, rows, cols):
    cells = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
    for x, y in cells:
        if x < 0 or y < 0 or x >= rows or y >= cols or room[x][y] == 'H':
            return False
    return True

def shortest_moves(room, rows, cols, start_pos, target_pos):
    q = deque()
    visited = set()
    q.append((start_pos[0], start_pos[1], 0))
    visited.add((start_pos[0], start_pos[1]))
    while q:
        r, c, dist = q.popleft()
        if (r, c) == target_pos:
            return dist
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if can_place(nr, nc, room, rows, cols) and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, dist+1))
    return "Impossible"

def main():
    rows, cols = map(int, input().split())
    room = [input().split() for _ in range(rows)]
    start_pos = end_pos = None
    for i in range(rows):
        for j in range(cols):
            if room[i][j] == 'S':
                start_pos = (i, j)
            elif room[i][j] == 's':
                end_pos = (i, j)
    result = shortest_moves(room, rows, cols, start_pos, end_pos)
    print(result)

if _name_ == "_main_":
    main()
