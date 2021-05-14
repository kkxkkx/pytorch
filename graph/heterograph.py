import dgl
#heterograph的参数时字典，（src_type,edge_type，dst_type）
ratings = dgl.heterograph(
    {('user', '+1', 'movie') : [(0, 0), (0, 1), (1, 0)],
     ('user', '-1', 'movie') : [(2, 1)]})