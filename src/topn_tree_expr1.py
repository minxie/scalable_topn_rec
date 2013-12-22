import argparse


class topn_tree_node:
    def __init__(self, item):
        self.p_item = item
        self.p_children = NULL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-infile', nargs='?', dest='infile', default="", type=str,
                        help='Input File location.')
    parser.add_argument('-oufile', nargs='?', dest='oufile', default="", type=str,
                        help='Output File location.')
    args = parser.parse_args()

    iid = open(args.infile, 'r')
    oid = open(args.oufile, 'a')

    tree_root = topn_tree_node(-1)
    
    for line in iid:
        line = line.rstrip()
        fields = line.split()
        cur_node = tree_root
        for field in fields:
            item = int(field)
            if cur_node.p_children == NULL:
                cur_node.p_children = []
                child_node = topn_tree_node(item)
                cur_node.p_children.append(child_node)
                cur_node = child_node
            else:
                tmp_node = cur_node
                for child_node in cur_node.p_children:
                    if child_node.p_item == item:
                        cur_node = child_node
                if tmp_node == cur_node: # Didn't find match
                    child_node = topn_tree_node(item)
                    cur_node.p_children.append(child_node)
                    cur_node = child_node

    node_stack = []
    node_stack.append([tree_root, 0])
    while node_stack:
        cur_node, level = node_stack.pop()
        for i in xrange(level):
            oid.write('-')
        oid.write(str(cur_node.p_item) + '\n')
        if cur_node.p_children != NULL:
            for child_node in cur_node.p_children:
                node_stack.push([child_node, level+1])

    oid.close()
    iid.close()
