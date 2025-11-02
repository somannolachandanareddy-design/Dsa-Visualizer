from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random
from collections import deque
import heapq

app = Flask(__name__)
CORS(app)

class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        steps = []
        n = len(arr)
        arr_copy = arr.copy()
        
        for i in range(n):
            for j in range(0, n - i - 1):
                steps.append({
                    'array': arr_copy.copy(),
                    'comparing': [j, j + 1],
                    'sorted': list(range(n - i, n)),
                    'action': f'Comparing {arr_copy[j]} and {arr_copy[j+1]}'
                })
                
                if arr_copy[j] > arr_copy[j + 1]:
                    arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                    steps.append({
                        'array': arr_copy.copy(),
                        'swapped': [j, j + 1],
                        'sorted': list(range(n - i, n)),
                        'action': f'Swapped {arr_copy[j+1]} and {arr_copy[j]}'
                    })
        
        steps.append({
            'array': arr_copy.copy(),
            'sorted': list(range(n)),
            'action': 'Sorting complete!'
        })
        
        return steps
    
    @staticmethod
    def quick_sort(arr):
        steps = []
        arr_copy = arr.copy()
        
        def partition(low, high):
            pivot = arr_copy[high]
            i = low - 1
            
            steps.append({
                'array': arr_copy.copy(),
                'pivot': [high],
                'action': f'Pivot: {pivot}'
            })
            
            for j in range(low, high):
                steps.append({
                    'array': arr_copy.copy(),
                    'comparing': [j, high],
                    'action': f'Comparing {arr_copy[j]} with pivot {pivot}'
                })
                
                if arr_copy[j] < pivot:
                    i += 1
                    arr_copy[i], arr_copy[j] = arr_copy[j], arr_copy[i]
                    steps.append({
                        'array': arr_copy.copy(),
                        'swapped': [i, j],
                        'action': f'Swapped {arr_copy[j]} and {arr_copy[i]}'
                    })
            
            arr_copy[i + 1], arr_copy[high] = arr_copy[high], arr_copy[i + 1]
            steps.append({
                'array': arr_copy.copy(),
                'swapped': [i + 1, high],
                'action': f'Placed pivot at position {i + 1}'
            })
            
            return i + 1
        
        def quick_sort_helper(low, high):
            if low < high:
                pi = partition(low, high)
                quick_sort_helper(low, pi - 1)
                quick_sort_helper(pi + 1, high)
        
        quick_sort_helper(0, len(arr_copy) - 1)
        
        steps.append({
            'array': arr_copy.copy(),
            'sorted': list(range(len(arr_copy))),
            'action': 'Quick Sort complete!'
        })
        
        return steps
    
    @staticmethod
    def merge_sort(arr):
        steps = []
        arr_copy = arr.copy()
        
        def merge(left, mid, right):
            left_arr = arr_copy[left:mid+1]
            right_arr = arr_copy[mid+1:right+1]
            
            i = j = 0
            k = left
            
            while i < len(left_arr) and j < len(right_arr):
                steps.append({
                    'array': arr_copy.copy(),
                    'comparing': [left + i, mid + 1 + j],
                    'action': f'Merging: comparing {left_arr[i]} and {right_arr[j]}'
                })
                
                if left_arr[i] <= right_arr[j]:
                    arr_copy[k] = left_arr[i]
                    i += 1
                else:
                    arr_copy[k] = right_arr[j]
                    j += 1
                k += 1
                
                steps.append({
                    'array': arr_copy.copy(),
                    'merging': [left, right],
                    'action': f'Placed {arr_copy[k-1]} at position {k-1}'
                })
            
            while i < len(left_arr):
                arr_copy[k] = left_arr[i]
                i += 1
                k += 1
            
            while j < len(right_arr):
                arr_copy[k] = right_arr[j]
                j += 1
                k += 1
        
        def merge_sort_helper(left, right):
            if left < right:
                mid = (left + right) // 2
                merge_sort_helper(left, mid)
                merge_sort_helper(mid + 1, right)
                merge(left, mid, right)
        
        merge_sort_helper(0, len(arr_copy) - 1)
        
        steps.append({
            'array': arr_copy.copy(),
            'sorted': list(range(len(arr_copy))),
            'action': 'Merge Sort complete!'
        })
        
        return steps

class SearchAlgorithms:
    @staticmethod
    def linear_search(arr, target):
        steps = []
        
        for i in range(len(arr)):
            steps.append({
                'array': arr.copy(),
                'checking': [i],
                'action': f'Checking index {i}: {arr[i]}'
            })
            
            if arr[i] == target:
                steps.append({
                    'array': arr.copy(),
                    'found': [i],
                    'action': f'Found {target} at index {i}!'
                })
                return steps
        
        steps.append({
            'array': arr.copy(),
            'action': f'{target} not found in array'
        })
        return steps
    
    @staticmethod
    def binary_search(arr, target):
        steps = []
        arr_sorted = sorted(arr)
        left, right = 0, len(arr_sorted) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            steps.append({
                'array': arr_sorted.copy(),
                'range': [left, right],
                'checking': [mid],
                'action': f'Checking middle index {mid}: {arr_sorted[mid]}'
            })
            
            if arr_sorted[mid] == target:
                steps.append({
                    'array': arr_sorted.copy(),
                    'found': [mid],
                    'action': f'Found {target} at index {mid}!'
                })
                return steps
            elif arr_sorted[mid] < target:
                left = mid + 1
                steps.append({
                    'array': arr_sorted.copy(),
                    'range': [left, right],
                    'action': f'{target} > {arr_sorted[mid]}, search right half'
                })
            else:
                right = mid - 1
                steps.append({
                    'array': arr_sorted.copy(),
                    'range': [left, right],
                    'action': f'{target} < {arr_sorted[mid]}, search left half'
                })
        
        steps.append({
            'array': arr_sorted.copy(),
            'action': f'{target} not found in array'
        })
        return steps

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class TreeAlgorithms:
    @staticmethod
    def build_bst(values):
        steps = []
        root = None
        
        def insert(node, value, level=0):
            if not node:
                steps.append({
                    'tree': TreeAlgorithms.serialize_tree(root) if root else {'value': value},
                    'action': f'Inserted {value}',
                    'highlight': [value]
                })
                return TreeNode(value)
            
            steps.append({
                'tree': TreeAlgorithms.serialize_tree(root),
                'action': f'Comparing {value} with {node.value}',
                'highlight': [node.value]
            })
            
            if value < node.value:
                node.left = insert(node.left, value, level + 1)
            else:
                node.right = insert(node.right, value, level + 1)
            
            return node
        
        for value in values:
            root = insert(root, value)
        
        return steps
    
    @staticmethod
    def inorder_traversal(root):
        steps = []
        visited = []
        
        def traverse(node):
            if node:
                traverse(node.left)
                visited.append(node.value)
                steps.append({
                    'tree': TreeAlgorithms.serialize_tree(root),
                    'action': f'Visiting {node.value}',
                    'highlight': [node.value],
                    'visited': visited.copy()
                })
                traverse(node.right)
        
        traverse(root)
        return steps
    
    @staticmethod
    def serialize_tree(node):
        if not node:
            return None
        return {
            'value': node.value,
            'left': TreeAlgorithms.serialize_tree(node.left),
            'right': TreeAlgorithms.serialize_tree(node.right)
        }

class GraphAlgorithms:
    @staticmethod
    def bfs(graph, start):
        steps = []
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        steps.append({
            'graph': graph,
            'visited': list(visited),
            'queue': list(queue),
            'current': start,
            'action': f'Starting BFS from {start}'
        })
        
        while queue:
            node = queue.popleft()
            
            steps.append({
                'graph': graph,
                'visited': list(visited),
                'queue': list(queue),
                'current': node,
                'action': f'Visiting node {node}'
            })
            
            for neighbor in sorted(graph.get(node, [])):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    steps.append({
                        'graph': graph,
                        'visited': list(visited),
                        'queue': list(queue),
                        'current': node,
                        'discovering': neighbor,
                        'action': f'Discovered {neighbor} from {node}'
                    })
        
        steps.append({
            'graph': graph,
            'visited': list(visited),
            'action': 'BFS complete!'
        })
        
        return steps
    
    @staticmethod
    def dfs(graph, start):
        steps = []
        visited = set()
        
        def dfs_helper(node):
            visited.add(node)
            steps.append({
                'graph': graph,
                'visited': list(visited),
                'current': node,
                'action': f'Visiting node {node}'
            })
            
            for neighbor in sorted(graph.get(node, [])):
                if neighbor not in visited:
                    steps.append({
                        'graph': graph,
                        'visited': list(visited),
                        'current': node,
                        'exploring': neighbor,
                        'action': f'Exploring {neighbor} from {node}'
                    })
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        
        steps.append({
            'graph': graph,
            'visited': list(visited),
            'action': 'DFS complete!'
        })
        
        return steps
    
    @staticmethod
    def dijkstra(graph, start):
        steps = []
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        visited = set()
        pq = [(0, start)]
        
        steps.append({
            'graph': graph,
            'distances': distances.copy(),
            'visited': list(visited),
            'current': start,
            'action': f'Starting Dijkstra from {start}'
        })
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            steps.append({
                'graph': graph,
                'distances': distances.copy(),
                'visited': list(visited),
                'current': current,
                'action': f'Visiting {current}, distance: {current_dist}'
            })
            
            for neighbor, weight in graph[current].items():
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
                    
                    steps.append({
                        'graph': graph,
                        'distances': distances.copy(),
                        'visited': list(visited),
                        'current': current,
                        'updating': neighbor,
                        'action': f'Updated distance to {neighbor}: {distance}'
                    })
        
        steps.append({
            'graph': graph,
            'distances': distances,
            'visited': list(visited),
            'action': 'Dijkstra complete!'
        })
        
        return steps

class StackQueue:
    @staticmethod
    def stack_operations(operations):
        steps = []
        stack = []
        
        for op in operations:
            if op['type'] == 'push':
                stack.append(op['value'])
                steps.append({
                    'stack': stack.copy(),
                    'action': f'Pushed {op["value"]}',
                    'highlight': [len(stack) - 1]
                })
            elif op['type'] == 'pop':
                if stack:
                    value = stack.pop()
                    steps.append({
                        'stack': stack.copy(),
                        'action': f'Popped {value}',
                        'popped': value
                    })
                else:
                    steps.append({
                        'stack': stack.copy(),
                        'action': 'Stack is empty!'
                    })
        
        return steps
    
    @staticmethod
    def queue_operations(operations):
        steps = []
        queue = []
        
        for op in operations:
            if op['type'] == 'enqueue':
                queue.append(op['value'])
                steps.append({
                    'queue': queue.copy(),
                    'action': f'Enqueued {op["value"]}',
                    'highlight': [len(queue) - 1]
                })
            elif op['type'] == 'dequeue':
                if queue:
                    value = queue.pop(0)
                    steps.append({
                        'queue': queue.copy(),
                        'action': f'Dequeued {value}',
                        'dequeued': value
                    })
                else:
                    steps.append({
                        'queue': queue.copy(),
                        'action': 'Queue is empty!'
                    })
        
        return steps

class HashMap:
    @staticmethod
    def operations(operations, size):
        steps = []
        hashmap = {}
        
        for op in operations:
            if op['type'] == 'insert':
                key = op['key']
                value = op['value']
                hash_val = hash(key) % size
                
                steps.append({
                    'hashmap': hashmap.copy(),
                    'action': f'Hash({key}) = {hash_val}',
                    'highlight': [hash_val],
                    'operation': 'hash'
                })
                
                if hash_val not in hashmap:
                    hashmap[hash_val] = []
                
                if hashmap[hash_val]:
                    steps.append({
                        'hashmap': hashmap.copy(),
                        'action': f'Collision at {hash_val}!',
                        'highlight': [hash_val],
                        'operation': 'collision'
                    })
                
                hashmap[hash_val].append((key, value))
                steps.append({
                    'hashmap': hashmap.copy(),
                    'action': f'Inserted ({key}, {value}) at {hash_val}',
                    'highlight': [hash_val],
                    'operation': 'insert'
                })
                
            elif op['type'] == 'search':
                key = op['key']
                hash_val = hash(key) % size
                found = False
                
                if hash_val in hashmap:
                    for k, v in hashmap[hash_val]:
                        if k == key:
                            steps.append({
                                'hashmap': hashmap.copy(),
                                'action': f'Found {key}: {v}',
                                'highlight': [hash_val],
                                'operation': 'found'
                            })
                            found = True
                            break
                
                if not found:
                    steps.append({
                        'hashmap': hashmap.copy(),
                        'action': f'{key} not found',
                        'highlight': [hash_val],
                        'operation': 'notfound'
                    })
        
        return steps

# Routes
@app.route('/dsavisualizer')
def index():
    return render_template('index.html')

@app.route('/api/sort', methods=['POST'])
def sort_array():
    data = request.json
    array = data.get('array', [])
    algorithm = data.get('algorithm', 'bubble')
    
    if algorithm == 'bubble':
        steps = SortingAlgorithms.bubble_sort(array)
    elif algorithm == 'quick':
        steps = SortingAlgorithms.quick_sort(array)
    elif algorithm == 'merge':
        steps = SortingAlgorithms.merge_sort(array)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/search', methods=['POST'])
def search_array():
    data = request.json
    array = data.get('array', [])
    target = data.get('target', 0)
    algorithm = data.get('algorithm', 'linear')
    
    if algorithm == 'linear':
        steps = SearchAlgorithms.linear_search(array, target)
    elif algorithm == 'binary':
        steps = SearchAlgorithms.binary_search(array, target)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/tree/build', methods=['POST'])
def build_tree():
    data = request.json
    values = data.get('values', [])
    steps = TreeAlgorithms.build_bst(values)
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/tree/traverse', methods=['POST'])
def traverse_tree():
    data = request.json
    values = data.get('values', [])
    
    root = None
    for value in values:
        if not root:
            root = TreeNode(value)
        else:
            def insert(node, val):
                if val < node.value:
                    if node.left:
                        insert(node.left, val)
                    else:
                        node.left = TreeNode(val)
                else:
                    if node.right:
                        insert(node.right, val)
                    else:
                        node.right = TreeNode(val)
            insert(root, value)
    
    steps = TreeAlgorithms.inorder_traversal(root)
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/graph/traverse', methods=['POST'])
def traverse_graph():
    data = request.json
    graph = data.get('graph', {})
    start = data.get('start')
    algorithm = data.get('algorithm', 'bfs')
    
    if algorithm == 'bfs':
        steps = GraphAlgorithms.bfs(graph, start)
    elif algorithm == 'dfs':
        steps = GraphAlgorithms.dfs(graph, start)
    elif algorithm == 'dijkstra':
        steps = GraphAlgorithms.dijkstra(graph, start)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/stack-queue', methods=['POST'])
def stack_queue_ops():
    data = request.json
    operations = data.get('operations', [])
    structure_type = data.get('type', 'stack')
    
    if structure_type == 'stack':
        steps = StackQueue.stack_operations(operations)
    else:
        steps = StackQueue.queue_operations(operations)
    
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/hashmap', methods=['POST'])
def hashmap_ops():
    data = request.json
    operations = data.get('operations', [])
    size = data.get('size', 10)
    
    steps = HashMap.operations(operations, size)
    return jsonify({'steps': steps, 'total_steps': len(steps)})

@app.route('/api/generate-array', methods=['POST'])
def generate_array():
    data = request.json
    size = data.get('size', 10)
    min_val = data.get('min', 1)
    max_val = data.get('max', 100)
    
    array = [random.randint(min_val, max_val) for _ in range(size)]
    return jsonify({'array': array})

if __name__ == '__main__':
    app.run(debug=True, port=5000)