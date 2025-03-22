#!/usr/bin/env python
# coding: utf-8

# # 1. Array

# # An array is a linear data structure that stores elements of the same data type in contiguous memory locations. 
# #### 🔹Example-

# In[2]:


arr = [10, 20, 30, 40, 50]
print(arr)
print(arr[0])


# ## 📌Types of Arrays in Python 

# ### Python does not have built-in arrays like C/C++, but we can use lists or the array module.
# #### 1. One-Dimensional Array (List in Python)

# In[3]:


import array

arr = array.array('i', [10, 20, 30, 40, 50])  # 'i' indicates integer type
print(arr[2]) # Output: 30


# #### 2. Multi-Dimensional Array (NumPy) 

# In[4]:


import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[1][2])  # Output: 6


# ## 📌Array Operations & Time Complexities

# # ![image.png](attachment:image.png)

# ## 📌Searching Algorithms in Arrays 

# ### 🔍 1. Linear Search (O(n)) 

# In[5]:


def linearSearch(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1
arr = [10, 20, 30, 40, 50]
print(linearSearch(arr, 30))


# ### 🔍 2. Binary Search (O(log n)) – Requires Sorted Array

# In[6]:


def binary_search(arr, key):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
    return -1
arr = [10, 20, 30, 40, 50]
print(binary_search(arr, 30))  # Output: 2


# ## 📌Sorting Algorithms for Arrays 

# ### 1️⃣ Bubble Sort (O(n²)) – Simple but Inefficient 

# In[7]:


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
print(arr)  # Output: [11, 12, 22, 25, 34, 64, 90]


# ### 2️⃣ Quick Sort (O(n log n)) – Efficient for Large Datasets 

# In[8]:


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

arr = [64, 34, 25, 12, 22, 11, 90]
print(quick_sort(arr))  # Output: [11, 12, 22, 25, 34, 64, 90]


# ## 📌Advanced Array Problems & Solutions 

# ### ✅ Find the Largest & Smallest Element in an Array

# In[9]:


def find_largest_smallest(arr):
    return max(arr), min(arr)

arr = [10, 20, 30, 5, 99]
print(find_largest_smallest(arr))  # Output: (99, 5)


# ### ✅ Reverse an Array

# In[10]:


def reverse_array(arr):
    return arr[::-1]

arr = [10, 20, 30, 40, 50]
print(reverse_array(arr))  # Output: [50, 40, 30, 20, 10]


# ### ✅ Merge Two Sorted Arrays (O(n)) 

# In[11]:


def merge_sorted_arrays(arr1, arr2):
    return sorted(arr1 + arr2)

arr1 = [1, 3, 5]
arr2 = [2, 4, 6]
print(merge_sorted_arrays(arr1, arr2))  # Output: [1, 2, 3, 4, 5, 6]


# ## 📌Applications of Arrays

# ### ✔ Data Storage: Used in databases and spreadsheets.
# ### ✔ Image Processing: 2D arrays store pixel values in image processing.
# ### ✔ Game Development: Used for storing grids (e.g., chess, tic-tac-toe).
# ### ✔ Networking: Routers use arrays to store and process data packets.

# ## 📌Interview Questions on Arrays (GATE & Coding Rounds) 

# ### 💡 Basic Questions:
# ###      1️⃣ Find the largest & smallest elements in an array.
# ###      2️⃣ Reverse an array in-place.
# ### 💡 Medium Questions:
# ###      3️⃣ Find the second largest element without sorting.
# ###      4️⃣ Merge two sorted arrays without extra space.
# ### 💡 Advanced Questions:
# ###      5️⃣ Find subarrays with a given sum (Kadane’s Algorithm).
# ###      6️⃣ Find the missing number in an array of size n-1.

# # 2. LinkedList

# ## A linked list is a linear data structure where elements are stored in nodes and connected by pointers. Unlike arrays, linked lists do not require contiguous memory allocation. 

# ## 🔹 Types of Linked Lists: 

# ### ✔ Singly Linked List (SLL): Each node points to the next node.
# ### ✔ Doubly Linked List (DLL): Each node has pointers to both previous and next nodes.
# ### ✔ Circular Linked List (CLL): The last node connects to the first node.

# ## 📌Node Structure in Python 

# In[12]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


# ## 📌Singly Linked List (SLL) Operations & Complexity 

# # ![image.png](attachment:image.png)

# ## 📌Implementing a Singly Linked List in Python 

# In[13]:


class LinkedList:
    def __init__(self):
        self.head = None

    # Insert at the beginning
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    # Insert at the end
    def insert_at_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node

    # Delete a node
    def delete_node(self, key):
        temp = self.head
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        if temp is None:
            return
        prev.next = temp.next
        temp = None

    # Display linked list
    def display(self):
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

# Example Usage
ll = LinkedList()
ll.insert_at_beginning(10)
ll.insert_at_end(20)
ll.insert_at_end(30)
ll.display()  # Output: 10 -> 20 -> 30 -> None
ll.delete_node(20)
ll.display()  # Output: 10 -> 30 -> None


# ## 📌Doubly Linked List (DLL) Implementation 
# ### Each node has two pointers: one pointing to the next node and one pointing to the previous node.

# In[14]:


class DNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    # Insert at the beginning
    def insert_at_beginning(self, data):
        new_node = DNode(data)
        new_node.next = self.head
        if self.head:
            self.head.prev = new_node
        self.head = new_node

    # Insert at the end
    def insert_at_end(self, data):
        new_node = DNode(data)
        if not self.head:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
        new_node.prev = temp

    # Display list forward
    def display_forward(self):
        temp = self.head
        while temp:
            print(temp.data, end=" <-> ")
            temp = temp.next
        print("None")

# Example Usage
dll = DoublyLinkedList()
dll.insert_at_beginning(10)
dll.insert_at_end(20)
dll.insert_at_end(30)
dll.display_forward()  # Output: 10 <-> 20 <-> 30 <-> None


# ## 📌 Circular Linked List (CLL) Implementation 

# ### In a Circular Linked List, the last node points back to the first node.

# In[15]:


class CNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None

    # Insert at the end
    def insert(self, data):
        new_node = CNode(data)
        if not self.head:
            self.head = new_node
            self.head.next = self.head
            return
        temp = self.head
        while temp.next != self.head:
            temp = temp.next
        temp.next = new_node
        new_node.next = self.head

    # Display Circular List
    def display(self):
        temp = self.head
        if not temp:
            return
        while True:
            print(temp.data, end=" -> ")
            temp = temp.next
            if temp == self.head:
                break
        print("Back to Head")

# Example Usage
cll = CircularLinkedList()
cll.insert(10)
cll.insert(20)
cll.insert(30)
cll.display()  # Output: 10 -> 20 -> 30 -> Back to Head


# ## 📌Linked List Applications 

# ### ✔ Dynamic Memory Allocation: Used in OS for memory management.
# ### ✔ Graph Implementation: Used to represent adjacency lists in graphs.
# ### ✔ Undo/Redo Operations: Implemented in text editors.
# ### ✔ Browser History: Back and forward navigation.

# ## 📌 Common Linked List Interview Questions 

# ### 💡 Basic Questions:
# ###      1️⃣ Reverse a linked list.
# ###      2️⃣ Find the middle element of a linked list.
# 
# ### 💡 Medium Questions:
# ###      3️⃣ Detect and remove loops in a linked list.
# ###      4️⃣ Merge two sorted linked lists.
# 
# ### 💡 Advanced Questions:
# ###      5️⃣ Find the intersection point of two linked lists.
# ###      6️⃣ Clone a linked list with a random pointer. 

# ## 📌Key Takeaways

# ### ✅ Linked lists are dynamic (memory-efficient compared to arrays).
# ### ✅ Insertion & deletion are O(1) at the head, but O(n) at the tail/middle.
# ### ✅ Used in memory allocation, graph representation, undo/redo operations. 

# # 3. Stack

# ## A stack is a linear data structure that follows the Last In, First Out (LIFO) principle, meaning the last element added is the first one to be removed.

# ## 🔹 Key Stack Operations:
# ### ✔ Push: Add an element to the top of the stack.
# ### ✔ Pop: Remove the top element from the stack.
# ### ✔ Peek (Top): View the top element without removing it.
# ### ✔ isEmpty: Check if the stack is empty.

# ## 📌 Stack Operations & Time Complexity 

# # ![image.png](attachment:image.png)

# ## 📌 Implementing Stack in Python
# ### 1️⃣ Stack Using List

# In[16]:


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, data):
        self.stack.append(data)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return "Stack is empty"

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        return "Stack is empty"

    def is_empty(self):
        return len(self.stack) == 0

    def display(self):
        print(self.stack)

# Example Usage
s = Stack()
s.push(10)
s.push(20)
s.push(30)
s.display()  # Output: [10, 20, 30]
print(s.pop())  # Output: 30
print(s.peek())  # Output: 20


# ## 2️⃣ Stack Using Linked List 

# In[17]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class StackLL:
    def __init__(self):
        self.top = None

    def push(self, data):
        new_node = Node(data)
        new_node.next = self.top
        self.top = new_node

    def pop(self):
        if self.is_empty():
            return "Stack is empty"
        popped_data = self.top.data
        self.top = self.top.next
        return popped_data

    def peek(self):
        if self.is_empty():
            return "Stack is empty"
        return self.top.data

    def is_empty(self):
        return self.top is None

    def display(self):
        temp = self.top
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

# Example Usage
stack = StackLL()
stack.push(10)
stack.push(20)
stack.push(30)
stack.display()  # Output: 30 -> 20 -> 10 -> None
print(stack.pop())  # Output: 30
print(stack.peek())  # Output: 20


# ## 📌 Applications of Stack

# ### ✔ Expression Evaluation: Used in converting infix to postfix expressions.
# ### ✔ Function Call Management: Manages function calls using a call stack in programming languages.
# ### ✔ Undo/Redo Operations: Used in text editors and IDEs.
# ### ✔ Backtracking: Used in solving mazes, recursion, and pathfinding.
# ### ✔ Browser History Navigation: Manages back and forward navigation.

# ## 📌 Advanced Stack Problems & Solutions
# ### ✅ Reverse a String Using Stack 

# In[18]:


def reverse_string(s):
    stack = []
    for char in s:
        stack.append(char)
    return ''.join(stack.pop() for _ in range(len(stack)))

print(reverse_string("hello"))  # Output: "olleh"


# ### ✅ Balanced Parentheses Checker 

# In[19]:


def is_balanced(expression):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for char in expression:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack

print(is_balanced("{[()]}"))  # Output: True
print(is_balanced("{[(])}"))  # Output: False


# ### ✅ Implementing a Min Stack (Stack with Min Function)

# In[20]:


class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, data):
        self.stack.append(data)
        if not self.min_stack or data <= self.min_stack[-1]:
            self.min_stack.append(data)

    def pop(self):
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            return self.stack.pop()
        return "Stack is empty"

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else "Stack is empty"

# Example Usage
min_stack = MinStack()
min_stack.push(10)
min_stack.push(20)
min_stack.push(5)
print(min_stack.get_min())  # Output: 5
min_stack.pop()
print(min_stack.get_min())  # Output: 10


# ## 📌 Common Stack Interview Questions

# ### 💡 Basic Questions:
# ###      1️⃣ Implement a stack using an array.
# ###      2️⃣ Implement a stack using a linked list.
# 
# ### 💡 Medium Questions:
# ###      3️⃣ Reverse a string using a stack.
# ###      4️⃣ Check if a given expression has balanced parentheses.
# 
# ### 💡 Advanced Questions:
# ###      5️⃣ Design a stack that supports push, pop, and getMinimum in O(1).
# ###      6️⃣ Evaluate a postfix expression using a stack.

# ## 📌 Key Takeaways
# ### ✅ LIFO structure makes stacks useful for backtracking, recursion, and memory management.
# ### ✅ Push & Pop operations are O(1), making stack operations fast and efficient.
# ### ✅ Used in expression evaluation, function call stacks, and undo/redo features.

# # 4. Queue

# ## A queue is a linear data structure that follows the First In, First Out (FIFO) principle, meaning the first element added is the first one to be removed.

# ## 🔹 Key Queue Operations:
# ### ✔ Enqueue: Add an element to the rear (end) of the queue.
# ### ✔ Dequeue: Remove an element from the front of the queue.
# ### ✔ Front (Peek): Retrieve the front element without removing it.
# ### ✔ isEmpty: Check if the queue is empty.

# ## 📌 Queue Operations & Time Complexity 

# # ![image.png](attachment:image.png)

# ## 📌 Implementing Queue in Python
# ### 1️⃣ Queue Using List 

# In[21]:


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, data):
        self.queue.append(data)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        return "Queue is empty"

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        return "Queue is empty"

    def is_empty(self):
        return len(self.queue) == 0

    def display(self):
        print(self.queue)

# Example Usage
q = Queue()
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
q.display()  # Output: [10, 20, 30]
print(q.dequeue())  # Output: 10
print(q.peek())  # Output: 20


# ### 2️⃣ Queue Using Linked List 

# In[22]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class QueueLL:
    def __init__(self):
        self.front = self.rear = None

    def enqueue(self, data):
        new_node = Node(data)
        if not self.rear:
            self.front = self.rear = new_node
            return
        self.rear.next = new_node
        self.rear = new_node

    def dequeue(self):
        if not self.front:
            return "Queue is empty"
        temp = self.front
        self.front = self.front.next
        if not self.front:
            self.rear = None
        return temp.data

    def peek(self):
        if self.front:
            return self.front.data
        return "Queue is empty"

    def is_empty(self):
        return self.front is None

    def display(self):
        temp = self.front
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

# Example Usage
queue = QueueLL()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)
queue.display()  # Output: 10 -> 20 -> 30 -> None
print(queue.dequeue())  # Output: 10
print(queue.peek())  # Output: 20


# ## 📌 Types of Queues
# ### 1️⃣ Circular Queue
# #### A queue where the last element connects back to the first, making it circular. 

# In[23]:


class CircularQueue:
    def __init__(self, size):
        self.queue = [None] * size
        self.size = size
        self.front = self.rear = -1

    def enqueue(self, data):
        if (self.rear + 1) % self.size == self.front:
            return "Queue is full"
        elif self.front == -1:
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = data

    def dequeue(self):
        if self.front == -1:
            return "Queue is empty"
        data = self.queue[self.front]
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.size
        return data

    def display(self):
        if self.front == -1:
            print("Queue is empty")
            return
        index = self.front
        while True:
            print(self.queue[index], end=" -> ")
            if index == self.rear:
                break
            index = (index + 1) % self.size
        print("None")

# Example Usage
cq = CircularQueue(5)
cq.enqueue(10)
cq.enqueue(20)
cq.enqueue(30)
cq.display()  # Output: 10 -> 20 -> 30 -> None
print(cq.dequeue())  # Output: 10


# ### 2️⃣ Priority Queue
# #### A queue where elements are dequeued based on priority instead of FIFO.

# In[24]:


import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, data, priority):
        heapq.heappush(self.queue, (priority, data))

    def dequeue(self):
        if self.queue:
            return heapq.heappop(self.queue)[1]
        return "Queue is empty"

# Example Usage
pq = PriorityQueue()
pq.enqueue("Task1", 2)
pq.enqueue("Task2", 1)  # Higher priority
pq.enqueue("Task3", 3)
print(pq.dequeue())  # Output: Task2 (Highest priority)


# ## 📌 Applications of Queue
# ### ✔ CPU Scheduling: Used in round-robin scheduling.
# ### ✔ Printing Jobs: Manages print queue in printers.
# ### ✔ Graph Traversal: BFS (Breadth-First Search) uses queues.
# ### ✔ Networking: Used in data packet scheduling.

# ## 📌 Advanced Queue Problems & Solutions
# ### ✅ Reverse a Queue

# In[25]:


from collections import deque

def reverse_queue(q):
    stack = []
    while q:
        stack.append(q.popleft())
    while stack:
        q.append(stack.pop())

q = deque([10, 20, 30, 40])
reverse_queue(q)
print(list(q))  # Output: [40, 30, 20, 10]


# ### ✅ Generate Binary Numbers Using Queue

# In[26]:


from collections import deque

def generate_binary(n):
    q = deque()
    q.append("1")
    for _ in range(n):
        front = q.popleft()
        print(front, end=" ")
        q.append(front + "0")
        q.append(front + "1")

generate_binary(5)  # Output: 1 10 11 100 101


# ## 📌 Common Queue Interview Questions
# ### 💡 Basic Questions:
# ###       1️⃣ Implement a queue using an array.
# ###       2️⃣ Implement a queue using a linked list.
# 
# ### 💡 Medium Questions:
# ###      3️⃣ Reverse a queue using recursion.
# ###      4️⃣ Implement a circular queue.
# 
# ### 💡 Advanced Questions:
# ###      5️⃣ Design a priority queue.
# ###      6️⃣ Implement a queue using two stacks.

# ## 📌 Key Takeaways
# ### ✅ FIFO structure makes queues useful for task scheduling, graph traversal, and buffering.
# ### ✅ Enqueue & Dequeue operations are O(1), making queue operations fast and efficient.
# ### ✅ Used in CPU scheduling, networking, and BFS traversal.

# # 5. Tree
# 
# ## A tree is a non-linear hierarchical data structure consisting of nodes, where each node contains a value and pointers to child nodes.
# 
# ## 🔹 Key Terms in Trees:
# ### ✔ Root Node: The topmost node of the tree.
# ### ✔ Parent Node: A node that has child nodes.
# ### ✔ Child Node: Nodes connected under a parent node.
# ### ✔ Leaf Node: Nodes without children.
# ### ✔ Height of Tree: The longest path from the root to a leaf.
# ### ✔ Depth of Node: Distance from the root to a particular node. 

# ## 📌 Types of Trees
# ### ✔ Binary Tree: Each node has at most two children.
# ### ✔ Binary Search Tree (BST): Left subtree contains smaller values, right subtree contains larger values.
# ### ✔ Balanced Tree: Tree where the height difference between left and right subtrees is minimal (e.g., AVL Tree).
# ### ✔ Heap: A complete binary tree used in priority queues.
# ### ✔ Trie (Prefix Tree): Tree used for storing strings efficiently.

# ## 📌 Tree Operations & Time Complexity

# # ![image.png](attachment:image.png)

# ## 📌 Implementing a Binary Tree in Python
# ### 1️⃣ Node & Tree Class

# In[27]:


class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None


# ## 📌 Tree Traversal Techniques
# ### 1️⃣ Depth-First Search (DFS)
# ### ✔ Inorder Traversal (Left → Root → Right) – Used for retrieving sorted elements from a BST.
# ### ✔ Preorder Traversal (Root → Left → Right) – Used for creating a copy of the tree.
# ### ✔ Postorder Traversal (Left → Right → Root) – Used for deleting or freeing memory of nodes.

# In[28]:


def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.data, end=" ")
        inorder_traversal(root.right)

def preorder_traversal(root):
    if root:
        print(root.data, end=" ")
        preorder_traversal(root.left)
        preorder_traversal(root.right)

def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)
        postorder_traversal(root.right)
        print(root.data, end=" ")

# Example Usage
tree = BinaryTree()
tree.root = Node(10)
tree.root.left = Node(20)
tree.root.right = Node(30)
tree.root.left.left = Node(40)
print("Inorder Traversal:", end=" ")
inorder_traversal(tree.root)  # Output: 40 20 10 30


# ### 2️⃣ Breadth-First Search (BFS) – Level Order Traversal

# In[29]:


from collections import deque

def level_order_traversal(root):
    if not root:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.data, end=" ")
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

# Example Usage
print("\nLevel Order Traversal:", end=" ")
level_order_traversal(tree.root)  # Output: 10 20 30 40


# ## 📌 Binary Search Tree (BST) Operations
# ### 1️⃣ Insertion in BST

# In[30]:


def insert(root, key):
    if not root:
        return Node(key)
    if key < root.data:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root

# Example Usage
bst = BinaryTree()
bst.root = insert(bst.root, 50)
insert(bst.root, 30)
insert(bst.root, 70)
insert(bst.root, 20)
insert(bst.root, 40)
print("\nInorder Traversal (BST):", end=" ")
inorder_traversal(bst.root)  # Output: 20 30 40 50 70


# ### 2️⃣ Searching in BST

# In[31]:


def search(root, key):
    if not root or root.data == key:
        return root
    if key < root.data:
        return search(root.left, key)
    return search(root.right, key)

# Example Usage
found = search(bst.root, 40)
print("\nElement Found:", found.data if found else "Not Found")  # Output: 40


# ### 3️⃣ Deletion in BST

# In[32]:


def find_min(node):
    while node.left:
        node = node.left
    return node

def delete(root, key):
    if not root:
        return root
    if key < root.data:
        root.left = delete(root.left, key)
    elif key > root.data:
        root.right = delete(root.right, key)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        temp = find_min(root.right)
        root.data = temp.data
        root.right = delete(root.right, temp.data)
    return root

# Example Usage
bst.root = delete(bst.root, 30)
print("\nInorder After Deletion:", end=" ")
inorder_traversal(bst.root)  # Output: 20 40 50 70


# ## 📌 Applications of Trees
# ### ✔ Hierarchical Data Representation: Used in databases, file systems.
# ### ✔ Search Operations: Binary Search Trees (BSTs) allow efficient searching.
# ### ✔ Compiler Design: Abstract Syntax Trees (ASTs) for code parsing.
# ### ✔ AI & Machine Learning: Decision trees and random forests.
# ### ✔ Routing Algorithms: Tries in IP routing.

# ## 📌 Advanced Tree Problems & Solutions
# ### ✅ Find the Height of a Tree

# In[33]:


def tree_height(root):
    if not root:
        return 0
    return 1 + max(tree_height(root.left), tree_height(root.right))

print("\nHeight of Tree:", tree_height(bst.root))  # Output: 3


# ### ✅ Check if a Binary Tree is a BST

# In[34]:


def is_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.data <= min_val or root.data >= max_val:
        return False
    return is_bst(root.left, min_val, root.data) and is_bst(root.right, root.data, max_val)

print("\nIs BST:", is_bst(bst.root))  # Output: True


# ## 📌 Common Tree Interview Questions
# ### 💡 Basic Questions:
# ###      1️⃣ Implement a Binary Tree and its traversals.
# ###      2️⃣ Find the height of a tree.
# 
# ### 💡 Medium Questions:
# ###      3️⃣ Implement insertion and deletion in BST.
# ###      4️⃣ Check if a Binary Tree is a BST.
# 
# ### 💡 Advanced Questions:
# ###      5️⃣ Convert a Binary Tree to a Doubly Linked List.
# ###      6️⃣ Find the Lowest Common Ancestor (LCA) in a BST.

# ## 📌 Key Takeaways
# ### ✅ Trees are hierarchical structures used for efficient searching and sorting.
# ### ✅ BSTs provide O(log n) time complexity for insertion, deletion, and search.
# ### ✅ Used in AI, database indexing, routing algorithms, and compiler design.

# # 6. Heap

# In[ ]:




