# What's berto's inferno?

the main purpose of this, is to explain and internalize these weird solutions that i could never come up with on my own, with the goal of preparing for interviews and pushing through this hell up to heaven, like dante with beatrice

![bertosinferno-img](./img/bertosinferno.jpg)

# What's my current approach?

anki deck for spaced repetition and

- **leetcode free:**
randomly working through problems from [sean prashad list](https://seanprashad.com/leetcode-patterns/)

- **leetcode premium:**
following company specific problems based on frequency

i try solving the question for 30 minutes, then look at the solution until i fully understand it

# Problems index
- [14. Longest Common Prefix](#14-longest-common-prefix)
- [15. 3Sum](#15-3sum)
- [17. Letter Combinations of a Phone Number](#17-letter-combinations-of-a-phone-number)
- [23. Merge k Sorted Lists](#23-merge-k-sorted-lists)
- [31. Next Permutation](#31-next-permutation)
- [34. Find First and Last Position of Element in Sorted Array](#34-find-first-and-last-position-of-element-in-sorted-array)
- [46. Permutations | 77. Combinations | 78. Subsets](#46-permutations--77-combinations--78-subsets)
- [50. Pow(x,n)](#50-powxn)
- [56. Merge Intervals](#56-merge-intervals)
- [71. Simplify Path](#71-simplify-path)
- [88. Merge Sorted Array](#88-merge-sorted-array)
- [129. Sum Root to Leaf Numbers](#129-sum-root-to-leaf-numbers)
- [133. Clone Graph](#133-clone-graph)
- [138. Copy List with Random Pointer](#138-copy-list-with-random-pointer)
- [146. LRU Cache](#146-lru-cache)
- [162. Find Peak Element](#162-find-peak-element)
- [199. Binary Tree Right Side View](#199-binary-tree-right-side-view)
- [207. Course Schedule](#207-course-schedule)
- [227. Basic Calculator II](#227-basic-calculator-ii)
- [215. Kth Largest Element in an Array](#215-kth-largest-element-in-an-array)
- [236. Lowest Common Ancestor of a Binary Tree](#236-lowest-common-ancestor-of-a-binary-tree)
- [238. Product of Array Except Self](#238-product-of-array-except-self)
- [314. Binary Tree Vertical Order Traversal](#314-binary-tree-vertical-order-traversal-premium--premium)
- [339. Nested List Weight Sum](#339-nested-list-weight-sum-premium)
- [346. Moving Average from Data Stream](#346-moving-average-from-data-stream-premium--premium) 
- [347. Top K Frequent Elements](#347-top-k-frequent-elements)
- [398. Random Pick Index](#398-random-pick-index)
- [426. Convert Binary Search Tree to Sorted Doubly Linked List](#426-convert-binary-search-tree-to-sorted-doubly-linked-list-premium--premium) 
- [523. Continuous Subarray Sum](#523-continuous-subarray-sum)
- [528. Random Pick with Weight](#528-random-pick-with-weight)
- [543. Diameter of Binary Tree](#543-diameter-of-binary-tree)
- [560. Subarray Sum Equals K](#560-subarray-sum-equals-k)
- [637. Valid Word Abbreviation](#637-valid-word-abbreviation-premium--premium)
- [670. Maximum Swap](#670-maximum-swap)
- [680. Valid Palindrome II](#680-valid-palindrome-ii)
- [791. Custom Sort String](#791-custom-sort-string)
- [827. Making A Large Island](#827-making-a-large-island)
- [921. Minimum Add to Make Parentheses Valid](#921-minimum-add-to-make-parentheses-valid)
- [938. Range Sum of BST](#938-range-sum-of-bst)
- [973. K Closest Points to Origin](#973-k-closest-points-to-origin)
- [1004. Max Consecutive Ones III](#1004-max-consecutive-ones-iii)
- [1091. Shortest Path in Binary Matrix](#1091-shortest-path-in-binary-matrix)
- [1249. Minimum Remove to Make Valid Parentheses](#1249-minimum-remove-to-make-valid-parentheses)
- [1570. Dot Product of Two Sparse Vectors](#1570-dot-product-of-two-sparse-vectors-premium--premium)
- [1650. Lowest Common Ancestor of a Binary Tree III](#1650-lowest-common-ancestor-of-a-binary-tree-iii-premium)
- [1762. Buildings With an Ocean View](#1762-buildings-with-an-ocean-view-premium)

---

## [14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix)

### key idea

vertical scanning

pick a base, loop through all the words and as soon as one character is different from the base or we went through the whole word (this will hit when we encounter the shortest word), then return the prefix from the start to the point we arrived 

~~~py
class Solution:
    def longestCommonPrefix(self, v: List[str]) -> str:
        if len(v) == 0:
            return ""

        base = v[0]
        for i in range(len(base)):
            for word in v[1:]:
                if i == len(word) or word[i] != base[i]:
                    return base[0:i]
        
        return base
~~~

**complexity**
~~~
time = O(S)
~~~
S -> the sum of all chars in all strings, in the worst case n equal strings with length m so S = n * m

~~~
space = O(1)
~~~
no extra space used

## [15. 3Sum](https://leetcode.com/problems/3sum/description)

### key idea

use three pointers, one fixed at the beginning, two that moves from i + 1 to nums.length

the key here is:

- check if the total is < 0, > 0 or == 0
- < 0, increment j so the number gets bigger
- \> 0, decrement k so the number gets smaller
- == 0 append to res and move the two pointers
- <ins>skip whenever i or j or k number is the same as the one before</ins>

~~~py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i in range(len(nums) - 2):
            # skip i duplicates
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            j, k = i + 1, len(nums) - 1
            while j < k:
                if nums[i] + nums[j] + nums[k] < 0:
                    j += 1
                elif nums[i] + nums[j] + nums[k] > 0:
                    k -= 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1

                    # skip j duplicates
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1

                    # skip k duplicates
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        
        return res
~~~

**complexity**
~~~
time = O(N^2 + NlogN) = O(N^2)
          ^       ^
    for & while  sorting
~~~

~~~
space = O(1) or O(N) <--
~~~
in python the sort takes N of space complexity due to how the library implemented the method

## [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number)

### key idea

hashmap to map the numbers to its respective letters

then backtrack to explore all different possibilities going by each digit one by one

~~~py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        num_to_chars = {
            "2": ["a", "b", "c"], 
            "3": ["d", "e", "f"], 
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"], 
            "6": ["m", "n", "o"], 
            "7": ["p", "q", "r", "s"], 
            "8": ["t", "u", "v"], 
            "9": ["w", "x", "y", "z"]
            }
        res = []

        def dfs(i, cur_string):
            if i == len(digits):
                res.append("".join(cur_string))
                return

            for c in num_to_chars[digits[i]]:
                cur_string.append(c)
                dfs(i + 1, cur_string)
                cur_string.pop()
        
        dfs(0, [])
        return res
~~~

**complexity**
~~~
time = O(4^N * N)
~~~
in the worst case we could get N digits of only '7' and/or '9' and as you can see its corresponding letters array is of length 4

~~~
space = O(N) or O(4^N) if we count recursive stack space
~~~

## [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists)

### key idea

heap solution works but it's not optimized since the heap take some space

the most optimal solution is merging the lists two by two by defining an interval where we pick the two lists

each time this interval doubles so the result halves each time and at the end we are going to have the first list that represents the result, so all the lists merged into the first one

~~~py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
            
        def merge(l1, l2):
            if not l1:
                return l2
            if not l2:
                return l1
            else:
                if l1.val <= l2.val:
                    l1.next = merge(l1.next, l2)
                    return l1
                else:
                    l2.next = merge(l1, l2.next)
                    return l2

        interval = 1

        while interval < len(lists):
            for i in range(0, len(lists) - interval, interval * 2):
                lists[i] = merge(lists[i], lists[i + interval])
            interval *= 2
        return lists[0]
~~~

**complexity**
~~~
time = O(N*logK)
~~~
N -> total number of elements\\
log(k) -> we halves each time the input by picking only two lists at a time

~~~
space = O(1)
~~~
no extra memory used, the result will be the first list that is already present in the input

## [31. Next Permutation](https://leetcode.com/problems/next-permutation)

### key idea

the idea is simple, why it works is a little bit harder to explain

<ins>**idea**</ins>
since it must be done in place, there needs to be some sort of swapping

if we look from right to left and all the numbers are in increasing order, then there is no next lexicographical permutation

now we must go in reverse order, finding the pivot, then after the pivot is found we must swap the pivot with the first number that is greater than the pivot from right to left

and after this reverse the part of the array after the pivot

<ins>**why it works?**</ins>

going from right to left and finding the pivot, means that at the right of the pivot, since the array is in increasing order (reversed), there can't be some rearrangement that create a larger lexicographical permutation

now we want to create a permutation that is just larger than the current one, therefore we need to replace the pivot with the number which is just larger than itself among the numbers on the right part of the pivot

something like this lol

~~~py
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # 1 4 5 8 7

        pivot = None

        # find the pivot
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                pivot = i - 1
                break
        else: # if not found (full iteration done), then just return the 
              #     reversed array as in the example
            nums.reverse()
            return
        
        # swap the pivot with the first greater num found from right to left
        swap = len(nums) - 1
        while nums[swap] <= nums[pivot]:
            swap -= 1

        nums[swap], nums[pivot] = nums[pivot], nums[swap]

        # reverse the part after the pivot because yes
        nums[pivot + 1:] = reversed(nums[pivot + 1:])
~~~

**complexity**
~~~
time = O(N + N) -> O(N)
~~~
in the worst case the for loop finishes, so full iteration + reversing

~~~
space = O(1)
~~~
all done in-place, baby :*

## [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description)

### key idea

it's a binary search slightly modified

basically we search for the left most and right most index of the target value, using two binary search

~~~py
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search_idx(is_left_check):
            l, r = 0, len(nums) - 1
            idx = -1

            while l <= r:
                mid = (l + r) // 2

                if nums[mid] < target:
                    l = mid + 1
                elif nums[mid] > target:
                    r = mid - 1
                else:
                    idx = mid
                    if is_left_check:
                        r = mid - 1
                    else:
                        l = mid + 1
            return idx

        left_idx = search_idx(True)
        right_idx = search_idx(False)

        return [left_idx, right_idx]
~~~

**complexity**
~~~
time = O(logN)
~~~
binary search, input halves each time

~~~
space = O(1)
~~~
no extra memory used

## [46. Permutations](https://leetcode.com/problems/permutations/description/) | [77. Combinations](https://leetcode.com/problems/combinations/description/) | [78. Subsets](https://leetcode.com/problems/subsets/description/)

### key idea

these 3 problems are quite similar, the purpose is to arrange the given numbers based on some constraints

tipically when we want to arrange something, we use *backtracking* because recursion provides us an handy way to handle arrangements

### Permutations

**constraints**
- literally every elements but in different order

~~~py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        permutations = []
        checked = [False] * 21

        def dfs(i):
            if i == len(nums):
                res.append(permutations[:])
                return
            
            for j in range(len(nums)):
                if checked[j]:
                    continue
                
                checked[j] = True
                permutations.append(nums[j])
                
                dfs(i + 1)

                checked[j] = False
                permutations.pop()

        dfs(0)
        return res
~~~

**complexity**
~~~
time = O(N! * N^2) 
~~~
n! = order of permutations & n*n = n elements insertion into permutations array for each iterations

~~~
space = O(N! * N) 
~~~
if we are not counting the res array, because each permutation is of size n! and we building permutations iteratively so n


### Combinations

**constraints**
- each combination of numbers n must be of size k 
- no repetitions

~~~py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        combinations = []

        def dfs(i):
            if i > n + 1:
                return

            if len(combinations) == k:
                res.append(combinations[:])
                return

            for j in range(i, n + 1):
                combinations.append(j)
                dfs(j + 1)
                combinations.pop()

        dfs(1)
        return res
~~~

**complexity**
~~~
time = O(N choose K * K) 
~~~
the dfs generates all combinations of k elements from a set of n elements (binomial coefficient)
for each combination, there is an appending operation of k-elements to res array

~~~
space = O(N choose K) 
~~~
n choose k = order of combinations


### Subsets

**constraints**
- can contain empty array
- order is important

~~~py
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [] 
        subset = []

        def dfs(i):
            if i == len(nums):
                res.append(subset[:])
                return

            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            dfs(i + 1)
    
        dfs(0)
        return res
~~~

**complexity**
~~~
time = O(N * 2^N) 
~~~
n = how many subsets do we need & 2^n = subset order

~~~
space = O(2^N) 
~~~
2^N = order of subsets

## [50. Pow(x,n)](https://leetcode.com/problems/powx-n/description)

### key idea
the iterative approach of multiplying n times is not optimized

so we need to find a way to compute half of the calculations

the basic idea is to divide up exponents: 
- 2^4 = 2^2 * 2^2 = 2^1 * 2^1 * 2^1 * 2^1
- 2^5 = 2^1 * 2^4 = ...

if it's even, then divide by 2, if it's odd we should 'remove' 1 and then divide by 2 the remaining part

thanks to the halving, we can memoize the computations efficiently

~~~py
class Solution:
    def myPow(self, x: float, n: int) -> float:
        isExpNegative = n < 0
        n = abs(n)
        expDict = defaultdict(int)

        def dfs(exp) -> float:
            if exp == 0:
                return 1
            
            if exp == 1:
                return x
            
            if exp in expDict:
                return expDict[exp]
            
            # (x if exp % 2 == 1 else 1) is for dividing up odd number 
            expDict[exp] = dfs(exp // 2) * dfs(exp // 2) * (x if exp % 2 == 1 else 1)

            return expDict[exp]
        
        return 1/dfs(n) if isExpNegative else dfs(n)
~~~

**complexity**
~~~
time = O(logN) 
~~~
we cut by 2 the computation eachtime

~~~
space = O(logN) 
~~~
the dictionary stores only 'intermediate' results thanks to the halving

## [56. Merge Intervals](https://leetcode.com/problems/merge-intervals)

## key idea
first sort the intervals by the first value and put the first interval inside res

res is gonna manage the merged intervals (and it's gonna be the final result obv)

when to merge an interval? if the previous interval END is grater or equal than the current interval START, really simple

~~~py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 1:
            return intervals
        
        intervals.sort()

        res = [intervals[0]]

        for start, end in intervals:
            if res[-1][1] >= start:
                res[-1][1] = max(res[-1][1], end)
            else:
                res.append([start, end])
            
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through all the intervals

~~~
space = O(N) 
~~~
in the worst case all the intervals are disjoint (no overlap) so |res| = |intervals|

## [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array)

## key idea
use 3 pointers, that points:

- at the last position of nums1 (last)
- at the last element of nums1 (p1)
- at the last element of nums2 (p2)

now it consists of comparing p1 and p2, whoever is the largest, just put it in the last position and decrement the pointers accordingly

there is one edge case where p1 becomes 0 but there are still p2 elements remaining in nums2, in that case pour all of them in nums1

~~~py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        last = m + n - 1
        p1 = m - 1
        p2 = n - 1

        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[last] = nums2[p2]
                p2 -= 1
            else:
                nums1[last] = nums1[p1]
                p1 -= 1
            last -= 1
        
        # edge case: if there are remaining elements in nums2
        while p2 >= 0:
            nums1[last] = nums2[p2]
            p2 -= 1
            last -= 1
~~~

**complexity**
~~~
time = O(N + M) 
~~~
we go through both of the arrays once

~~~
space = O(1) 
~~~
we do the operations in place without extra memory

## [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers)

### key idea

really simple just read the code

~~~py
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node, num):
            if not node:
                return 0
            
            num = num * 10 + node.val
            if not node.left and not node.right:
                return num
            
            return dfs(node.left, num) + dfs(node.right, num)
        
        return dfs(root, 0)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree

~~~
space = O(1) or O(height) if recursive stack is counted
~~~

## [133. Clone Graph](https://leetcode.com/problems/clone-graph)

### key idea

use an hashmap to keep the old_node : cloned_node mapping

do a bfs, clone the nodes and remember to populate neighbors array for each clone

~~~py
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None

        cloned = {}
        cloned[node] = Node(node.val, [])

        q = deque([node])

        while q:
            cur = q.popleft()

            for neigh in cur.neighbors:
                if neigh not in cloned:
                    cloned[neigh] = Node(neigh.val, [])
                    q.append(neigh)
                
                cloned[cur].neighbors.append(cloned[neigh])

        return cloned[node]
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole graph

~~~
space = O(N) 
~~~
hashmap size, storing all the nodes of the graph

## [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer)

### key idea

there is a simple way that involves an hash map but that would take O(n) space

this approac ensures O(1) space thanks to the interleaving of the nodes

basically we create a specular copy / clone of each node as their next

then we assign the random pointers to the cloned nodes thanks to the original ones

at the end we skip the original nodes by skipping the original ones, and thanks to this trick we'll have a deep copy of the original list due to different references for each node

~~~py
class Solution:
  def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: 
            return head

        # create cloned node as the next of the respective original node
        cur = head
        while cur:
            newnode = Node(cur.val, cur.next)
            cur.next = newnode
            cur = newnode.next
        
        # assign the random original node pointer to the cloned node
        cur = head
        while cur:
            cur.next.random = cur.random.next if cur.random else None
            cur = cur.next.next
        
        # skip original nodes to return only the cloned nodes
        cur = head.next
        while :
            cur.next = cur.next.next if cur.next else None
            cur = cur.next
        
        return head.next
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole list

~~~
space = O(1) 
~~~
no extra space used thanks to interleaving nodes


## [146. LRU Cache](https://leetcode.com/problems/lru-cache/description)

### key idea

the intuition is to have an hash map that stores these key value pairs

the value is not actually the value but is a pointer to a node that stores key and value

thanks to this we can use pointers to handle the lru mechanism

double linked list

left and right pointers (left = LRU, right = MRU)

watch neetcode video in resources, his explanation is really good

~~~py
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self.prev = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.kv = {}
        self.left, self.right = Node(0,0), Node(0,0)
        self.left.next, self.right.prev = self.right, self.left
    
    # insert to right (MRU, Most Recently Used)
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = node
        nxt.prev = node
        node.next = nxt
        node.prev = prev
    
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def get(self, key: int) -> int:
        if key in self.kv:
            self.remove(self.kv[key])
            self.insert(self.kv[key])
            return self.kv[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        # to update if it already exists (put it on most right position)
        if key in self.kv:
            self.remove(self.kv[key])
        
        self.kv[key] = Node(key, value)
        self.insert(self.kv[key])

        # eviction of LRU
        if len(self.kv) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.kv[lru.key]
~~~

**complexity**

~~~
time = O(1) 
~~~
no iterations or expensive operations

~~~
space = O(N) 
~~~
the hash map

### resources

- [neetcode video](https://www.youtube.com/watch?v=7ABFKPK2hD4)


## [162. Find Peak Element](https://leetcode.com/problems/find-peak-element)

### key idea
just compare the mid value, literally a binary search with some alterations

remember to put the right/left to -inf if needed (mid - 1 or mid + 1 out of bound)

~~~py
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        if len(nums) == 2:
            return 0 if nums[0] > nums[1] else 1
        
        l,r = 0, len(nums) - 1
        
        while l <= r:
            mid = (l + r) // 2

            mid_val = nums[mid]
            right_val = nums[mid + 1] if mid < len(nums) - 1 else float(-inf)
            left_val = nums[mid - 1] if mid > 0 else float(-inf)

            if left_val < mid_val > right_val:
                return mid
            elif mid_val < right_val:
                l = mid + 1
            else:
                r = mid - 1
        
        return 0
~~~

**complexity**
~~~
time = O(logN) 
~~~
as the problem statement requested, we half it each time

~~~
space = O(1) 
~~~
no extra space is used

## [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/description)

### key idea

really simple, since you need to return the right most value, we leverage the concept of the BFS to process the whole level and as soon as we arrive at the last element of the processed level, then we append it to the res

~~~py
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        queue = deque([root])
        res = []

        while queue:
            cur_level_length = len(queue)

            for i in range(cur_level_length):
                node = queue.popleft()

                if i == cur_level_length - 1:
                    res.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
since we touch every single node of the tree due to bfs 

~~~
space = O(N) 
~~~
it depends on the height of the tree, but in the worst case (complete tree) we gonna store n elements

## [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

### key idea

here if you do some examples, you can see that:
- this is a graph problem
- you just need to check for cycles

so construct a graph, do a dfs to check for cycles and that's it

~~~py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)

        # 0 = white, 1 = grey, 2 = black
        visited = [0] * (numCourses)

        # graph construction
        for src, dst in prerequisites:
            graph[src].append(dst)

        # dfs to check for cycles
        def courseCheck(node, check):
            visited[node] = 1

            for adj in graph[node]:
                if visited[adj] == 1:
                    return False

                if visited[adj] == 0:
                    check = courseCheck(adj, check)
                    if check == False:
                        return False
                
            visited[node] = 2
            return check
        
        # dfs every source
        vertices = list(graph.keys())
        for v in vertices:
            if visited[v] == 0:
                check = courseCheck(v, True)
                if check == False:
                    return False

        return True
~~~

**complexity**
~~~
time = O(V + E) 
~~~
typical graph dfs complexity, we go through all edges and pass through all vertices in the worst case -> we traverse the whole graph

~~~
space = O(V) 
~~~
graph size, number of vertices

## [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description)

### key idea
minheap and whenever the minheap size reaches k, pop from minheap (so it will pop the smallest element each time)

at the end in the first position we will have the smallest element of k-sized array, that is the k largest element

the same applies to maxheap

if you want it to be optimized more than the heap approach, a *quickselect* is needed

**min heap:**
~~~py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []

        for n in nums:
            heapq.heappush(heap, n)

            if len(heap) > k:
                heapq.heappop(heap)

        return heap[0]
~~~

**max heap:**
~~~py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = [-num for num in nums]

        heapq.heapify(heap)

        for _ in range(k - 1):
            heapq.heappop(heap)

        return -heap[0]
~~~

**complexity**
~~~
time = O(N + K logn) 
~~~
n = to turn the array in a heap and

k + logn = logn for each pop operation (to heapify) and k pop operations

~~~
space = O(N) 
~~~
for the heap size

### optimization

quickselect (TO DO)

## [71. Simplify Path](https://leetcode.com/problems/simplify-path)

### key idea
the only relevant special character for this problem is ".."

use a stack for storing words, whenever ".." is encountered if there are words in the stack, pop

for "" and "." just continue the iteration cause we just skip them

at the end we join all the words from the stack by separating them with "/"

~~~py
class Solution:
    def simplifyPath(self, path: str) -> str:
        words_stk = []
        items = path.split("/")

        for item in items:
            if item == "" or item == ".":
                continue
            
            if item == "..":
                if words_stk:
                    words_stk.pop()
            else:
                words_stk.append(item)
        
        return "/" + "/".join(words_stk)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the path of size n

~~~
space = O(N) 
~~~
we store words from the path in the stack

## [227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/description)

### key idea
not using a stack due to extra space complexity

doing the operation iteratively, from left to right by keeping track of the previous and current number

as soon as we encounter * or /, we need to undo the previous + or - operation since the order of operations is / -> * -> +/-

remember to parse the numbers > 10 by doing the usual while loop

~~~py
class Solution:
    def calculate(self, s: str) -> int:
        cur = prev = res = 0
        op = "+"

        i = 0
        while i < len(s):
            char = s[i]
            if char.isdigit():
                # parsing numbers
                while i < len(s) and s[i].isdigit(): 
                    cur = cur * 10 + int(s[i])
                    i += 1
                i -= 1

                if op == "+":
                    res += cur
                    prev = cur
                elif op == "-":
                    res -= cur
                    prev = -cur
                elif op == "*":
                    res -= prev

                    res += prev * cur
                    prev = cur * prev
                elif op == "/":
                    res -= prev

                    # python has problems with negative numbers
                    #   so // will not work correctly
                    res += int(prev/cur)
                    prev = int(prev/cur)

                cur = 0
            elif char != " ":
                op = char

            i += 1
            
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole string

~~~
space = O(1) 
~~~
no extra space, we happy

## [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description)

### key idea
if the left OR right subtree returns null, then it means that both p and q are in the subtree that didn't return null, and the node returned is the LCA

otherwise if both subtree return null, it means that the LCA is the parent of the left and right subtree

~~~py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        if root == p or root == q:
            return root
        
        l = self.lowestCommonAncestor(root.left, p, q) 
        r = self.lowestCommonAncestor(root.right, p, q) 

        if l and r:
            return root
        else:
            return l or r
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case it checks the whole tree till the leaf nodes

~~~
space = O(N) 
~~~
n elements in the tree, so the size of the recursive stack is n, otherwise O(1)

### optimization

if you want to optimize it further watch this [video by errichto](https://www.youtube.com/watch?v=dOAxrhAUIhA&list=PLl0KD3g-oDOEbtmoKT5UWZ-0_JbyLnHPZ&index=17)

## [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)

### key idea
- compute the prefix product (the product of all the elements preceding the current element)
- computer the suffix product (the product of all the elements following the current element)
- the product of these two arrays results in an array where each element is the product of all elements except itself

~~~py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prefix = [1] * (len(nums) + 1)
        suffix = [1] * (len(nums) + 1)
        res = [0] * len(nums)

        for i in range(len(nums)):
            prefix[i + 1] = prefix[i] * nums[i]

        for i in range(len(nums) - 1, 0, -1):
            suffix[i - 1] = suffix[i] * nums[i]

        for i in range(len(nums)):
            res[i] = prefix[i] * suffix[i] 

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
since we compute prefix, suffix and res by only going through all the elements once each iteration

~~~
space = O(N) 
~~~
since the auxiliary arrays (prefix and suffix) are of size n + 1

### optimization

the solution above could be simplified by 'merging' together the prefix and suffix arrays, so by doing the operation in place, but focus on the first solution for a true understanding of the problem since the optimized one is less obvious

~~~py
class Solution:
    # example: [1, 2, 3, 4]
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix = prefix * nums[i]

        # print(res) -> [1, 1, 2, 6]

        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            # [6 * 1, 2 * 4, 1 * 12, 1 * 24] in reverse since
            #    the loop is decrementing
            res[i] = res[i] * suffix

            # 1 -> 4 -> 12 -> 24 -> 24
            suffix = suffix * nums[i]

        # print(res) -> [24, 12, 8, 6] 

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
like before

~~~
space = O(1) 
~~~
since there are no auxiliary arrays except the result array

### useful resources

[prefix sum video by errichto](https://www.youtube.com/watch?v=bNvIQI2wAjk)

## 314. Binary Tree Vertical Order Traversal [(premium)](https://leetcode.com/problems/binary-tree-vertical-order-traversal/description) | [('premium')](https://www.lintcode.com/problem/651/)

### key idea
each node has a column associated to it and based on the columns the node values must be put in an hashmap structured this way column (key) : node (value)

there is a simple calculation to do with columns

- go left -> column - 1
- go right -> column + 1

the hashmap must be populated using **bfs** and then sorted otherwise with dfs you wont have the correct order

~~~py
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        columns = defaultdict(list)
        queue = deque([(0, root)])

        while queue:
            col, node = queue.popleft()

            columns[col].append(node.val)

            if node.left:
                queue.append((col - 1, node.left))

            if node.right:
                queue.append((col + 1, node.right))

        res = dict(sorted(columns.items()))
        return list(res.values())
~~~

### optimization

this little optimization eliminates the need to sort the hashmap, basically by defining a range of columns from *min_col* to *max_col*

~~~py
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        res = []
        hmap = defaultdict(list)
        queue = deque([(0, root)])
        
        min_col = float(inf)
        max_col = float(-inf)

        while queue:
            col, node = queue.popleft()

            hmap[col].append(node.val)

            min_col = min(min_col, col)
            max_col = max(max_col, col)

            if node.left:
                queue.append((col - 1, node.left))
            if node.right:
                queue.append((col + 1, node.right))
        
        for col in range(min_col, max_col + 1):
            res.append(hmap[col])

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
n = height of the tree in the worst case

~~~
space = O(N) 
~~~
because we are only storing values of the tree in the hashmap/result and there are n values

## 339. Nested List Weight Sum [(premium)](https://leetcode.com/problems/nested-list-weight-sum/description)

### key idea
the difficulty of this problem is due to the interface implementation, because the algorithm is really straightforward

here it's the simple dfs solution

~~~py
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def dfs(nested_list, depth) -> int:
            total = 0

            for nested in nested_list:
                if nested.isInteger():
                    total += nested.getInteger() * depth
                else:
                    total += dfs(nested.getList(), depth + 1)
            return total
            
        return dfs(nestedList, 1)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole list

~~~
space = O(N) 
~~~
recursive call stack worst case '[[[[[[]]]]]]'

### alternative (iterative) RECOMMENDED

if the interviewer alter the question by changing the constraint of the depth, a stack overflow might occur with the recursive approach (dfs), so its good to have in mind the iterative solution (bfs) as well

the key idea here is to spread out the list at the end of the queue each time one is encountered, so after every iteration we can go one level deeper

~~~py
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        depth = 1
        res = 0
        # spread out the whole list in the queue (the first level) for the bfs
        queue = deque(nestedList)

        while queue:
            for _ in range(len(queue)):
                item = queue.popleft()
                if item.isInteger():
                    res += item.getInteger() * depth
                else:
                    queue.extend(item.getList())
            
            depth += 1
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
same as dfs

~~~
space = O(N) 
~~~
same as dfs

## 346. Moving Average from Data Stream [(premium)](https://leetcode.com/problems/moving-average-from-data-stream) | [('premium')](https://www.lintcode.com/problem/642/)

### key idea

the naive solution is with a queue, but using a circular array is slightly better because:

- moving pointers, automatically discard the 'out of bound' elements, while with the queue we should do a deque operation each time
- we only need to store the pointer that points to the head and not on both side like the queue (head for popleft and tail for append)


so that's it basically, just remember to keep the size bounded to the requested size of the data stream

~~~py
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.circular_array = [0] * self.size
        self.head = 0
        self.count = 0
        self.res = 0

    def next(self, val: int) -> float:
        self.count += 1

        tail = (self.head + 1) % self.size
        self.res = self.res - self.circular_array[tail] + val

        self.head = (self.head + 1) % self.size
        self.circular_array[self.head] = val

        return self.res / min(self.size, self.count)
~~~

**complexity**
~~~
time = O(1) 
~~~
every call of next does only constant time operations

~~~
space = O(N) 
~~~
size of circular array

## [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements)

### key idea
frequency count with hash map

since heap would lead us to O(k*logn) complexity, we use a slight variation of a technique called *bucket sort*

we construct an array based on the hash map that has the frequency count as index and a list of variables (with the same frequency) as a value

then we iterate the frequency array from right to left, to find the k top elements

~~~py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = defaultdict(int) 
        freq = [[] for i in range(len(nums) + 1)]
        res = []

        # count numbers frequency
        for n in nums:
            count[n] += 1
        
        # construct array with count as indices and a list of number as value
        for num, cnt in count.items():
            freq[cnt].append(num)

        # iterate from left to right for k elements
        for i in range(len(freq) - 1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through the whole array nums + construct the hash map (both O(n) operations)

~~~
space = O(N) 
~~~
for the hash map/freq array

## [398. Random Pick Index](https://leetcode.com/problems/random-pick-index/description)

### key idea

reservoir sampling

this technique let us choose a random index without using extra space complexity

~~~py
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        count = pick_index = 0

        for i, n in enumerate(self.nums):
            if n == target:
                count += 1

                if random.randint(1, count) == count:
                    pick_index = i
        return pick_index
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through all nums

~~~
space = O(1) 
~~~
no extra memory used

### resources

- [reservoir sampling article](https://florian.github.io/reservoir-sampling/)

## 426. Convert Binary Search Tree to Sorted Doubly Linked List [(premium)](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/description/) | [('premium')](https://www.lintcode.com/problem/1534/)

### key idea

in-order traversal

keep track of the first (so the most left value) and last (every return we modify this last, always in-order)

while doing the recursive calls, as operations, we need to modify the left (predecessor) and right (successor) pointers

at the end remember to link first and last

~~~py
class Solution:
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None

        first = None
        last = None

        def dfs(node):
            if not node:
                return 
            
            dfs(node.left)

            nonlocal first
            nonlocal last

            if not last:
                first = node
            else:
                last.right = node
                node.left = last

            last = node

            dfs(node.right)

        dfs(root)

        first.left = last
        last.right = first

        return first
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree where n is the tree size

~~~
space = O(N) 
~~~
O(logN) if the tree is balanced, but in the worst case the tree is not balanced, so N is the size of recursive call stack

## [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

### key idea

hasmap and prefix sum

~~~
prefix[j] % k = prefix[i] % k  (with i < j)
~~~

store in the hashmap the reminder (key) and index (value), as soon a reminder is found, it means that two prefix sums have the same reminder and the last thing to check is if the length is greater than 1

example: 

[23, 2, 4, 3, 3] with k = 6

- 23 % 6 = 5
- 29 % 6 = 5
- 35 % 6 = 5

it means that all three make up a subarray of length 3 where the sum of all elements is a multiple of k, because we'll have:
- 23 - 29 = 6 (6 is a multiple of k)
- 23 - 35 = 12 (12 is a multiple of k)
- and obviously 29 - 35 = 6 (6 is a multiple of k)

~~~py
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        hmap = defaultdict(int)
        prefix_sum = 0
        hmap[0] = -1
        
        for i in range(len(nums)):
            prefix_sum += nums[i]

            if prefix_sum % k in hmap:
                if i - hmap[prefix_sum % k] > 1:
                    return True
            else:
                hmap[prefix_sum % k] = i
        
        return False
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate only once

~~~
space = O(N) 
~~~
hash map size

## [528. Random Pick with Weight]()

### key idea

this problem has a tricky intuition and the problem statement is written by monkeys

given an array of weights corresponding to their indexes, as the weight increases, the probabilty of randomly picking that specific weight's index also increases

basically, the problem is saying: "randomly pick an index but take its weight into account"

to do this we utilize the concept of a cumulative distribution function (prefix sum)

it allows us to get intervals, and these intervals simulate the weight for their specific index by considering their length.

'target = random.uniform(0, self.maxValue)' is used to get a random variable that is uniformly distributed, and thanks to this uniform distribution, we can say that the probability of this target being chosen is based on the length of the intervals created by the cdf

the solution is actually simple to remember

~~~py
class Solution:
    def __init__(self, w: List[int]):
        self.prefix_sum = []

        total = 0

        for weight in w:
            total += weight

            self.prefix_sum.append(total)

        self.maxValue = 0

    def pickIndex(self) -> int:
        target = random.uniform(0, self.maxValue)

        l = 0
        r = len(self.prefix_sum)

        while l < r:
            mid = (l + r) // 2

            if self.prefix_sum[mid] < target:
                l = mid + 1
            else:
                r = mid
        
        return l
~~~

**complexity**
~~~
time = init: O(N) - pickIndex: O(logN)
~~~

~~~
space = init: O(N) - pickIndex: O(1)
~~~

### resources

- [video explanation](https://www.youtube.com/watch?v=7x7Ydq2Wfvw)

## [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree)

### key idea
the thing to remember for this problem that i always forget is that you need to add 1 at the end (to count edges) and not for every dfs call (to count nodes)

plus the diameter could be:

- diameter = current node left path + current node right path
- diameter = curret node left OR right path + old left OR right path

that's why we need a nonlocal variable, since the diameter for some cases cannot be calculated solely from the current node but requires information from previous dfs calls as well

~~~py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        diameter = 0

        def dfs(node) -> int:
            if not node:
                return 0
            nonlocal diameter

            leftPath = dfs(node.left)
            rightPath = dfs(node.right) 

            diameter = max(diameter, leftPath + rightPath)
            
            return max(leftPath, rightPath) + 1
            
        dfs(root)
        return diameter
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through each node once with the dfs

~~~
space = O(N) 
~~~
the call stack is based on the height of the tree and in the worst case it could be O(N)

## [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k)

### key idea

we use prefix sum because the difference between two prefix sums gives us the sum of the subarray between those two indices:

~~~
prefix_sum_i - prefix_sum_j = subarray_sum_ij = k
~~~

let's use this concept to find efficiently how many subarray will give k as our result:

~~~
prefixsum_i - k = prefixsum_j where i > j
~~~

we can calculate prefix sums and count those using an hashmap

then whenever 'prefixsum - k' is in the hashmap, it means that we have found a new subarray

easy

~~~py
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_dict = defaultdict(int)
        prefix_sum = res = 0

        # this is important for some edge cases to count the very first subarray
        prefix_dict[0] = 1

        for num in nums:
            prefix_sum += num

            if prefix_sum - k in prefix_dict:
                res += prefix_dict[prefix_sum - k]
            prefix_dict[prefix_sum] += 1
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate the whole array nums

~~~
space = O(N) 
~~~
for the hashmap

## 637. Valid Word Abbreviation [(premium)](https://leetcode.com/problems/valid-word-abbreviation/description) | [('premium')](https://www.lintcode.com/problem/637/)

### key idea
the idea is simple, two pointers, *word_ptr* and *abbr_ptr*

whenever a number is encountered in the abbr, process the steps based on the number and increment the word_ptr by these steps

but the execution is BULLSHIT, too many edge cases and not an easy question

~~~py
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        word_ptr, abbr_ptr = 0, 0

        while word_ptr < len(word) and abbr_ptr < len(abbr):
            if abbr[abbr_ptr].isdigit():
                if abbr[abbr_ptr] == "0":
                    return False
                
                step = 0
                while abbr_ptr < len(abbr) and abbr[abbr_ptr].isdigit():
                    step = step * 10 + int(abbr[abbr_ptr])
                    abbr_ptr += 1
                
                word_ptr += step
            else:
                if word[word_ptr] != abbr[abbr_ptr]:
                    return False
                
                word_ptr += 1
                abbr_ptr += 1
            
        return word_ptr == len(word) and abbr_ptr == len(abbr)
~~~

**complexity**
~~~
time = O(N) 
~~~
both word and abbr are of size n

~~~
space = O(1) 
~~~
we only use pointers

## [670. Maximum Swap](https://leetcode.com/problems/maximum-swap)

### key idea

the intuition is simple, just find the leftmost smallest value with on his right the rightmost largest value

the code is a little tricky but it's fun because we are parsing back and forth the number

~~~py
class Solution:
    def maximumSwap(self, num: int) -> int:
        if num <= 11:
            return num
        
        # convert the integer number in an array
        num_as_arr = deque()
        while num:
            num_as_arr.appendleft(num % 10)
            num //= 10
        
        # get the rightmost largest value for each number
        max_seen, max_seen_at = -1, len(num_as_arr)
        for i in range(len(num_as_arr) - 1, -1, -1):
            cur_num = num_as_arr[i]
            num_as_arr[i] = (cur_num, max_seen, max_seen_at)

            if cur_num > max_seen:
                max_seen = cur_num
                max_seen_at = i
        
        # find the first smallest value and swap it with the rightmost largest value
        for i in range(len(num_as_arr)):
            cur_num, max_seen, max_seen_at = num_as_arr[i]

            if cur_num < max_seen:
                num_as_arr[i], num_as_arr[max_seen_at] = num_as_arr[max_seen_at], num_as_arr[i]
                break
        
        # convert the array back into an integer
        num = 0
        for n, _, _ in num_as_arr:
            num = num * 10 + n
        
        return num
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the number, value by value

~~~
space = O(N) 
~~~
number array size


## [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii)

### key idea
*two pointers*

whenever one of both sides is not equal, check if by removing one of the two sides it will become palindrome, by comparing the string without the left character and the string without the right character using its reverse (the code is self-explanatory with some python knowledge)

**remember** to check both possibilites, so skip left, check, skip right, check

~~~py
class Solution:
    def validPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while l < r:
            if s[l] != s[r]:
                skipLeft, skipRight = s[l + 1:r + 1], s[l:r]

                # palindrome check with reverse technique
                # if both are false it means that even with deleting the character
                #   the string is still not palindrome
                return (skipLeft == skipLeft[::-1]) or (skipRight == skipRight[::-1])
            
            l += 1
            r -= 1

        return True
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string

~~~
space = O(N) 
~~~
for skipLeft/skipRight

## optimization

we can optimize this problem by eliminating skipLeft/skipRight and reversing

by using only pointers

## [791. Custom Sort String](https://leetcode.com/problems/custom-sort-string)

### key idea

hashmap that counts the chars in s

for each char in order, if it's also present in the hashmap build the string with n times that char

at the end the string s could be left with some extra characters not present in order, so iterate through the remaining char in the hashmap and construct the final result

the code is pretty self explanatory

~~~py
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        counter = collections.Counter(s)
        res = []

        for c in order:
            if c in counter:
                res.extend([c] * counter[c])
                del counter[c]
        
        for c, n in counter.items():
                res.extend([c] * n)
        
        return "".join(res)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string + counter operation

~~~
space = O(N) 
~~~
counter size

## [827. Making A Large Island](https://leetcode.com/problems/making-a-large-island)

### key idea

first calculate the area of connected 1 islands and mark each island with a unique island_id

these islands will be store in an hashmap

after this, we go through all the 0s and check for surrounding islands, if there is a surrounding island then add its area (using the hashmap and island_id)

after all 0s are computed, we've found the largest island with at most one 0

~~~py
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        island_id = -1
        island_areas = {}
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def compute_area(i, j):
            if (0 <= i < len(grid)) and (0 <= j < len(grid[0])) and grid[i][j] == 1:
                grid[i][j] = island_id

                area = 1
                for r_inc, c_inc in directions:
                    new_i = i + i_inc
                    new_j = j + j_inc
                    area += compute_area(new_i, new_j)
                
                return area
            else:
                return 0

        # compute the areas of all islands and assign them an id
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    area = compute_area(i, j)

                    island_areas[island_id] = area
                    island_id -= 1
        
        max_area = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    area = 1

                    # build the surrounding
                    surrounding = set()
                    for i_inc, j_inc in directions:
                        new_i = i + i_inc
                        new_j = j + j_inc

                        if (0 <= new_i < len(grid)) and (0 <= new_j < len(grid[0]) and grid[new_i][new_j] != 0):
                            surrounding.add(grid[new_i][new_j])
                     
                    # use the surrounding of the 0 to compute the whole area
                    for island_id in surrounding:
                        area += island_areas[island_id]
                    
                    max_area = max(max_area, area)
        
        # there could happen a case where the whole grid is 1
        return max_area if max_area else len(grid) ** 2
~~~

**complexity**
~~~
time = O(N^2) 
~~~
N * N size of the matrix, so it's the iterations through all the matrix

~~~
space = O(N^2 / 2) 
~~~
this is the worst case if the matrix is half 0 and half 1

## [921. Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description)

### key idea

check for extra right parentheses and whenever one is found increment a counter that counts how many open parentheses do we need to add

~~~py
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        left_count = right_count = added = 0

        for c in s:
            if c == "(":
                left_count += 1
            else:
                if right_count < left_count:
                    right_count += 1
                else:
                    added += 1
        
        added += left_count - right_count
        
        return added
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string

~~~
space = O(1) 
~~~
only counters, no extra space used

## [938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst)

### key idea
whenever you are out of range, return accordingly

the return is used to go right or left without computing the sum at the end, so it's a sort of roadblock

we compute the sum only if we are inside the range

~~~py
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        def dfs(node) -> int:
            if not node:
                return 0
            
            if node.val < low:
                return dfs(node.right)
                
            if node.val > high:
                return dfs(node.left)

            return node.val + dfs(node.left) + dfs(node.right)
        
        return dfs(root)
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case we'll visit all the tree (if low and high are at the edge of the tree)

~~~
space = O(N) 
~~~
recursive call stack size since in the worst case we recurse n times (how many nodes there are)

## [973. K Closest Points to Origin]()

### key idea
use a min heap, the closest points to the origin are the one at the start of the min heap

~~~py
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        minHeap = []

        for point in points:
            dist = point[0] ** 2 + point[1] ** 2
            heapq.heappush(minHeap, (dist, point))
        
        res = []
        for _ in range(k):
            res.append(heapq.heappop(minHeap)[1])
        
        return res
~~~

**complexity**
~~~
time = O(N log N + K log N) 
~~~
build the heap and process every point

~~~
space = O(N) 
~~~
heap takes n points

## [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/description)

### key idea

two pointers

look at example diagram / draw it yourself, then try to come up with all the possibles outcomes and code them out

~~~py
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        if not firstList or not secondList:
            return []
            
        p1, p2 = 0, 0
        res = []

        while p1 < len(firstList) and p2 < len(secondList):
            start1, start2 = firstList[p1][0], secondList[p2][0]
            end1, end2 = firstList[p1][1], secondList[p2][1]

            if start1 > end2: # check if they are disjoint
                p2 += 1
            elif start2 > end1:
                p1 += 1
            else: # if they are overlapping
                res.append([max(start1, start2), min(end1, end2)])

                if end1 > end2: # check if there is some processing left
                    p2 += 1
                else:
                    p1 += 1
        return res
~~~

**complexity**
~~~
time = O(N + M)
~~~
if both l1.length = N and l2.length = M are of the same size

~~~
space = O(1) or O(N) if considering the res array
~~~

## [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii)

### key idea

greedy sliding window approach

if we encounter a 0, we decrement k

if k goes negative, that means that we took too many 0s, so we must move the left pointer up and we need to retake one 0 each time

check the maximum of all the results

~~~py
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        l = res = 0

        for r, num in enumerate(nums):
            # if num == 0:
            #     k -= 1
            k -= 1 - num
            
            if k < 0:
                # if nums[l] == 0:
                #   k += 1
                k += 1 - nums[l]
                
                l += 1
            else:
                res = max(res, (r - l) + 1)

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
only one iteration

~~~
space = O(1) 
~~~
only pointers, no extra space used

## [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix)

### key idea

bfs to find the shortest path

remember that you can go diagonally so there are 8 possible directions

the bfs queue stores (row, column, length) and as soon row and column reach N - 1, return the length

~~~py
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        queue = deque([(0,0,1)])
        directions = [[1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [1,-1], [-1,-1]]
        visited = set((0,0))

        while queue:
            r, c, length = queue.popleft()

            if (min(r, c) < 0 or max(r, c) == len(grid) or grid[r][c] == 1):
                continue
            
            if r == len(grid) - 1 and c == len(grid) - 1:
                return length
            
            for row_dir, col_dir in directions:
                new_row = r + row_dir
                new_col = c + col_dir
                if (new_row, new_col) not in visited:
                    queue.append((new_row, new_col, length + 1))
                    visited.add((new_row, new_col))
        
        return -1
~~~

**complexity**
~~~
time = O(N * M) 
~~~
we visit all the 2d matrix

~~~
space = O(N * M) 
~~~
we might have the entire matrix in the queue in the worst case

## [1249. Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses)

### key idea
keep a count of open parentheses and whenever we encounter a closing parentheses we decrement the count

there is a little evil edgecase where there could be no closing parentheses but only open parentheses and that's why we have two iterations:

1) iteration to remove extra closing parentheses
2) iteration to remove the most right extra open parentheses (that's why we iterate the reversed array)

~~~py
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        res = []
        openCount = 0

        # skip extra closing parentheses
        for c in s:
            if c == "(":
                res.append(c)
                openCount += 1
            elif c == ")" and openCount > 0:
                res.append(c)
                openCount -= 1
            elif c != ")":
                res.append(c)
        
        filtered = []

        # skip extra most right open parentheses
        for c in res[::-1]:
            if c == "(" and openCount > 0:
                openCount -= 1
            else:
                filtered.append(c)
        
        return "".join(filtered[::-1])
~~~

**complexity**
~~~
time = O(N) 
~~~
two indipendent iterations that goes only through all the elements of the string

~~~
space = O(N) 
~~~
res/filtered size is at most n

## 1570. Dot Product of Two Sparse Vectors [(premium)](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/description) | [('premium')](https://www.lintcode.com/problem/3691/)

### key idea
there are actually 3 solutions:

- naive: store the entire array and do the 1 by 1 product
- hashmap: don't store the whole array by storing only the non-zero values as index -> value
- tuple array and two pointers: array of tuples (idx, val) with two pointers, one for each vector

the tuple array is the most efficient one since there could be problems if the hash functions sucks basically

really easy to implement

~~~py
class SparseVector:
    def __init__(self, nums: List[int]):
        self.tuple_arr = []

        for i, n in enumerate(nums):
            if n != 0:
                self.tuple_arr.append((i, n))

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        p1 = p2 = 0       

        while p1 < len(self.tuple_arr) and p2 < len(vec.tuple_arr):
            p1_idx, p1_val = self.tuple_arr[p1]
            p2_idx, p2_val = vec.tuple_arr[p2]

            if p1_idx == p2_idx:
                res += p1_val * p2_val
                p1 += 1
                p2 += 1
            else:
                if p1_idx > p2_idx:
                    p2 += 1
                else:
                    p1 += 1
        return res
~~~

**complexity**
~~~
time = O(N + M) 
~~~
in the worst case the idx wont match and a full iteration for both has to be done

~~~
space = O(N + M) 
~~~
we store two tuple arrays

## 1650. Lowest Common Ancestor of a Binary Tree III [(premium)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii)

### key idea
not the traditional solution explained in the tutorials, but i like this one

basically is SFS (Swim First Search)

based on the distance from the top, if the two distances are different, then the one that is lower need to swim till the one higher, as soon as they are at the same height then they swim together until the LCA

~~~py
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        def distFromTop(node) -> int:
            if not node:
                return -1
            return 1 + distFromTop(node.parent)
        
        pDistFromTop = distFromTop(p)
        qDistFromTop = distFromTop(q)

        while pDistFromTop > qDistFromTop:
            p = p.parent
            pDistFromTop -= 1
        while qDistFromTop > pDistFromTop:
            q = q.parent
            qDistFromTop -= 1
        
        while p != q:
            p = p.parent
            q = q.parent
        
        return p
~~~


**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree in the worst case

~~~
space = O(1) 
~~~
no extra space used, only iterations (if we don't count the call stack to go to the top)

## 1762. Buildings With an Ocean View [(premium)](https://leetcode.com/problems/buildings-with-an-ocean-view/description)

### key idea
easy but marked medium, just watch for the tallest building with a ptr, skip every smaller building and just insert in the result array the indexes of the higher buildings. the code is self explanatory

we use a deque to save time by appending at the left instead of reversing the array at the end

~~~py
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        if not heights:
            return []

        answer = deque([])
        max_height = -1

        for i in range(len(heights) - 1, -1, -1):
            cur = heights[i]

            if cur > max_height:
                answer.appendleft(i)

                max_height = cur

        return answer
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the buildings in the worst case

~~~
space = O(N) 
~~~
we need to store n elements in res array in the worst case

### alternative (from left to right)

it might happen that the interviewer will change the constraint for example you are only allowed to go from left to right, then use this solution with a stack

~~~py
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        n = len(heights)
        answer = []

        for cur_idx in range(n):
            while answer and heights[answer[-1]] <= heights[cur_idx]:
                answer.pop()
            answer.append(cur_idx)
        
        return answer
~~~

**complexity**
~~~
time = O(N) 
~~~
same as before

~~~
space = O(N) 
~~~
same as before
