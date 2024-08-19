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
- [46. Permutations | 77. Combinations | 78. Subsets](#46-permutations--77-combinations--78-subsets)
- [88. Merge Sorted Array](#88-merge-sorted-array)
- [215. Kth Largest Element in an Array](#215-kth-largest-element-in-an-array)
- [236. Lowest Common Ancestor of a Binary Tree](#236-lowest-common-ancestor-of-a-binary-tree)
- [238. Product of Array Except Self](#238-product-of-array-except-self)
- [314. Binary Tree Vertical Order Traversal](#314-binary-tree-vertical-order-traversal-premium--premium)
- [543. Diameter of Binary Tree](#543-diameter-of-binary-tree)
- [637. Valid Word Abbreviation](#637-valid-word-abbreviation-premium--premium)
- [680. Valid Palindrome II](#680-valid-palindrome-ii)
- [938. Range Sum of BST](#938-range-sum-of-bst)
- [1249. Minimum Remove to Make Valid Parentheses](#1249-minimum-remove-to-make-valid-parentheses)

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

## [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description)

### key idea
if the left OR right subtree returns null, then it means that both p and q are in the subtree that returned the node, and the node returned is the LCA

otherwise if both return null, it means that the LCA is the parent of the left and right subtree

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

        res = []
        hmap = defaultdict(list)
        queue = deque([(0, root)])

        while queue:
            col, node = queue.popleft()

            hmap[col].append(node.val)

            if node.left:
                queue.append((col - 1, node.left))
            if node.right:
                queue.append((col + 1, node.right))
        
        sortHmap = dict(sorted(hmap.items()))

        for item in sortHmap.values():
            res.append(item)

        return res
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

## 637. Valid Word Abbreviation [(premium)](https://leetcode.com/problems/valid-word-abbreviation/description)[('premium')](https://www.lintcode.com/problem/637/)

BULLSHIT

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

